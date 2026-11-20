use std::{any::TypeId, fmt::Debug};

use ff::Field;
use group::{Curve, Group};
use itertools::izip;
use midnight_curves::{
    msm::msm_best,
    pairing::{Engine, MillerLoopResult, MultiMillerLoop},
    CurveAffine, Fq, G1Projective,
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use super::params::ParamsVerifierKZG;
use crate::{
    poly::{
        commitment::{Guard, PolynomialCommitmentScheme},
        kzg::KZGCommitmentScheme,
        CommitmentLabel, Error,
    },
    utils::{
        arithmetic::{CurveExt, MSM},
        helpers::ProcessedSerdeObject,
    },
};

/// A multi-scalar multiplication in the polynomial commitment scheme.
/// For every i, term (bases_i, scalars_i) may be have an optional
/// label_i for debugging or other purposes.
#[derive(Clone, Default, Debug)]
pub struct MSMKZG<E: Engine> {
    pub(crate) scalars: Vec<E::Fr>,
    pub(crate) bases: Vec<E::G1>,
    pub(crate) labels: Vec<CommitmentLabel>,
}

impl<E: Engine> MSMKZG<E> {
    /// Create an empty MSM instance
    pub fn init() -> Self {
        MSMKZG {
            scalars: vec![],
            bases: vec![],
            labels: vec![],
        }
    }

    /// Create an MSM from various MSMs
    pub fn from_many(msms: Vec<Self>) -> Self {
        let len = msms.iter().map(|m| m.scalars.len()).sum();

        let mut scalars = Vec::with_capacity(len);
        let mut bases = Vec::with_capacity(len);
        let mut labels = Vec::with_capacity(len);

        for mut msm in msms {
            scalars.append(&mut msm.scalars);
            bases.append(&mut msm.bases);
            labels.append(&mut msm.labels);
        }

        Self {
            scalars,
            bases,
            labels,
        }
    }

    /// Create a new MSM from a given base (with scalar of 1).
    pub fn from_base(base: &E::G1) -> Self {
        MSMKZG {
            scalars: vec![E::Fr::ONE],
            bases: vec![*base],
            labels: vec![CommitmentLabel::NoLabel],
        }
    }
}

impl<E: Engine + Debug> MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Evaluates the MSM to a single point and replaces all terms with that
    /// single point (scalar = 1, label = `NoLabel`).
    ///
    /// This mirrors `AssignedMsm::collapse` in the circuits crate.
    ///
    /// # Panics (in debug mode)
    ///
    /// If any term carries a label other than `NoLabel` or `Advice`.
    //
    // We only allow `NoLabel` or `Advice` because these types of labels are
    // not relevant for the `verifier_gadget` in `midnight-circuits` (at least for
    // now). Other types of labels may carry information that we do not want to lose
    // when "collapsing".
    pub fn collapse(&mut self) {
        debug_assert!(
            self.labels
                .iter()
                .all(|l| matches!(l, CommitmentLabel::NoLabel | CommitmentLabel::Advice(_))),
            "collapse: all labels must be NoLabel or Advice, found: {:?}",
            self.labels,
        );
        let point = self.eval();
        self.scalars = vec![E::Fr::ONE];
        self.bases = vec![point];
        self.labels = vec![CommitmentLabel::NoLabel];
    }
}

impl<E: Engine + Debug> MSM<E::G1Affine> for MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn append_term(&mut self, scalar: E::Fr, point: E::G1, label: CommitmentLabel) {
        self.scalars.push(scalar);
        self.bases.push(point);
        self.labels.push(label);
    }

    fn add_msm(&mut self, other: &Self) {
        self.scalars.reserve(other.scalars().len());
        self.scalars.extend_from_slice(&other.scalars());

        self.bases.reserve(other.bases().len());
        self.bases.extend_from_slice(&other.bases());

        self.labels.reserve(other.labels().len());
        self.labels.extend_from_slice(&other.labels());
    }

    fn scale(&mut self, factor: E::Fr) {
        self.scalars.par_iter_mut().for_each(|s| {
            *s *= &factor;
        })
    }

    fn check(&self) -> bool {
        bool::from(self.eval().is_identity())
    }

    fn eval(&self) -> E::G1 {
        if self.scalars == vec![E::Fr::ONE] {
            self.bases[0]
        } else {
            msm_specific::<E::G1Affine>(&self.scalars, &self.bases)
        }
    }

    fn bases(&self) -> Vec<E::G1> {
        self.bases.clone()
    }

    fn scalars(&self) -> Vec<E::Fr> {
        self.scalars.clone()
    }

    fn labels(&self) -> Vec<CommitmentLabel> {
        self.labels.clone()
    }
}

#[allow(unsafe_code)]
/// Wrapper over the MSM function to use the blstrs underlying function
pub fn msm_specific<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    // We remove zeros (keep only non-zero coefficients)
    let (coeffs, bases): (Vec<C::Scalar>, Vec<C::Curve>) = coeffs
        .iter()
        .zip(bases)
        .filter(|(s, _)| !s.is_zero_vartime())
        .map(|(s, b)| (*s, *b))
        .unzip();

    if coeffs.is_empty() {
        return C::Curve::identity();
    }

    // For BLS12-381 G1 always use blst's Pippenger. The former cut-over to
    // `msm_best` above 2^19 terms ("blst regresses above 2^18") no longer holds,
    // and `msm_best` pays the `from_xy` subgroup check introduced in
    // midnight-curves 0.3.1 once per base and once per bucket, which makes it
    // several times slower on large MSMs.
    if TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>() {
        // Safe: we just checked type
        let coeffs_slice = coeffs.as_slice();
        let bases_slice = bases.as_slice();
        let coeffs = unsafe { &*(coeffs_slice as *const _ as *const [Fq]) };
        let bases = unsafe { &*(bases_slice as *const _ as *const [G1Projective]) };
        let res = G1Projective::multi_exp(bases, coeffs);
        unsafe { std::mem::transmute_copy(&res) }
    } else {
        let mut affine_bases = vec![C::identity(); coeffs.len()];
        C::Curve::batch_normalize(&bases, &mut affine_bases);
        msm_best(&coeffs, &affine_bases)
    }
}

#[allow(unsafe_code)]
/// The MSM of `coeffs` with the first `coeffs.len()` elements of the Lagrange
/// basis of `params`, for BLS12-381: the same group element as
/// [msm_specific] gives for them, computed with the basis converted to affine
/// form once per `params` instead of once per MSM, and with only as many bits
/// per scalar as the longest scalar has. `None` for other curves.
pub(crate) fn msm_lagrange_cached<E: Engine>(
    params: &super::params::ParamsKZG<E>,
    coeffs: &[E::Fr],
) -> Option<E::G1> {
    use blst::{blst_p1, blst_p1_affine, p1_affines, MultiPoint};
    use rayon::iter::IntoParallelRefIterator;

    if TypeId::of::<E::G1>() != TypeId::of::<G1Projective>()
        || TypeId::of::<E::Fr>() != TypeId::of::<Fq>()
    {
        return None;
    }
    assert!(coeffs.len() <= params.g_lagrange.len());
    // Safe: we just checked the types.
    let coeffs = unsafe { &*(coeffs as *const [E::Fr] as *const [Fq]) };
    let affine = params.g_lagrange_affine.get_or_init(|| {
        let bases = params.g_lagrange.as_slice();
        // Safe: E::G1 is G1Projective, a transparent wrapper around blst_p1.
        let bases =
            unsafe { std::slice::from_raw_parts(bases.as_ptr() as *const blst_p1, bases.len()) };
        p1_affines::from(bases).as_slice().into()
    });

    let bytes: Vec<[u8; 32]> = coeffs.par_iter().map(|s| s.to_bytes_le()).collect();
    // The MSM only has to cover the bits of the longest scalar, which is far
    // fewer than 255 for selectors, tables and constants.
    let bit_len = |b: &[u8; 32]| {
        b.iter()
            .rposition(|x| *x != 0)
            .map_or(0, |i| i * 8 + 8 - b[i].leading_zeros() as usize)
    };
    let nbits = bytes.par_iter().map(bit_len).max().unwrap_or(0);
    if nbits == 0 {
        return Some(E::G1::identity());
    }
    let nbytes = nbits.div_ceil(8);

    // As in [msm_specific], only the non-zero coefficients take part.
    let is_nonzero = |b: &[u8; 32]| *b != [0u8; 32];
    let n = bytes.par_iter().filter(|b| is_nonzero(b)).count();
    let mut scalars = Vec::with_capacity(n * nbytes);
    let res: blst_p1 = if n == coeffs.len() {
        bytes.iter().for_each(|b| scalars.extend_from_slice(&b[..nbytes]));
        affine[..coeffs.len()].mult(&scalars, nbits)
    } else {
        let mut points: Vec<blst_p1_affine> = Vec::with_capacity(n);
        for (b, p) in bytes.iter().zip(affine.iter()).filter(|(b, _)| is_nonzero(b)) {
            points.push(*p);
            scalars.extend_from_slice(&b[..nbytes]);
        }
        points.as_slice().mult(&scalars, nbits)
    };
    // Safe: E::G1 is G1Projective, a transparent wrapper around blst_p1.
    Some(unsafe { std::mem::transmute_copy(&res) })
}

/// Two channel MSM accumulator
#[derive(Debug, Clone)]
pub struct DualMSM<E: Engine> {
    pub(crate) left: MSMKZG<E>,
    pub(crate) right: MSMKZG<E>,
}

/// A [DualMSM] split into left and right vectors of `(Scalar, Point)` tuples
pub type SplitDualMSM<'a, E> = (
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
);

impl<E: MultiMillerLoop + Debug> Default for DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn default() -> Self {
        Self::init()
    }
}

impl<E: MultiMillerLoop> Guard<E::Fr, KZGCommitmentScheme<E>> for DualMSM<E>
where
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn verify(
        self,
        params: &<KZGCommitmentScheme<E> as PolynomialCommitmentScheme<E::Fr>>::VerifierParameters,
    ) -> Result<(), Error> {
        self.check(params).then_some(()).ok_or(Error::OpeningError)
    }
}

impl<E: MultiMillerLoop + Debug> DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Create an empty two channel MSM accumulator instance
    pub fn init() -> Self {
        Self {
            left: MSMKZG::init(),
            right: MSMKZG::init(),
        }
    }

    /// Create a new two channel MSM accumulator instance
    pub fn new(left: MSMKZG<E>, right: MSMKZG<E>) -> Self {
        Self { left, right }
    }

    /// Split the [DualMSM] into `left` and `right`
    pub fn split(&self) -> SplitDualMSM<'_, E> {
        let left = izip!(
            self.left.labels.iter(),
            self.left.scalars.iter(),
            self.left.bases.iter()
        )
        .collect();
        let right = izip!(
            self.right.labels.iter(),
            self.right.scalars.iter(),
            self.right.bases.iter(),
        )
        .collect();
        (left, right)
    }

    /// Scale all scalars in the MSM by some scaling factor
    pub fn scale(&mut self, e: E::Fr) {
        self.left.scale(e);
        self.right.scale(e);
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: Self) {
        self.left.add_msm(&other.left);
        self.right.add_msm(&other.right);
    }

    /// Performs final pairing check with given verifier params and two channel
    /// linear combination
    pub fn check(self, params: &ParamsVerifierKZG<E>) -> bool {
        let left = if self.left.scalars.len() == 1 && self.left.scalars[0] == E::Fr::ONE {
            self.left.bases[0]
        } else {
            self.left.eval()
        };

        let right = self.right.eval();

        let (term_1, term_2) = (
            (&left.into(), &params.s_g2_prepared),
            (&right.into(), &params.n_g2_prepared),
        );
        let terms = &[term_1, term_2];

        bool::from(E::multi_miller_loop(&terms[..]).final_exponentiation().is_identity())
    }
}
