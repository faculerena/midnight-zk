use std::{fmt::Debug, io};

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use midnight_curves::pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;

use crate::{
    poly::commitment::Params,
    utils::{
        arithmetic::{g_to_lagrange, parallelize},
        helpers::ProcessedSerdeObject,
        SerdeFormat,
    },
};

/// These are the public parameters for the polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct ParamsKZG<E: Engine> {
    pub(crate) g: Vec<E::G1>,
    pub(crate) g_lagrange: Vec<E::G1>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
    /// `g_lagrange` in the affine form the MSM backend works on, made on the
    /// first Lagrange commitment and reused by the following ones.
    pub(crate) g_lagrange_affine:
        std::sync::OnceLock<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// `g[1]`, that is `[tau]_1`, once it has been checked against the
    /// Lagrange basis and the G2 elements; `None` if the check failed. See
    /// [`ParamsKZG::checked_tau_g1`].
    pub(crate) tau_g1: std::sync::OnceLock<Option<E::G1>>,
}

impl<E: Engine> Params for ParamsKZG<E> {
    fn max_k(&self) -> u32 {
        assert_eq!(self.g.len(), self.g_lagrange.len());

        self.g.len().ilog2()
    }

    fn downsize(&mut self, new_k: u32) {
        ParamsKZG::<E>::downsize(self, new_k)
    }
}

impl<E: Engine + Debug> ParamsKZG<E> {
    /// Downsize the current parameters to match a smaller `k`.
    pub fn downsize(&mut self, new_k: u32) {
        if self.max_k() == new_k {
            return;
        }

        let n = 1 << new_k;
        assert!(n < self.g_lagrange.len());
        self.g.truncate(n);
        self.g_lagrange = g_to_lagrange(&self.g, new_k);
        self.g_lagrange_affine = Default::default();
        self.tau_g1 = Default::default();
    }

    /// Initializes parameters for the curve, draws toxic secret from given rng.
    /// MUST NOT be used in production.
    pub fn unsafe_setup<R: RngCore>(k: u32, rng: R) -> Self {
        // Largest root of unity exponent of the Engine is `2^E::Fr::S`, so we can
        // only support FFTs of polynomials below degree `2^E::Fr::S`.
        assert!(k <= E::Fr::S);
        let n: u64 = 1 << k;

        // Calculate g = [G1, [s] G1, [s^2] G1, ..., [s^(n-1)] G1] in parallel.
        let g1 = E::G1::generator();
        let s = <E::Fr>::random(rng);

        let mut g = vec![E::G1::identity(); n as usize];
        parallelize(&mut g, |g, start| {
            let mut current_g: E::G1 = g1;
            current_g *= s.pow_vartime([start as u64]);
            for g in g.iter_mut() {
                *g = current_g;
                current_g *= s;
            }
        });

        let mut g_lagrange = vec![E::G1::identity(); n as usize];
        let mut root = E::Fr::ROOT_OF_UNITY;
        for _ in k..E::Fr::S {
            root = root.square();
        }
        let n_inv = E::Fr::from(n).invert().expect("inversion should be ok for n = 1<<k");
        let multiplier = (s.pow_vartime([n]) - E::Fr::ONE) * n_inv;
        parallelize(&mut g_lagrange, |g, start| {
            for (idx, g) in g.iter_mut().enumerate() {
                let offset = start + idx;
                let root_pow = root.pow_vartime([offset as u64]);
                let scalar = multiplier * root_pow * (s - root_pow).invert().unwrap();
                *g = g1 * scalar;
            }
        });

        let g2 = E::G2::generator();
        let s_g2 = g2 * s;

        Self {
            g,
            g_lagrange,
            g2,
            s_g2,
            g_lagrange_affine: Default::default(),
            tau_g1: Default::default(),
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2
    pub fn from_parts(
        k: u32,
        g: Vec<E::G1>,
        g_lagrange: Option<Vec<E::G1>>,
        g2: E::G2,
        s_g2: E::G2,
    ) -> Self {
        Self {
            g_lagrange: match g_lagrange {
                Some(g_l) => g_l,
                None => g_to_lagrange(&g, k),
            },
            g,
            g2,
            s_g2,
            g_lagrange_affine: Default::default(),
            tau_g1: Default::default(),
        }
    }

    /// Returns the committed lagrange polynomials of these KZG params.
    pub fn g_lagrange(&self) -> &[E::G1] {
        &self.g_lagrange
    }

    /// Returns generator on G2
    pub fn g2(&self) -> E::G2 {
        self.g2
    }

    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        writer.write_all(&self.g.len().ilog2().to_le_bytes())?;
        for el in self.g.iter() {
            el.write(writer, format)?;
        }
        for el in self.g_lagrange.iter() {
            el.write(writer, format)?;
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<Option<E::G1>>> {
                        let mut points_compressed =
                            vec![<<E as Engine>::G1 as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }

                        let mut points = vec![Option::<E::G1>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point =
                                    Option::from(E::G1::from_bytes(&points_compressed[chunks + i]));
                            }
                        });
                        Ok(points)
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g: Vec<<E as Engine>::G1> = g
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                let g_lagrange: Vec<<E as Engine>::G1> = g_lagrange
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                // avoid try branching for performance
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        Ok(Self {
            g,
            g_lagrange,
            g2,
            s_g2,
            g_lagrange_affine: Default::default(),
            tau_g1: Default::default(),
        })
    }
}

// TODO: see the issue at https://github.com/appliedzkp/halo2/issues/45
// So we probably need much smaller verifier key. However for new bases in g1
// should be in verifier keys.
/// KZG multi-open verification parameters
#[derive(Clone, Debug)]
pub struct ParamsVerifierKZG<E: MultiMillerLoop> {
    pub(crate) s_g2: E::G2,
    pub(crate) n_g2_prepared: E::G2Prepared,
    pub(crate) s_g2_prepared: E::G2Prepared,
}

impl<E: MultiMillerLoop + Debug> ParamsVerifierKZG<E>
where
    E::G2: Curve + ProcessedSerdeObject,
{
    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let s_g2 = E::G2::read(reader, format)?;
        let s_g2_prepared = E::G2Prepared::from(s_g2.into());
        let n_g2_prepared = E::G2Prepared::from(-E::G2Affine::generator());

        Ok(Self {
            s_g2,
            n_g2_prepared,
            s_g2_prepared,
        })
    }
}

impl<E: MultiMillerLoop + Debug> ParamsKZG<E> {
    /// `[tau]_1`, the commitment to the polynomial `X`, taken from the
    /// monomial basis (`g[1]`) after checking that it is the element the
    /// Lagrange basis and the G2 elements stand for; `None` if it is not.
    ///
    /// Commitments to polynomials in Lagrange form otherwise read the
    /// Lagrange basis only, so without this check a corrupt monomial region
    /// of a params file would silently change whatever is built from this
    /// element. The check is done once per parameter set:
    ///
    /// 1. `g[0] == sum_j g_lagrange[j]`: the Lagrange polynomials add up to
    ///    the constant 1, so this ties `g[0]` to the Lagrange basis;
    /// 2. `e(g[1], g2) == e(g[0], s_g2)`: `g[1]` is `tau * g[0]` for the same
    ///    `tau` as the verifier's G2 element, the equation every KZG opening
    ///    proof made with the Lagrange basis has to satisfy.
    ///
    /// For a well-formed SRS (Lagrange basis and G2 elements of the same
    /// `tau`, which proving and verifying need anyway) these two hold exactly
    /// when `g[1] == sum_j omega^j * g_lagrange[j]`, the Lagrange commitment
    /// to `X`, and they cost n point additions and two pairings instead of a
    /// full-width multi-scalar multiplication.
    pub(crate) fn checked_tau_g1(&self) -> Option<E::G1>
    where
        E::G1: Send + Sync,
    {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        *self.tau_g1.get_or_init(|| {
            if self.g.len() < 2 {
                return None;
            }
            let (g0, g1) = (self.g[0], self.g[1]);
            let sum =
                self.g_lagrange.par_iter().copied().reduce(E::G1::identity, |a, b| a + b);
            if sum != g0 || bool::from(g0.is_identity()) {
                return None;
            }
            let lhs = E::pairing(&g1.into(), &self.g2.into());
            let rhs = E::pairing(&g0.into(), &self.s_g2.into());
            (lhs == rhs).then_some(g1)
        })
    }

    /// Consume the prover parameters into verifier parameters. Need to specify
    /// the size of public inputs.
    pub fn verifier_params(&self) -> ParamsVerifierKZG<E> {
        let n_g2_prepared = E::G2Prepared::from((-self.g2).into());
        let s_g2_prepared = E::G2Prepared::from(self.s_g2.into());
        ParamsVerifierKZG {
            s_g2: self.s_g2,
            n_g2_prepared,
            s_g2_prepared,
        }
    }
}

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use crate::{
        poly::{
            commitment::PolynomialCommitmentScheme,
            kzg::{params::ParamsKZG, KZGCommitmentScheme},
        },
        utils::SerdeFormat,
    };

    #[test]
    fn test_commit_lagrange() {
        const K: u32 = 6;

        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        let mut a = domain.empty_lagrange();

        for (i, a) in a.iter_mut().enumerate() {
            *a = Fq::from(i as u64);
        }

        let b = domain.lagrange_to_coeff(a.clone());

        let tmp = KZGCommitmentScheme::commit_lagrange(&params, &a);
        let commitment = KZGCommitmentScheme::commit(&params, &b);

        assert_eq!(commitment, tmp);
    }

    /// The commitment to `X` taken from the monomial basis is the Lagrange
    /// commitment to `X`, and a monomial basis, Lagrange basis or G2 element
    /// that does not agree with the rest is refused rather than used.
    #[test]
    fn test_commit_x_checks_the_monomial_basis() {
        use ff::Field;
        use group::Group;
        use midnight_curves::{Bls12, Fq, G1Projective, G2Projective};

        use crate::{plonk::Error, poly::EvaluationDomain};

        for k in [1u32, 2, 6, 9] {
            let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, OsRng);
            let domain = EvaluationDomain::<Fq>::new(1, k);
            let mut x = domain.empty_lagrange();
            let mut cur = Fq::ONE;
            for v in x.iter_mut() {
                *v = cur;
                cur *= domain.get_omega();
            }
            let expected = KZGCommitmentScheme::commit_lagrange(&params, &x);
            assert_eq!(KZGCommitmentScheme::commit_x(&params, &x).unwrap(), expected);
            assert_eq!(params.checked_tau_g1(), Some(expected));

            let corrupt = |f: &dyn Fn(&mut ParamsKZG<Bls12>)| {
                let mut p = params.clone();
                p.tau_g1 = Default::default();
                f(&mut p);
                assert!(p.checked_tau_g1().is_none());
                assert!(matches!(
                    KZGCommitmentScheme::commit_x(&p, &x),
                    Err(Error::InconsistentSrs)
                ));
            };
            corrupt(&|p| p.g[1] = p.g[1] + G1Projective::generator());
            corrupt(&|p| p.g[1] = G1Projective::identity());
            corrupt(&|p| p.g[0] = p.g[0].double());
            corrupt(&|p| {
                p.g[0] = p.g[0].double();
                p.g[1] = p.g[1].double();
            });
            corrupt(&|p| p.g.swap(0, 1));
            corrupt(&|p| p.g_lagrange[0] = p.g_lagrange[0] + G1Projective::generator());
            corrupt(&|p| p.s_g2 = p.s_g2 + G2Projective::generator());
            corrupt(&|p| p.g2 = p.g2.double());
        }

        // Parameters larger than the domain: the Lagrange route, as before.
        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(6, OsRng);
        let domain = EvaluationDomain::<Fq>::new(1, 5);
        let x = domain.lagrange_from_vec(
            (0..32).map(|j| domain.get_omega().pow_vartime([j as u64])).collect(),
        );
        assert_eq!(
            KZGCommitmentScheme::commit_x(&params, &x).unwrap(),
            KZGCommitmentScheme::commit_lagrange(&params, &x)
        );
    }

    #[test]
    fn test_parameter_serialisation_roundtrip() {
        const K: u32 = 4;

        use midnight_curves::Bls12;

        let params0: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut data = vec![];
        ParamsKZG::write_custom(&params0, &mut data, SerdeFormat::RawBytesUnchecked).unwrap();
        let params1 =
            ParamsKZG::<Bls12>::read_custom::<_>(&mut &data[..], SerdeFormat::RawBytesUnchecked)
                .unwrap();

        assert_eq!(params0.g.len(), params1.g.len());
        assert_eq!(params0.g_lagrange.len(), params1.g_lagrange.len());

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange, params1.g_lagrange);
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }
}
