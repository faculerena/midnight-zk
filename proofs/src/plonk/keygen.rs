#![allow(clippy::int_plus_one)]

use std::ops::Range;

use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        Selector,
    },
    evaluation::Evaluator,
    permutation, Challenge, Error, LagrangeCoeff, Polynomial, ProvingKey, VerifyingKey,
};
use crate::{
    circuit::Value,
    dev::cost_model::cost_model_options,
    poly::{
        batch_invert_rational,
        commitment::{Params, PolynomialCommitmentScheme},
        EvaluationDomain, ExtendedLagrangeCoeff,
    },
    utils::{arithmetic::parallelize, rational::Rational, tmark},
};

pub(crate) fn create_domain<F, ConcreteCircuit>(
    k: u32,
    #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
) -> (
    EvaluationDomain<F>,
    ConstraintSystem<F>,
    ConcreteCircuit::Config,
)
where
    F: WithSmallOrderMulGroup<3>,
    ConcreteCircuit: Circuit<F>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, params);
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);

    let degree = cs.degree();

    let domain = EvaluationDomain::new(degree as u32, k);

    (domain, cs, config)
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    k: u32,
    fixed: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Vec<bool>>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.selectors[selector.0][row] = true;

        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Rational<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about fixed columns here
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Rational<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.permutation.copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Rational<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let col = self.fixed.get_mut(column.index()).ok_or(Error::BoundsFailure)?;

        let filler = to.assign()?;
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

/// Compute the minimal `k` to compute a circuit.
pub fn k_from_circuit<F: Ord + Field + FromUniformBytes<64>, C: Circuit<F>>(circuit: &C) -> u32 {
    cost_model_options(circuit).min_k as u32
}

/// Generates a `VerifyingKey` from a `Circuit` instance.
///
/// Automatically determines the smallest `k` required for the given circuit
/// and adjusts the received parameters to match the circuit's size.
/// Use `keygen_vk_with_k` to specify a custom `k` value.
pub fn keygen_vk<F, CS, ConcreteCircuit>(
    params: &CS::Parameters,
    circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<F, CS>, Error>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64> + Ord,
    CS: PolynomialCommitmentScheme<F>,
    ConcreteCircuit: Circuit<F>,
{
    let k = k_from_circuit(circuit);

    if params.max_k() != k {
        return Err(Error::SrsError(params.max_k() as usize, k as usize));
    }

    keygen_vk_with_k(params, circuit, k)
}

/// Generate a `VerifyingKey` from an instance of `Circuit`.
pub fn keygen_vk_with_k<F, CS, ConcreteCircuit>(
    params: &CS::Parameters,
    circuit: &ConcreteCircuit,
    k: u32,
) -> Result<VerifyingKey<F, CS>, Error>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64> + Ord,
    CS: PolynomialCommitmentScheme<F>,
    ConcreteCircuit: Circuit<F>,
{
    if params.max_k() < k {
        return Err(Error::NotEnoughRowsAvailable {
            current_k: params.max_k(),
        });
    }

    let (domain, cs, config) = create_domain::<F, ConcreteCircuit>(
        k,
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );

    if (domain.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(domain.k()));
    }

    tmark("vk: start (domain+configure done)");
    let mut assembly: Assembly<F> = Assembly {
        k: domain.k(),
        fixed: vec![domain.empty_lagrange_rational(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(domain.n as usize, &cs.permutation),
        selectors: vec![vec![false; domain.n as usize]; cs.num_selectors],
        usable_rows: 0..domain.n as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    tmark("vk: synthesize");
    let mut fixed = batch_invert_rational(assembly.fixed);
    // After this, the ConstraintSystem should not have any selectors: `verify` does
    // not need them, and `keygen_pk` regenerates `cs` from scratch anyways.
    let selectors = std::mem::take(&mut assembly.selectors);
    let (cs, selector_polys) = cs.directly_convert_selectors_to_fixed(selectors);
    fixed.extend(selector_polys.into_iter().map(|poly| domain.lagrange_from_vec(poly)));

    tmark("vk: batch_invert + selectors");
    let permutation_vk = assembly.permutation.build_vk(params, &domain, &cs.permutation);

    tmark("vk: permutation build_vk (polys + commitments)");
    let fixed_commitments: Vec<_> = fixed.iter().map(|poly| CS::commit_lagrange(params, poly)).collect();
    tmark(&format!("vk: {} fixed commitments", fixed_commitments.len()));

    Ok(VerifyingKey::from_parts(
        domain,
        fixed_commitments,
        permutation_vk,
        cs,
    ))
}

/// The part of a [`ProvingKey`] that [`ProvingKey::write`] serialises, and
/// nothing else: it can be written out, and that is all it can do.
///
/// It holds the verifying key, the fixed columns and the permutation
/// polynomials in Lagrange form, which is all that `ProvingKey::write` emits
/// and all that `ProvingKey::read` consumes. The data a prover derives from
/// those (coefficient and extended forms, `l0`, `l_last`, `l_active_row` and
/// the evaluator) is not computed, so this is deliberately a type of its own
/// with no way to get a `ProvingKey` out of it: to prove, read the written
/// key back with [`ProvingKey::read`], which gives a complete proving key.
#[derive(Debug)]
pub struct WriteOnlyProvingKey<F: ff::PrimeField, CS: PolynomialCommitmentScheme<F>> {
    /// A `ProvingKey` whose derived fields are empty. Never handed out.
    inner: ProvingKey<F, CS>,
}

impl<F, CS> WriteOnlyProvingKey<F, CS>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64> + midnight_curves::serde::SerdeObject,
    CS: PolynomialCommitmentScheme<F>,
{
    /// Writes the proving key to a buffer: the same bytes as
    /// [`ProvingKey::write`] gives for the proving key of [`keygen_pk`].
    pub fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        format: crate::utils::SerdeFormat,
    ) -> std::io::Result<()> {
        self.inner.write(writer, format)
    }
}

/// Generates a `VerifyingKey` and a [`WriteOnlyProvingKey`] with a single
/// synthesis of the circuit, for when the keys are only to be written out.
pub fn keygen_for_write_with_k<F, CS, ConcreteCircuit>(
    params: &CS::Parameters,
    circuit: &ConcreteCircuit,
    k: u32,
) -> Result<(VerifyingKey<F, CS>, WriteOnlyProvingKey<F, CS>), Error>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64> + Ord,
    CS: PolynomialCommitmentScheme<F>,
    ConcreteCircuit: Circuit<F>,
{
    if params.max_k() < k {
        return Err(Error::NotEnoughRowsAvailable {
            current_k: params.max_k(),
        });
    }

    let (domain, cs, config) = create_domain::<F, ConcreteCircuit>(
        k,
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );

    if (domain.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(domain.k()));
    }

    tmark("w: start (domain+configure done)");
    let mut assembly: Assembly<F> = Assembly {
        k: domain.k(),
        fixed: vec![domain.empty_lagrange_rational(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(domain.n as usize, &cs.permutation),
        selectors: vec![vec![false; domain.n as usize]; cs.num_selectors],
        usable_rows: 0..domain.n as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    tmark("w: synthesize");
    let mut fixed = batch_invert_rational(assembly.fixed);
    // After this, the ConstraintSystem should not have any selectors: `verify` does
    // not need them, and `keygen_pk` regenerates `cs` from scratch anyways.
    let selectors = std::mem::take(&mut assembly.selectors);
    let (cs, selector_polys) = cs.directly_convert_selectors_to_fixed(selectors);
    fixed.extend(selector_polys.into_iter().map(|poly| domain.lagrange_from_vec(poly)));

    tmark("w: batch_invert + selectors");
    let (permutation_vk, permutations) =
        assembly.permutation.build_vk_and_permutations(params, &domain, &cs.permutation)?;
    tmark("w: permutation polys + commitments");

    let fixed_commitments: Vec<_> =
        fixed.iter().map(|poly| CS::commit_lagrange(params, poly)).collect();
    tmark(&format!("w: {} fixed commitments", fixed_commitments.len()));

    let vk = VerifyingKey::from_parts(domain, fixed_commitments, permutation_vk, cs);
    let empty = || Polynomial {
        values: Vec::new(),
        _marker: std::marker::PhantomData,
    };
    let pk = ProvingKey {
        vk: vk.clone(),
        l0: empty(),
        l_last: empty(),
        l_active_row: empty(),
        fixed_values: fixed,
        fixed_polys: Vec::new(),
        fixed_cosets: Vec::new(),
        permutation: permutation::ProvingKey {
            permutations,
            polys: Vec::new(),
            cosets: Vec::new(),
        },
        ev: Evaluator::default(),
    };
    tmark("w: assemble");
    Ok((vk, WriteOnlyProvingKey { inner: pk }))
}


/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub fn keygen_pk<F, CS, ConcreteCircuit>(
    vk: VerifyingKey<F, CS>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<F, CS>, Error>
where
    F: WithSmallOrderMulGroup<3>,
    CS: PolynomialCommitmentScheme<F>,
    ConcreteCircuit: Circuit<F>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);

    let cs = cs;

    let n = vk.domain.n as usize;
    tmark("pk: start (configure done)");
    let mut assembly: Assembly<F> = Assembly {
        k: vk.domain.k(),
        fixed: vec![vk.domain.empty_lagrange_rational(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(n, &cs.permutation),
        selectors: vec![vec![false; n]; cs.num_selectors],
        usable_rows: 0..n - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    tmark("pk: synthesize");
    let mut fixed = batch_invert_rational(assembly.fixed);
    let (cs, selector_polys) = cs.directly_convert_selectors_to_fixed(assembly.selectors);
    fixed.extend(selector_polys.into_iter().map(|poly| vk.domain.lagrange_from_vec(poly)));

    tmark("pk: batch_invert + selectors");
    let fixed_polys: Vec<_> =
        fixed.par_iter().map(|poly| vk.domain.lagrange_to_coeff(poly.clone())).collect();

    tmark("pk: fixed lagrange_to_coeff");
    let fixed_cosets = fixed_polys
        .par_iter()
        .map(|poly| vk.domain.coeff_to_extended(poly.clone()))
        .collect();

    tmark("pk: fixed coeff_to_extended");
    let permutation_pk = assembly.permutation.build_pk::<F>(&vk.domain, &cs.permutation);

    tmark("pk: permutation build_pk (polys + coeff + cosets)");
    let [l0, l_last, l_active_row] = compute_lagrange_polys(&vk, &cs);
    tmark("pk: l0/l_last/l_active");
    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);
    tmark("pk: evaluator");
    Ok(ProvingKey {
        vk,
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        fixed_polys,
        fixed_cosets,
        permutation: permutation_pk,
        ev,
    })
}

pub(crate) fn compute_lagrange_polys<F, CS>(
    vk: &VerifyingKey<F, CS>,
    cs: &ConstraintSystem<F>,
) -> [Polynomial<F, ExtendedLagrangeCoeff>; 3]
where
    F: WithSmallOrderMulGroup<3>,
    CS: PolynomialCommitmentScheme<F>,
{
    // Compute l_0(X)
    // TODO: this can be done more efficiently
    let mut l0 = vk.domain.empty_lagrange();
    l0[0] = F::ONE;
    let l0 = vk.domain.lagrange_to_coeff(l0);
    let l0 = vk.domain.coeff_to_extended(l0);

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = vk.domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
        *evaluation = F::ONE;
    }
    let l_blind = vk.domain.lagrange_to_coeff(l_blind);
    let l_blind = vk.domain.coeff_to_extended(l_blind);

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = vk.domain.empty_lagrange();
    let n = vk.domain.n as usize;
    l_last[n - cs.blinding_factors() - 1] = F::ONE;
    let l_last = vk.domain.lagrange_to_coeff(l_last);
    let l_last = vk.domain.coeff_to_extended(l_last);

    // Compute l_active_row(X)
    let one = F::ONE;
    let mut l_active_row = vk.domain.empty_extended();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    [l0, l_last, l_active_row]
}
