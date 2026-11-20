use ff::WithSmallOrderMulGroup;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::{Argument, ProvingKey, VerifyingKey};
use crate::{
    plonk::{Any, Column, Error},
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial,
    },
    utils::arithmetic::parallelize,
};

/// Struct that accumulates all the necessary data in order to construct the
/// permutation argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    /// Columns that participate on the copy permutation argument.
    columns: Vec<Column<Any>>,
    /// Mapping of the actual copies done.
    mapping: Vec<Vec<(usize, usize)>>,
    /// Some aux data used to swap positions directly when sorting.
    aux: Vec<Vec<(usize, usize)>>,
    /// More aux data
    sizes: Vec<Vec<usize>>,
}

impl Assembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        // Initialize the copy vector to keep track of copy constraints in all
        // the permutation arguments.
        let mut columns = vec![];
        for i in 0..p.columns.len() {
            // Computes [(i, 0), (i, 1), ..., (i, n - 1)]
            columns.push((0..n).map(|j| (i, j)).collect());
        }

        // Before any equality constraints are applied, every cell in the permutation is
        // in a 1-cycle; therefore mapping and aux are identical, because every cell is
        // its own distinguished element.
        Assembly {
            columns: p.columns.clone(),
            mapping: columns.clone(),
            aux: columns,
            sizes: vec![vec![1usize; n]; p.columns.len()],
        }
    }

    pub(crate) fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(Error::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(Error::ColumnNotInPermutation(right_column))?;

        // Check bounds
        if left_row >= self.mapping[left_column].len()
            || right_row >= self.mapping[right_column].len()
        {
            return Err(Error::BoundsFailure);
        }

        // See book/src/design/permutation.md for a description of this algorithm.

        let mut left_cycle = self.aux[left_column][left_row];
        let mut right_cycle = self.aux[right_column][right_row];

        // If left and right are in the same cycle, do nothing.
        if left_cycle == right_cycle {
            return Ok(());
        }

        if self.sizes[left_cycle.0][left_cycle.1] < self.sizes[right_cycle.0][right_cycle.1] {
            std::mem::swap(&mut left_cycle, &mut right_cycle);
        }

        // Merge the right cycle into the left one.
        self.sizes[left_cycle.0][left_cycle.1] += self.sizes[right_cycle.0][right_cycle.1];
        let mut i = right_cycle;
        loop {
            self.aux[i.0][i.1] = left_cycle;
            i = self.mapping[i.0][i.1];
            if i == right_cycle {
                break;
            }
        }

        let tmp = self.mapping[left_column][left_row];
        self.mapping[left_column][left_row] = self.mapping[right_column][right_row];
        self.mapping[right_column][right_row] = tmp;

        Ok(())
    }

    pub(crate) fn build_vk<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        p: &Argument,
    ) -> Result<VerifyingKey<F, CS>, Error> {
        Ok(build_vk(params, domain, p, |i, j| self.mapping[i][j])?.0)
    }

    /// The verifying key of [`Self::build_vk`], and the permutation
    /// polynomials in Lagrange form it commits to.
    #[allow(clippy::type_complexity)]
    pub(crate) fn build_vk_and_permutations<
        F: WithSmallOrderMulGroup<3>,
        CS: PolynomialCommitmentScheme<F>,
    >(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        p: &Argument,
    ) -> Result<(VerifyingKey<F, CS>, Vec<Polynomial<F, LagrangeCoeff>>), Error> {
        build_vk(params, domain, p, |i, j| self.mapping[i][j])
    }

    pub(crate) fn build_pk<F: WithSmallOrderMulGroup<3>>(
        self,
        domain: &EvaluationDomain<F>,
        p: &Argument,
    ) -> ProvingKey<F> {
        build_pk::<_>(domain, p, |i, j| self.mapping[i][j])
    }

    /// Returns columns that participate in the permutation argument.
    pub fn columns(&self) -> &[Column<Any>] {
        &self.columns
    }

    /// Returns mappings of the copies.
    pub fn mapping(
        &self,
    ) -> impl Iterator<Item = impl IndexedParallelIterator<Item = (usize, usize)> + '_> {
        self.mapping.iter().map(|c| c.par_iter().copied())
    }
}

/// The permutation polynomials of `p` in Lagrange form, and the identity
/// permutation they are built from: `deltaomega[i][j] = delta^i * omega^j`.
#[allow(clippy::type_complexity)]
fn permutation_polys<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    p: &Argument,
    mapping: &(impl Fn(usize, usize) -> (usize, usize) + Sync),
) -> (Vec<Vec<F>>, Vec<Polynomial<F, LagrangeCoeff>>) {
    // Compute [omega^0, omega^1, ..., omega^{params.n - 1}]
    let mut omega_powers = vec![F::ZERO; domain.n as usize];
    {
        let omega = domain.get_omega();
        parallelize(&mut omega_powers, |o, start| {
            let mut cur = omega.pow_vartime([start as u64]);
            for v in o.iter_mut() {
                *v = cur;
                cur *= &omega;
            }
        })
    }

    // Compute [omega_powers * \delta^0, omega_powers * \delta^1, ..., omega_powers
    // * \delta^m]
    let mut deltaomega = vec![omega_powers; p.columns.len()];
    {
        parallelize(&mut deltaomega, |o, start| {
            let mut cur = F::DELTA.pow_vartime([start as u64]);
            for omega_powers in o.iter_mut() {
                for v in omega_powers {
                    *v *= &cur;
                }
                cur *= &F::DELTA;
            }
        });
    }

    // Compute the permutation polynomials from the mapping.
    let mut permutations = vec![domain.empty_lagrange(); p.columns.len()];
    {
        parallelize(&mut permutations, |o, start| {
            o.par_iter_mut().enumerate().for_each(|(x, permutation_poly)| {
                let i = start + x;
                permutation_poly.par_iter_mut().enumerate().for_each(|(j, p)| {
                    let (permuted_i, permuted_j) = mapping(i, j);
                    *p = deltaomega[permuted_i][permuted_j];
                })
            })
        });
    }

    (deltaomega, permutations)
}

pub(crate) fn build_pk<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> ProvingKey<F> {
    let (_, permutations) = permutation_polys(domain, p, &mapping);
    let (polys, cosets) = compute_polys_and_cosets::<F>(domain, p, &permutations);

    ProvingKey {
        permutations,
        polys,
        cosets,
    }
}

/// The verifying key of the permutation argument, and the permutation
/// polynomials in Lagrange form it commits to.
#[allow(clippy::type_complexity)]
pub(crate) fn build_vk<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
    params: &CS::Parameters,
    domain: &EvaluationDomain<F>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> Result<(VerifyingKey<F, CS>, Vec<Polynomial<F, LagrangeCoeff>>), Error> {
    let (deltaomega, permutations) = permutation_polys(domain, p, &mapping);

    // A cell that no copy constraint moves keeps the value of the identity
    // permutation, delta^i * omega^j, and the identity permutation of column
    // i is the polynomial delta^i * X. So the commitment to the permutation
    // polynomial of column i is delta^i * [X] plus the commitment to its
    // difference with the identity, which is zero outside the cells that copy
    // constraints move: that MSM only runs over those cells.
    let mut commitments = Vec::with_capacity(p.columns.len());
    if !p.columns.is_empty() {
        // deltaomega[0] is the powers of omega: X in Lagrange form.
        let x_commitment = CS::commit_x(params, &domain.lagrange_from_vec(deltaomega[0].clone()))?;
        let mut delta_i = F::ONE;
        for (i, permutation) in permutations.iter().enumerate() {
            let mut diff = domain.empty_lagrange();
            diff.par_iter_mut().enumerate().for_each(|(j, d)| {
                if mapping(i, j) != (i, j) {
                    *d = permutation[j] - deltaomega[i][j];
                }
            });
            commitments.push(CS::commit_lagrange(params, &diff) + x_commitment.clone() * delta_i);
            delta_i *= F::DELTA;
        }
    }

    Ok((VerifyingKey { commitments }, permutations))
}

#[allow(clippy::type_complexity)]
pub(crate) fn compute_polys_and_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    p: &Argument,
    permutations: &[Polynomial<F, LagrangeCoeff>],
) -> (
    Vec<Polynomial<F, Coeff>>,
    Vec<Polynomial<F, ExtendedLagrangeCoeff>>,
) {
    let mut polys = vec![domain.empty_coeff(); p.columns.len()];
    {
        parallelize(&mut polys, |o, start| {
            for (x, poly) in o.iter_mut().enumerate() {
                let i = start + x;
                let permutation_poly = permutations[i].clone();
                *poly = domain.lagrange_to_coeff(permutation_poly);
            }
        });
    }
    let mut cosets = vec![domain.empty_extended(); p.columns.len()];
    {
        parallelize(&mut cosets, |o, start| {
            for (x, coset) in o.iter_mut().enumerate() {
                let i = start + x;
                let poly = polys[i].clone();
                *coset = domain.coeff_to_extended(poly);
            }
        });
    }
    (polys, cosets)
}

#[cfg(test)]
mod tests {
    use midnight_curves::{Bls12, Fq};
    use rand_core::OsRng;

    use super::*;
    use crate::poly::kzg::{params::ParamsKZG, KZGCommitmentScheme};

    /// The commitments of `build_vk` are the Lagrange commitments to the
    /// permutation polynomials, and those are the ones of `build_pk`, with
    /// and without copy constraints, and for parameters larger than the
    /// domain.
    #[test]
    fn test_build_vk_commits_to_the_permutation_polys() {
        const K: u32 = 6;
        let n = 1 << K;
        let domain = EvaluationDomain::<Fq>::new(1, K);
        let p = Argument {
            columns: (0..4).map(|i| Column::new(i, Any::Fixed)).collect(),
        };
        let copies = [
            (0, 1, 1, 2),
            (1, 2, 3, 5),
            (2, 0, 2, 7),
            (0, 10, 3, 10),
            (3, 63, 0, 0),
        ];

        for params_k in [K, K + 1] {
            let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(params_k, OsRng);
            for nb_copies in [0, 1, copies.len()] {
                let mut assembly = Assembly::new(n, &p);
                for &(l, l_row, r, r_row) in &copies[..nb_copies] {
                    assembly.copy(p.columns[l], l_row, p.columns[r], r_row).unwrap();
                }
                let mapping = |i: usize, j: usize| assembly.mapping[i][j];

                let (vk, permutations) =
                    build_vk::<_, KZGCommitmentScheme<Bls12>>(&params, &domain, &p, mapping)
                        .unwrap();
                let expected: Vec<_> = permutations
                    .iter()
                    .map(|poly| KZGCommitmentScheme::commit_lagrange(&params, poly))
                    .collect();
                assert_eq!(vk.commitments, expected);

                let pk = build_pk(&domain, &p, mapping);
                assert_eq!(pk.permutations.len(), permutations.len());
                for (a, b) in pk.permutations.iter().zip(&permutations) {
                    assert_eq!(a.values, b.values);
                }
            }
        }
    }
}
