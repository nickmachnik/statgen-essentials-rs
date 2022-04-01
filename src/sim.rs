//! Simulation of genotype and phenotype data.

use approx::assert_abs_diff_eq;
use ndarray::{arr2, aview2, Array2, Axis};
use rand::distributions::Distribution;
use statrs::distribution::{Binomial, Uniform};

fn random_genotype_matrix(n: usize, p: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut m = Array2::<f64>::zeros((n, p));
    let u = Uniform::new(0., 0.5).unwrap();
    for cix in 0..p {
        let maf = u.sample(&mut rng);
        let bin = Binomial::new(maf, 2).unwrap();
        for rix in 0..n {
            m[[rix, cix]] = bin.sample(&mut rng);
        }
    }
    m
}

/// Subtracts the column mean from each column in a 2D array, returns the resulting array.
///
/// # Examples
/// ```
/// use statgen_essentials::sim::subtract_col_mean;
/// use ndarray::{Array2, aview2};
///
/// let m = Array2::from(vec![[1., 2., 3.], [4., 5., 6.]]);
/// assert_eq!(subtract_col_mean(&m), aview2(&[[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]));
/// ```
pub fn subtract_col_mean(m: &Array2<f64>) -> Array2<f64> {
    m - m.mean_axis(Axis(0)).unwrap()
}

pub fn standardize_cols(m: &Array2<f64>) -> Array2<f64> {
    subtract_col_mean(m) / m.std_axis(Axis(0), 1.)
}

fn corr(m: &Array2<f64>) -> Array2<f64> {
    let m_std = standardize_cols(m);
    m_std.t().dot(&m_std) / (m.dim().0 - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_genotype_matrix() {
        let n = 1000;
        let p = 100;
        let m = random_genotype_matrix(n, p);
        assert_eq!(m.dim(), (n, p));
    }

    #[test]
    fn test_corr() {
        let m = arr2(&[[0., 1.], [100., 0.], [200., 1.]]);
        let mcorr = corr(&m);
        println!("{:?}", mcorr);
        assert_abs_diff_eq!(mcorr, aview2(&[[1., 0.], [0., 1.]]), epsilon = 0.001);
    }

    #[test]
    fn test_random_genotype_matrix_has_independent_columns() {
        let n = 1000;
        let p = 2;
        let m = random_genotype_matrix(n, p);
        let mcorr = corr(&m);
        println!("{:?}", mcorr);
        assert_abs_diff_eq!(mcorr, Array2::eye(p), epsilon = 0.1);
    }
}
