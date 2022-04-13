//! Figuring out how generics work with ndarray and self implemented array types

use ndarray::{arr1, Array1};

struct Comp {}

impl Comp {
    pub fn new() -> Self {
        Comp {}
    }

    fn long_routine(&self, a: &Array1<f64>, b: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut res = Vec::new();
        // this needs Sub<T, Output = T>
        res.push(a - b);
        // this needs Add<T, Output = T>
        res.push(a + b);
        res.push(a * b);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_comp() {
        let a = arr1(&[1., 2.]);
        let b = arr1(&[0., 1.]);
        let res = Comp::new().long_routine(&a, &b);
        assert_eq!(res[0], arr1(&[1., 1.]));
        assert_eq!(res[1], arr1(&[1., 3.]));
        assert_eq!(res[2], arr1(&[0., 2.]));
    }
}
