extern crate blas_src;
extern crate openblas_src;

pub mod generics;
pub mod sim;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
