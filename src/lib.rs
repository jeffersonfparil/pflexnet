mod cv;
mod linalg;
mod regularise;

use crate::cv::penalised_lambda_path_with_k_fold_cross_validation;
use ndarray::prelude::*;
use std::io;

pub fn portho(
    x: &Array2<f64>,
    y: &Array2<f64>,
    row_idx: &Vec<usize>,
    alpha: f64,
    iterative: bool,
    lambda_step_size: f64,
    r: usize,
) -> io::Result<(Array2<f64>, Vec<f64>, Vec<f64>)> {
    penalised_lambda_path_with_k_fold_cross_validation(
        x,
        y,
        row_idx,
        alpha,
        iterative,
        lambda_step_size,
        r,
    )
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lib() {
        let n = 100;
        let p = 1_000;
        let x: Array2<f64> =
            Array2::from_shape_vec((n, p), (0..(n * p)).map(|x| x as f64).collect::<Vec<f64>>())
                .unwrap();
        let y: Array2<f64> =
            Array2::from_shape_vec((n, 1), (0..n).map(|x| x as f64 / 2.0).collect::<Vec<f64>>())
                .unwrap();
        let row_idx: Vec<usize> = (0..n).collect();
        let (b_hat_penalised, alphas, lambdas) =
            portho(&x, &y, &row_idx, 1.0, false, 0.1, 2).unwrap();
        assert_eq!(b_hat_penalised[(0, 0)].ceil(), 1.0);
        assert_eq!(alphas[0].ceil(), 1.0);
        assert_eq!(lambdas[0].ceil(), 0.0);
    }
}
