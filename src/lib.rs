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
    use crate::cv::*;
    use crate::linalg::*;
    use rand::distributions::*;
    #[test]
    fn test_lib() {
        let nt = 90;
        let nv = 10;
        let n = nt + nv;
        let r = 10;
        // let alpha = 1.0; // L1 norm
        // let alpha = 0.0; // L2 norm
        let alpha = -1.0; // elastic norm (i.e. any `alpha < 0.0`)
        let lambda_step_size = 0.1;
        let p = 1_000;
        let q = 2;
        let h2 = 0.75;
        let mut rng = rand::thread_rng();
        let dist_unif = statrs::distribution::Uniform::new(0.0, 1.0).unwrap();
        // Simulate allele frequencies
        let mut x: Array2<f64> = Array2::ones((n, p + 1));
        for i in 0..n {
            for j in 1..(p + 1) {
                x[(i, j)] = dist_unif.sample(&mut rng);
            }
        }
        // Simulate effects
        let mut b: Array2<f64> = Array2::zeros((p + 1, 1));
        let idx_b: Vec<usize> = dist_unif
            .sample_iter(&mut rng)
            .take(q)
            .map(|x| (x * p as f64).floor() as usize)
            .collect::<Vec<usize>>();
        for i in idx_b.into_iter() {
            b[(i, 0)] = 1.00;
        }
        // Simulate phenotype
        let xb = multiply_views_xx(
            &x,
            &b,
            &(0..n).collect::<Vec<usize>>(),
            &(0..(p + 1)).collect::<Vec<usize>>(),
            &(0..(p + 1)).collect::<Vec<usize>>(),
            &vec![0 as usize],
        )
        .unwrap();
        let vg = xb.var_axis(Axis(0), 0.0)[0];
        let ve = (vg / h2) - vg;
        let dist_gaus = statrs::distribution::Normal::new(0.0, ve.sqrt()).unwrap();
        let e: Array2<f64> = Array2::from_shape_vec(
            (n, 1),
            dist_gaus
                .sample_iter(&mut rng)
                .take(n)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let y = &xb + e;
        let idx_training: Vec<usize> = (0..nt).collect();
        let idx_validation: Vec<usize> = (0..nv).collect();
        let (b_hat_penalised, alphas, lambdas) =
            portho(&x, &y, &idx_training, alpha, false, lambda_step_size, r).unwrap();
        let idx_cols_x: Vec<usize> = (0..p).collect();
        let idx_rows_b = idx_cols_x.clone();
        let idx_cols_b: Vec<usize> = vec![0];
        let y_hat = multiply_views_xx(
            &x,
            &b_hat_penalised,
            &idx_validation,
            &idx_cols_x,
            &idx_rows_b,
            &idx_cols_b,
        )
        .unwrap();
        let y_true: Array2<f64> = Array2::from_shape_vec(
            (nv, 1),
            idx_validation
                .iter()
                .map(|&i| y[(i, 0)])
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        assert_eq!(b_hat_penalised[(0, 0)].ceil(), 1.0);
        assert_eq!(alphas[0].round(), alpha.abs());
        assert_eq!(lambdas[0].ceil(), 1.0);
        println!("alphas={:?}; lambdas={:?}", alphas, lambdas);
        assert_eq!(
            pearsons_correlation(&y_true.column(0), &y_hat.column(0))
                .unwrap()
                .0
                .ceil(),
            1.0
        );
        println!(
            "rho and p-value={:?}",
            pearsons_correlation(&y_true.column(0), &y_hat.column(0)).unwrap()
        );
        // assert_eq!(0, 1);
    }
}
