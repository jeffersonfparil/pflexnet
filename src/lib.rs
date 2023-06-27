mod cv;
mod linalg;
mod regularise;

use crate::cv::penalised_lambda_path_with_k_fold_cross_validation;
use ndarray::prelude::*;
use std::io;

pub fn pflexnet(
    x: &Array2<f64>,
    y: &Array1<f64>,
    row_idx: &Vec<usize>,
    alpha: f64,
    kinship_covar: bool,
    lambda_step_size: f64,
    r: usize,
) -> io::Result<(Array1<f64>, f64, f64)> {
    penalised_lambda_path_with_k_fold_cross_validation(
        x,
        y,
        row_idx,
        alpha,
        kinship_covar,
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
    use rand::{distributions::*, rngs::StdRng, SeedableRng};
    #[test]
    fn test_lib() {
        let mut rng = StdRng::seed_from_u64(42);
        let nt = 90;
        let nv = 10;
        let n = nt + nv;
        let r = 10;
        let alpha = 1.0; // L1 norm
        // let alpha = 0.0; // L2 norm
        // let alpha = -1.0; // elastic norm (i.e. any `alpha < 0.0`)
        let lambda_step_size = 0.1;
        let p = 1_000;
        let q = 2;
        let h2 = 0.75;
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
        let y: Array1<f64> = (&xb + e).column(0).to_owned();
        let idx_training: Vec<usize> = (0..nt).collect();
        let idx_validation: Vec<usize> = (nt..n).collect();
        let (b_hat_penalised, alpha, lambda) =
            pflexnet(&x, &y, &idx_training, alpha, false, lambda_step_size, r).unwrap();
        let idx_cols_x: Vec<usize> = (0..p + 1).collect();
        let idx_rows_b = idx_cols_x.clone();
        let idx_cols_b: Vec<usize> = vec![0];
        let y_hat: Array1<f64> = multiply_views_xx(
            &x,
            &b_hat_penalised.to_owned().into_shape((p + 1, 1)).unwrap(),
            &idx_validation,
            &idx_cols_x,
            &idx_rows_b,
            &idx_cols_b,
        )
        .unwrap()
        .into_shape(nv)
        .unwrap();
        let y_true: Array1<f64> = Array1::from_shape_vec(
            nv,
            idx_validation.iter().map(|&i| y[i]).collect::<Vec<f64>>(),
        )
        .unwrap();
        println!("y_true={:?}", y_true);
        println!("y_hat={:?}", y_hat);
        println!("b_hat_penalised={:?}", b_hat_penalised);
        println!("alpha={:?}; lambda={:?}", alpha, lambda);
        println!(
            "rho and p-value={:?}",
            pearsons_correlation(&y_true, &y_hat).unwrap()
        );
        println!("mae={:?}", mean_absolute_error(&y_true, &y_hat).unwrap());
        println!(
            "rmse={:?}",
            root_mean_squared_error(&y_true, &y_hat).unwrap()
        );
        // assert_eq!(0, 1);

        assert_eq!(b_hat_penalised[0].ceil(), 1.0);
        assert_eq!(alpha.round(), 1.0);
        assert_eq!(lambda.round(), 1.0);
        assert_eq!(pearsons_correlation(&y_true, &y_hat).unwrap().0.ceil(), 1.0);
    }
}
