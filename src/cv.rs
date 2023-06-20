use crate::linalg::*;
use crate::regularise::*;
use ndarray::{prelude::*, Zip};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::io::{self, Error, ErrorKind};

fn k_split(row_idx: &Vec<usize>, mut k: usize) -> io::Result<(Vec<usize>, usize, usize)> {
    let n = row_idx.len();
    if (k >= n) | (n <= 2) {
        return Err(Error::new(ErrorKind::Other, "The number of splits, i.e. k, needs to be less than the number of pools, n, and n > 2. We are aiming for fold sizes of 10 or greater."));
    }
    let mut s = (n as f64 / k as f64).floor() as usize;
    while s < 10 {
        if n < 20 {
            println!("Warning: number of pools is less than 20, so we're using k=2.");
            k = 2;
            s = (n as f64 / k as f64).floor() as usize;
            break;
        }
        k -= 1;
        s = (n as f64 / k as f64).floor() as usize;
    }
    let mut g = (0..k)
        .flat_map(|x| std::iter::repeat(x).take(s))
        .collect::<Vec<usize>>();
    if n - s > 0 {
        for _i in 0..(n - s) {
            g.push(k);
        }
    }
    let mut rng = rand::thread_rng();
    let shuffle = row_idx.iter().map(|&x| x).choose_multiple(&mut rng, n);
    let mut out: Vec<usize> = Vec::new();
    for i in 0..n {
        out.push(g[shuffle[i]]);
    }
    Ok((out, k, s))
}

fn pearsons_correlation(
    x: &ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 1]>>,
    y: &ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 1]>>,
) -> io::Result<(f64, f64)> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::new(
            ErrorKind::Other,
            "Input vectors are not the same size.",
        ));
    }
    let mu_x = x.mean().unwrap();
    let mu_y = y.mean().unwrap();
    let x_less_mu_x = x.map(|x| x - mu_x);
    let y_less_mu_y = y.map(|y| y - mu_y);
    let x_less_mu_x_squared = x_less_mu_x.map(|x| x.powf(2.0));
    let y_less_mu_y_squared = y_less_mu_y.map(|y| y.powf(2.0));
    let numerator = (x_less_mu_x * y_less_mu_y).sum();
    let denominator = x_less_mu_x_squared.sum().sqrt() * y_less_mu_y_squared.sum().sqrt();
    let r_tmp = numerator / denominator;
    let r = match r_tmp.is_nan() {
        true => 0.0,
        false => r_tmp,
    };
    let sigma_r_denominator = (1.0 - r.powf(2.0)) / (n as f64 - 2.0);
    if sigma_r_denominator <= 0.0 {
        // Essentially no variance in r2, hence very significant
        return Ok((r, f64::EPSILON));
    }
    let sigma_r = sigma_r_denominator.sqrt();
    let t = r / sigma_r;
    let d = StudentsT::new(0.0, 1.0, n as f64 - 2.0).unwrap();
    let pval = 2.00 * (1.00 - d.cdf(t.abs()));
    Ok((r, pval))
}

fn error_index(
    b_hat: &Array2<f64>,
    x: &Array2<f64>,
    y: &Array2<f64>,
    idx_validation: &Vec<usize>,
) -> io::Result<Vec<f64>> {
    let (n, p) = (idx_validation.len(), x.ncols());
    let k = y.ncols();
    let (p_, k_) = (b_hat.nrows(), b_hat.ncols());
    if p != p_ {
        return Err(Error::new(
            ErrorKind::Other,
            "The X matrix is incompatible with b_hat.",
        ));
    }
    if k != k_ {
        return Err(Error::new(
            ErrorKind::Other,
            "The y matrix/vector is incompatible with b_hat.",
        ));
    }
    let idx_b_hat = &(0..p).collect();
    let mut error_index: Vec<f64> = Vec::with_capacity(k);
    for j in 0..k {
        // let y_pred: Array2<f64> = x * b_hat;
        let y_true_j: Array2<f64> = Array2::from_shape_vec(
            (n, 1),
            idx_validation
                .iter()
                .map(|&i| y[(i, j)])
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let b_hat_j: Array2<f64> =
            Array2::from_shape_vec((p, 1), b_hat.column(j).to_owned().to_vec()).unwrap();
        let y_pred_j: Array2<f64> =
            multiply_views_xx(x, &b_hat_j, idx_validation, idx_b_hat, idx_b_hat, &vec![0]).unwrap();
        let _min = y_true_j
            .iter()
            .fold(y_true_j[(0, 0)], |min, &x| if x < min { x } else { min });
        let _max = y_true_j
            .iter()
            .fold(y_true_j[(0, 0)], |max, &x| if x > max { x } else { max });
        let (cor, _pval) = pearsons_correlation(&y_true_j.column(0), &y_pred_j.column(0)).unwrap();
        // let mbe = (y_true_j - &y_pred_j).mean() / (max - min);vec![0.0]
        let mae = (&y_true_j - &y_pred_j)
            .iter()
            .fold(0.0, |norm, &x| norm + x.abs());
        // / (max - min);
        let mse = (&y_true_j - &y_pred_j)
            .iter()
            .fold(0.0, |norm, &x| norm + x.powf(2.0));
        // / (max - min).powf(2.0);
        let rmse = mse.sqrt(); // / (max - min);
        error_index.push(((1.0 - cor.abs()) + mae + mse + rmse) / 4.0);
        // error_index.push(((1.0 - cor.abs()) + mae + mse) / 3.0);
        // error_index.push(((1.0 - cor.abs()) + rmse) / 2.0);
        // error_index.push(1.0 - cor.abs());
        // error_index.push(rmse);
        // error_index.push(mae);
    }
    Ok(error_index)
}

pub fn penalised_lambda_path_with_k_fold_cross_validation(
    x: &Array2<f64>,
    y: &Array2<f64>,
    row_idx: &Vec<usize>,
    alpha: f64,
    iterative: bool,
    lambda_step_size: f64,
    r: usize,
) -> io::Result<(Array2<f64>, Vec<f64>, Vec<f64>)> {
    let (_n, p) = (row_idx.len(), x.ncols());
    let k = y.ncols();
    let max_usize: usize = (1.0 / lambda_step_size).round() as usize;
    let parameters_path: Array1<f64> = (0..(max_usize + 1))
        .into_iter()
        .map(|x| (x as f64) / (max_usize as f64))
        .collect();
    let l = parameters_path.len();

    let (alpha_path, a): (Array2<f64>, usize) = if alpha >= 0.0 {
        // ridge or lasso optimise for lambda only
        (
            Array2::from_shape_vec((1, l), std::iter::repeat(alpha).take(l).collect()).unwrap(),
            1,
        )
    } else {
        // glmnet optimise for both alpha and lambda
        (
            Array2::from_shape_vec(
                (l, l),
                parameters_path
                    .clone()
                    .iter()
                    .flat_map(|&x| std::iter::repeat(x).take(l))
                    .collect(),
            )
            .unwrap(),
            l,
        )
    };
    let lambda_path: Array2<f64> = Array2::from_shape_vec(
        (a, l),
        std::iter::repeat(parameters_path.clone())
            .take(a)
            .flat_map(|x| x)
            .collect(),
    )
    .unwrap();

    let (_, nfolds, _s) = k_split(row_idx, 10).unwrap();
    let mut performances: Array5<f64> = Array5::from_elem((r, nfolds, a, l, k), f64::NAN);
    for rep in 0..r {
        let (groupings, _, _) = k_split(row_idx, 10).unwrap();
        for fold in 0..nfolds {
            let idx_validation: Vec<usize> = groupings
                .iter()
                .enumerate()
                .filter(|(_, x)| *x == &fold)
                .map(|(i, _)| row_idx[i])
                .collect();
            let idx_training: Vec<usize> = groupings
                .iter()
                .enumerate()
                .filter(|(_, x)| *x != &fold)
                .map(|(i, _)| row_idx[i])
                .collect();
            let b_hat = ols(&x, &y, &idx_training).unwrap();
            let mut errors: Array2<Vec<f64>> = Array2::from_elem((a, l), vec![]);
            let mut b_hats: Array2<Array2<f64>> =
                Array2::from_elem((a, l), Array2::from_elem((1, 1), f64::NAN));
            if iterative == false {
                Zip::from(&mut errors)
                    .and(&mut b_hats)
                    .and(&alpha_path)
                    .and(&lambda_path)
                    .par_for_each(|err, b, &alfa, &lambda| {
                        let b_hat_new: Array2<f64> =
                            expand_and_contract(&b_hat, &b_hat, alfa, lambda).unwrap();
                        *err = error_index(&b_hat_new, x, y, &idx_validation).unwrap();
                        *b = b_hat_new;
                    });
            } else {
                let b_hat_proxy = ols_iterative_with_kinship_pca_covariate(x, y, row_idx).unwrap();
                Zip::from(&mut errors)
                    .and(&mut b_hats)
                    .and(&alpha_path)
                    .and(&lambda_path)
                    .par_for_each(|err, b, &alfa, &lambda| {
                        let b_hat_new: Array2<f64> =
                            expand_and_contract(&b_hat, &b_hat_proxy, alfa, lambda).unwrap();
                        *err = error_index(&b_hat_new, x, y, &idx_validation).unwrap();
                        *b = b_hat_new;
                    });
            }

            // Append performances, i.e. error index: f(1-cor, rmse, mae, etc...)
            for i0 in 0..a {
                for i1 in 0..l {
                    for j in 0..k {
                        performances[(rep, fold, i0, i1, j)] = errors[(i0, i1)][j];
                    }
                }
            }
        }
    }

    // Find best alpha, lambda and beta on the full dataset
    // let mean_error_across_reps_and_folds: Array3<f64> = performances
    //     .mean_axis(Axis(0))
    //     .unwrap()
    //     .mean_axis(Axis(0))
    //     .unwrap();
    let b_hat = ols(x, y, row_idx).unwrap();
    let mut b_hat_penalised = b_hat.clone();
    let mut alphas = vec![];
    let mut lambdas = vec![];
    for j in 0..k {
        ///////////////////////////////////
        // TODO: Account for overfit cross-validation folds, i.e. filter them out, or just use mode of the lambda and alphas?
        let mut alpha_path_counts: Array1<usize> = Array1::from_elem(l, 0);
        let mut lambda_path_counts: Array1<usize> = Array1::from_elem(l, 0);
        for rep in 0..r {
            let mean_error_per_rep_across_folds: Array2<f64> = performances
                .slice(s![rep, .., .., .., j])
                .mean_axis(Axis(0))
                .unwrap();
            let min_error = mean_error_per_rep_across_folds.fold(
                mean_error_per_rep_across_folds[(0, 0)],
                |min, &x| {
                    if x < min {
                        x
                    } else {
                        min
                    }
                },
            );
            // println!("min_error={:?}", min_error);
            // println!("mean_error_per_rep_across_folds={:?}", mean_error_per_rep_across_folds);
            // println!("lambda_path_counts={:?}", lambda_path_counts);
            let ((idx_0, idx_1), _) = mean_error_per_rep_across_folds
                .indexed_iter()
                .find(|((_i, _j), &x)| x == min_error)
                .unwrap();
            // println!("lambda_path[(idx_0, idx_1)]={:?}", lambda_path[(idx_0, idx_1)]);
            for a in 0..l {
                if alpha_path[(idx_0, idx_1)] == parameters_path[a] {
                    alpha_path_counts[a] += 1;
                }
                if lambda_path[(idx_0, idx_1)] == parameters_path[a] {
                    lambda_path_counts[a] += 1;
                }
            }
        }
        // Find the mode alpha and lambda
        // println!("lambda_path_counts={:?}", lambda_path_counts);
        let alpha_max_count = alpha_path_counts.fold(0, |max, &x| if x > max { x } else { max });
        let (alpha_idx, _) = alpha_path_counts
            .indexed_iter()
            .find(|(_a, &x)| x == alpha_max_count)
            .unwrap();
        let lambda_max_count = lambda_path_counts.fold(0, |max, &x| if x > max { x } else { max });
        let (lambda_idx, _) = lambda_path_counts
            .indexed_iter()
            .find(|(_a, &x)| x == lambda_max_count)
            .unwrap();
        alphas.push(parameters_path[alpha_idx]);
        lambdas.push(parameters_path[lambda_idx]);
        ///////////////////////////////////

        // let min_error = mean_error_across_reps_and_folds
        //     .index_axis(Axis(2), j)
        //     .iter()
        //     .fold(mean_error_across_reps_and_folds[(0, 0, j)], |min, &x| {
        //         if x < min {
        //             x
        //         } else {
        //             min
        //         }
        //     });

        // let ((idx_0, idx_1), _) = mean_error_across_reps_and_folds
        //     .index_axis(Axis(2), j)
        //     .indexed_iter()
        //     .find(|((_i, _j), &x)| x == min_error)
        //     .unwrap();

        // alphas.push(alpha_path[(idx_0, idx_1)]);
        // lambdas.push(lambda_path[(idx_0, idx_1)]);

        // Note: Lasso/Ridge and glmnet have different best paramaters even though for example lasso seems to get the best performance while glmnet failed to get the same result even though it should given it searches other alphas including alpha=1.00 in lasso.
        //       This is because of the stochasticity per fold, i.e. glmnet might get an alpha  not equal to 1 which results in better performance.
        // println!("min_error={}; alpha={:?}; lambda={:?}", min_error, alphas, lambdas);
        let b_hat_penalised_2d: Array2<f64> = if iterative == false {
            expand_and_contract(&b_hat, &b_hat, alphas[j], lambdas[j]).unwrap()
        } else {
            let b_hat_proxy = ols_iterative_with_kinship_pca_covariate(x, y, row_idx).unwrap();
            expand_and_contract(&b_hat, &b_hat_proxy, alphas[j], lambdas[j]).unwrap()
        };

        for i in 0..p {
            b_hat_penalised[(i, j)] = b_hat_penalised_2d[(i, j)];
        }
    }
    // println!("#########################################");
    // println!("alphas={:?}", alphas);
    // println!("lambdas={:?}", lambdas);
    Ok((b_hat_penalised, alphas, lambdas))
    // Ok((b_hat_proxy, vec![0.0], vec![0.0]))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    #[test]
    fn test_porthos() {
        // Inputs
        let row_idx: Vec<usize> = (0..10).collect();
        let k = 2;
        // Outputs
        let (idx, k, s) = k_split(&row_idx, k).unwrap();
        // Assertions
        assert_eq!(idx, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(k, 2);
        assert_eq!(s, 5);
    }
}
