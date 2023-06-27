use ndarray::{prelude::*, Zip};
use ndarray_linalg::{eig::*, least_squares::*, svd::*};
use std::io::{self, Error, ErrorKind};

#[allow(dead_code)]
fn sensible_round(x: f64, n_digits: usize) -> f64 {
    let factor = ("1e".to_owned() + &n_digits.to_string())
        .parse::<f64>()
        .unwrap();
    (x * factor).round() / factor
}

pub fn multiply_views_xx(
    a: &Array2<f64>,
    b: &Array2<f64>,
    a_rows: &Vec<usize>,
    a_cols: &Vec<usize>,
    b_rows: &Vec<usize>,
    b_cols: &Vec<usize>,
) -> io::Result<Array2<f64>> {
    let n = a_rows.len();
    let m = b_cols.len();
    if a_cols.len() != b_rows.len() {
        return Err(Error::new(
            ErrorKind::Other,
            "The two matrices are incompatible.",
        ));
    }
    let mut out: Array2<f64> = Array2::zeros((n, m));
    let a_rows_mat = Array2::from_shape_vec((m, n), a_rows.repeat(m))
        .unwrap()
        .reversed_axes();
    let b_cols_mat = Array2::from_shape_vec((n, m), b_cols.repeat(n)).unwrap();
    Zip::from(&mut out)
        .and(&a_rows_mat)
        .and(&b_cols_mat)
        .par_for_each(|x, &a_i, &b_j| {
            for k in 0..a_cols.len() {
                let a_j = a_cols[k];
                let b_i = b_rows[k];
                *x += a[(a_i, a_j)] * b[(b_i, b_j)];
            }
        });
    Ok(out)
}

pub fn multiply_views_xtx(
    a: &Array2<f64>,
    b: &Array2<f64>,
    a_rows: &Vec<usize>,
    a_cols: &Vec<usize>,
    b_rows: &Vec<usize>,
    b_cols: &Vec<usize>,
) -> io::Result<Array2<f64>> {
    let n = a_cols.len(); // reversed a
    let m = b_cols.len();
    if a_rows.len() != b_rows.len() {
        // reversed a
        return Err(Error::new(
            ErrorKind::Other,
            "The two matrices are incompatible.",
        ));
    }
    let mut out: Array2<f64> = Array2::zeros((n, m));
    let a_cols_mat = Array2::from_shape_vec((m, n), a_cols.repeat(m))
        .unwrap()
        .reversed_axes();
    let b_cols_mat = Array2::from_shape_vec((n, m), b_cols.repeat(n)).unwrap();
    Zip::from(&mut out)
        .and(&a_cols_mat)
        .and(&b_cols_mat)
        .par_for_each(|x, &a_j, &b_j| {
            for k in 0..a_rows.len() {
                let a_i = a_rows[k];
                let b_i = b_rows[k];
                *x += a[(a_i, a_j)] * b[(b_i, b_j)];
            }
        });
    Ok(out)
}

pub fn multiply_views_xxt(
    a: &Array2<f64>,
    b: &Array2<f64>,
    a_rows: &Vec<usize>,
    a_cols: &Vec<usize>,
    b_rows: &Vec<usize>,
    b_cols: &Vec<usize>,
) -> io::Result<Array2<f64>> {
    let n = a_rows.len();
    let m = b_rows.len(); // reversed b
    if a_cols.len() != b_cols.len() {
        // reversed b
        return Err(Error::new(
            ErrorKind::Other,
            "The two matrices are incompatible.",
        ));
    }
    let mut out: Array2<f64> = Array2::zeros((n, m));
    let a_rows_mat = Array2::from_shape_vec((m, n), a_rows.repeat(m))
        .unwrap()
        .reversed_axes();
    let b_rows_mat = Array2::from_shape_vec((n, m), b_rows.repeat(n)).unwrap();
    Zip::from(&mut out)
        .and(&a_rows_mat)
        .and(&b_rows_mat)
        .par_for_each(|x, &a_i, &b_i| {
            for k in 0..a_cols.len() {
                let a_j = a_cols[k];
                let b_j = b_cols[k];
                *x += a[(a_i, a_j)] * b[(b_i, b_j)];
            }
        });
    Ok(out)
}

pub fn pinv(x: &Array2<f64>) -> io::Result<Array2<f64>> {
    let n: usize = x.nrows();
    let p: usize = x.ncols();
    let svd = x.svd(true, true).unwrap();
    let u: Array2<f64> = svd.0.unwrap();
    let s: Array1<f64> = svd.1;
    let vt: Array2<f64> = svd.2.unwrap();
    let mut s_inv: Array2<f64> = Array2::zeros((n, p));
    let tolerance: f64 =
        f64::EPSILON * (s.len() as f64) * s.fold(s[0], |max, &x| if x > max { x } else { max });
    for j in 0..p {
        if s[j] > tolerance {
            s_inv[(j, j)] = 1.0 / s[j];
        }
    }
    Ok(vt.t().dot(&s_inv.t()).dot(&u.t()))
}

pub fn ols(x: &Array2<f64>, y: &Array1<f64>, row_idx: &Vec<usize>) -> io::Result<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    assert!(
        n == y.len(),
        "Please check genotype and phenotype input as the number of observations are incompatible."
    );
    if x.column(0).sum() < n as f64 {
        return Err(Error::new(
            ErrorKind::Other,
            "Please add the intercept in the X matrix.",
        ));
    }
    let col_idx = &(0..p).collect::<Vec<usize>>();
    let new_row_idx = &(0..row_idx.len()).collect::<Vec<usize>>();
    let new_col_idx = &(0..1).collect::<Vec<usize>>();
    let b_hat: Array2<f64> = if n < p {
        multiply_views_xx(
            &multiply_views_xtx(
                x,
                &pinv(&multiply_views_xxt(x, x, row_idx, col_idx, row_idx, col_idx).unwrap())
                    .unwrap(),
                row_idx,
                col_idx,
                new_row_idx,
                new_row_idx,
            )
            .unwrap(),
            &(y.to_owned().into_shape((n, 1)).unwrap()),
            col_idx,
            new_row_idx,
            row_idx,
            new_col_idx,
        )
        .unwrap()
    } else {
        multiply_views_xx(
            &multiply_views_xxt(
                &pinv(&multiply_views_xtx(x, x, row_idx, col_idx, row_idx, col_idx).unwrap())
                    .unwrap(),
                x,
                col_idx,
                col_idx,
                row_idx,
                col_idx,
            )
            .unwrap(),
            &y.to_owned().into_shape((n, 1)).unwrap(),
            col_idx,
            new_row_idx,
            row_idx,
            new_col_idx,
        )
        .unwrap()
    };
    Ok(b_hat.column(0).to_owned())
}

pub fn ols_iterative_with_kinship_pca_covariate(
    x: &Array2<f64>,
    y: &Array1<f64>,
    row_idx: &Vec<usize>,
) -> io::Result<Array1<f64>> {
    let n = row_idx.len();
    let p = x.ncols();
    if x.column(0).sum() < x.nrows() as f64 {
        return Err(Error::new(
            ErrorKind::Other,
            "Please add the intercept in the X matrix.",
        ));
    }
    let col_idx = (0..p - 1).collect::<Vec<usize>>();
    let mut x_column_centred_no_intercept: Array2<f64> = Array2::from_elem((n, p - 1), f64::NAN);
    let rows_mat: Array2<usize> = Array2::from_shape_vec((p - 1, n), row_idx.repeat(p - 1))
        .unwrap()
        .reversed_axes();
    let cols_mat: Array2<usize> = Array2::from_shape_vec((n, p - 1), col_idx.repeat(n)).unwrap();
    Zip::from(&mut x_column_centred_no_intercept)
        .and(&rows_mat)
        .and(&cols_mat)
        .par_for_each(|x_new, &i, &j| {
            let mut mean = 0.0;
            for i_ in 0..n {
                mean += x[(i_, j)];
            }
            mean = mean / n as f64;
            *x_new = x[(i, j)] - mean;
        });
    let row_idx_new = (0..n).collect::<Vec<usize>>();
    let xxt: Array2<f64> = multiply_views_xxt(
        &x_column_centred_no_intercept,
        &x_column_centred_no_intercept,
        &row_idx_new,
        &col_idx,
        &row_idx_new,
        &col_idx,
    )
    .unwrap();
    let (_eigen_values, eigen_vectors): (Array1<_>, Array2<_>) = xxt.eig().unwrap();

    let mut b_hat: Array1<f64> = Array1::from_elem(p, f64::NAN);
    let mut y_sub: Array1<f64> = Array1::from_elem(n, f64::NAN);
    for i in 0..n {
        y_sub[i] = y[row_idx[i]];
    }
    let y_sub_mean: f64 = y_sub.mean().unwrap();
    let vec_j: Array1<usize> = Array1::from_shape_vec(p, (0..p).collect::<Vec<usize>>()).unwrap();

    Zip::from(&mut b_hat).and(&vec_j).par_for_each(|b, &j| {
        if j == 0 {
            *b = y_sub_mean;
        } else {
            let mut x_sub: Array2<f64> = Array2::ones((n, 3)); // intercept, 1st eigenvector, and the jth locus
            for i in 0..n {
                x_sub[(i, 1)] = eigen_vectors[(i, 0)].re; // extract the eigenvector value's real number component
                x_sub[(i, 2)] = x[(row_idx[i], j)]; // use the row_idx and add 1 to the column indexes to account for the intercept in the input x
            }
            *b = x_sub.least_squares(&y_sub).unwrap().solution[2]; // uing ndarray-linalg's built-in OLS
        }
    });
    Ok(b_hat)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use ndarray::concatenate;
    #[test]
    fn test_linalg() {
        // Inputs
        let a: Array2<f64> =
            Array2::from_shape_vec((5, 3), (0..15).map(|x| x as f64).collect::<Vec<f64>>())
                .unwrap();
        let b: Array2<f64> = Array2::from_shape_vec(
            (5, 3),
            (0..15).map(|x| x as f64 / 2.0).collect::<Vec<f64>>(),
        )
        .unwrap();
        let idx_w3: Vec<usize> = vec![1, 3, 4];
        let idx_x2: Vec<usize> = vec![0, 2];
        let idx_y2: Vec<usize> = vec![1, 3];
        let idx_z2: Vec<usize> = vec![0, 1];
        let intercept: Array2<f64> = Array2::ones((5, 1));

        let frequencies_wide = Array2::from_shape_vec(
            (5, 9),
            (1..46).map(|x| x as f64 / 45.0).collect::<Vec<f64>>(),
        )
        .unwrap();

        let frequencies_tall = Array2::from_shape_vec(
            (5, 2),
            (1..31)
                .step_by(3)
                .map(|x| x as f64 / 30.0)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let x_wide: Array2<f64> =
            concatenate(Axis(1), &[intercept.view(), frequencies_wide.view()]).unwrap();
        let x_tall: Array2<f64> =
            concatenate(Axis(1), &[intercept.view(), frequencies_tall.view()]).unwrap();

        let y: Array1<f64> =
            Array1::from_shape_vec(5, (1..6).map(|x| x as f64 / 5.0).collect::<Vec<f64>>())
                .unwrap();

        // Outputs
        let a_x_b = multiply_views_xx(&a, &b, &idx_w3, &idx_x2, &idx_y2, &idx_z2).unwrap();
        let at_x_b = multiply_views_xtx(&a, &b, &idx_w3, &idx_x2, &idx_w3, &idx_z2).unwrap();
        let a_x_bt = multiply_views_xxt(&a, &b, &idx_w3, &idx_x2, &idx_w3, &idx_z2).unwrap();
        let b_wide = ols(&x_wide, &y, &vec![0, 1, 2, 3, 4]).unwrap();
        let b_tall = ols(&x_tall, &y, &vec![0, 1, 2, 3, 4]).unwrap();
        let y_hat_wide = &x_wide.dot(&b_wide).mapv(|x| sensible_round(x, 4));
        let y_hat_tall = &x_tall.dot(&b_tall).mapv(|x| sensible_round(x, 4));
        let _b_wide_iterative =
            ols_iterative_with_kinship_pca_covariate(&x_wide, &y, &vec![0, 1, 2, 3, 4]).unwrap();
        // Assertions
        assert_eq!(
            a_x_b,
            Array2::from_shape_vec((3, 2), vec![27.0, 31.0, 63.0, 73.0, 81.0, 94.0]).unwrap()
        );
        assert_eq!(
            at_x_b,
            Array2::from_shape_vec((2, 2), vec![117.0, 129.0, 141.0, 156.0]).unwrap()
        );
        assert_eq!(
            a_x_bt,
            Array2::from_shape_vec(
                (3, 3),
                vec![14.5, 38.5, 50.5, 35.5, 95.5, 125.5, 46.0, 124.0, 163.0]
            )
            .unwrap()
        );
        assert_eq!(y, y_hat_wide);
        assert_eq!(y_hat_wide, y_hat_tall);
    }
}
