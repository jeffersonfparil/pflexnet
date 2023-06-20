use ndarray::prelude::*;
use std::io;

pub fn expand_and_contract(
    b_hat: &Array2<f64>,
    b_hat_proxy: &Array2<f64>,
    alpha: f64,
    lambda: f64,
) -> io::Result<Array2<f64>> {
    // Clone b_hat
    let mut b_hat: Array2<f64> = b_hat.clone();
    let (p, k) = (b_hat.nrows(), b_hat.ncols());
    for j in 0..k {
        // Exclude the intercept from penalisation
        let intercept = b_hat[(0, j)];
        // Norm 1 or norm 2 (exclude the intercept)
        let normed1: Array1<f64> = b_hat.column(j).slice(s![1..p]).map(|&x| x.abs());
        let normed2 = b_hat.column(j).slice(s![1..p]).map(|&x| x.powf(2.0));
        let normed = ((1.00 - alpha) * normed2 / 1.00) + (alpha * normed1);
        // Proxy norm 1 or norm 2 (exclude the intercept) for finding the loci that need to be penalised
        let normed1_proxy: Array1<f64> = b_hat_proxy.column(j).slice(s![1..p]).map(|&x| x.abs());
        let normed2_proxy = b_hat_proxy.column(j).slice(s![1..p]).map(|&x| x.powf(2.0));
        let normed_proxy = ((1.00 - alpha) * normed2_proxy / 1.00) + (alpha * normed1_proxy);
        // Find estimates that will be penalised using the proxy b_hat norms
        let normed_proxy_max =
            normed_proxy
                .iter()
                .fold(normed_proxy[0], |max, &x| if x > max { x } else { max });
        let normed_proxy_scaled: Array1<f64> = &normed_proxy / normed_proxy_max;
        let idx_penalised = normed_proxy_scaled
            .iter()
            .enumerate()
            .filter(|(_, &value)| value < lambda)
            .map(|(index, _)| index)
            .collect::<Vec<usize>>();
        let idx_depenalised = normed_proxy_scaled
            .iter()
            .enumerate()
            .filter(|(_, &value)| value >= lambda)
            .map(|(index, _)| index)
            .collect::<Vec<usize>>();
        // Penalise: contract using the non-proxy b_hat norms
        let mut subtracted_penalised = 0.0;
        let mut added_penalised = 0.0;
        for i in idx_penalised.into_iter() {
            if b_hat[(i + 1, j)] >= 0.0 {
                if (b_hat[(i + 1, j)] - normed[i]) < 0.0 {
                    subtracted_penalised += b_hat[(i + 1, j)];
                    b_hat[(i + 1, j)] = 0.0;
                } else {
                    subtracted_penalised += normed[i];
                    b_hat[(i + 1, j)] -= normed[i];
                };
            } else {
                if (b_hat[(i + 1, j)] + normed[i]) > 0.0 {
                    added_penalised += b_hat[(i + 1, j)].abs();
                    b_hat[(i + 1, j)] = 0.0;
                } else {
                    added_penalised += normed[i];
                    b_hat[(i + 1, j)] += normed[i];
                }
            }
        }
        // Find total depenalised (expanded) values
        let mut subtracted_depenalised = 0.0;
        let mut added_depenalised = 0.0;
        for i in idx_depenalised.clone().into_iter() {
            if b_hat[(i + 1, j)] >= 0.0 {
                subtracted_depenalised += normed[i];
            } else {
                added_depenalised += normed[i];
            }
        }

        // Account for the absence of available slots to transfer the contracted effects into
        if (subtracted_penalised > 0.0) & (subtracted_depenalised == 0.0) {
            added_penalised -= subtracted_penalised;
            subtracted_penalised = 0.0;
        } else if (added_penalised > 0.0) & (added_depenalised == 0.0) {
            subtracted_penalised -= added_penalised;
            added_penalised = 0.0;
        }
        // Depenalise: expand
        for i in idx_depenalised.into_iter() {
            if b_hat[(i + 1, j)] >= 0.0 {
                b_hat[(i + 1, j)] += subtracted_penalised * (normed[i] / subtracted_depenalised);
            } else {
                b_hat[(i + 1, j)] -= added_penalised * (normed[i] / added_depenalised);
            }
        }
        // Insert the unpenalised intercept
        b_hat[(0, j)] = intercept;
    }
    Ok(b_hat)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cv() {
        let b: Array2<f64> =
            Array2::from_shape_vec((7, 1), vec![5.0, -0.4, 0.0, 1.0, -0.1, 1.0, 0.0]).unwrap();
        let new_b: Array2<f64> = expand_and_contract(&b, &b, 1.00, 0.5).unwrap();
        let c: Array2<f64> =
            Array2::from_shape_vec((7, 1), vec![5.0, 0.4, 0.0, -1.0, 0.1, -1.0, 0.0]).unwrap();
        let new_c: Array2<f64> = expand_and_contract(&c, &c, 1.00, 0.5).unwrap();
        let expected_output1: Array2<f64> =
            Array2::from_shape_vec((7, 1), vec![5.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0]).unwrap();
        let expected_output2: Array2<f64> =
            Array2::from_shape_vec((7, 1), vec![5.0, 0.0, 0.0, -0.75, 0.0, -0.75, 0.0]).unwrap();
        assert_eq!(expected_output1, new_b);
        assert_eq!(expected_output2, new_c);
    }
}
