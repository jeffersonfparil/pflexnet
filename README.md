# pflexnet

**Pseudo-orthonormal elastic-net-like regularisation**: attempting to correct for non-orthonormality after soft-thresholding-based regularisation

|**Build Status**|**License**|
|:---:|:---:|
| <a href="https://github.com/jeffersonfparil/pflexnet/actions"><img src="https://github.com/jeffersonfparil/pflexnet/actions/workflows/rust.yml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

Let the linear model be:

$$
y = X\beta + \epsilon
$$

where $y$ is a vector of $n$ observations (*known*),
$X$ is the $n \times p$ matrix of explantory variables (*known*),
$\beta$ is a vector of $p$ explanatory effects (also called estimators or predictors; *unknown*), and
$\epsilon$ is the vector of residual effects (*unknown*), i.e.

$$
\epsilon \sim N(0, \sigma)
$$

hence

$$
\bar y = X\beta
$$

The objective is to estimate the explanatory effects, $\hat\beta$. 
The straightforward solution is called the ordinary least squares (OLS) solution, 
which minimises the squared differences in the predicted and observed phenotypes (cost function), i.e. 

$$
argmin_{\hat\beta} (X\hat\beta - y)^{2}
$$

OLS has a closed-form solution ($\hat\beta$ are estimated without the need for iterative optimisation minimising the above cost function), i.e. 

$$
\hat\beta = (X^{T}X)^{-1} X^{T}y
$$

Alternatively,

$$
\hat\beta = X^{T} (XX^{T})^{+}y
$$

where $A^{+}$ is the pseudoinverse of $A$.

When $n << p$, i.e. underdetermined cases,
more than 1 solution ($\hat\beta$) exists which leads to overfitting hence worse models,
i.e. not capturing the true explanatory effects. 
In these cases, the estimated effects ($\hat\beta$) need to be penalised, 
i.e. reduced to give less weights to unimportant explanatory variables. 
Ridge, least absolute shrinkage and selection operator (Lasso), and elasticnet regression are such penalised regression techniques. 
Unlike OLS, in addition to minimising the squared errors the magnitude of the explantory effects are penalised using a penalisation factor, $\lambda \in (0, +\infty)$. 
The cost function for ridge is

$$
argmin_{\hat\beta, \lambda} (X\hat\beta - y)^{2} + \lambda(\hat\beta)^{2}
$$

for Lasso it is

$$
argmin_{\hat\beta, \lambda} (X\hat\beta - y)^{2} + \lambda||\hat\beta||
$$

where $\|\|A\|\|$ is the absolute value of $A$, and elasticnet combines these two penalisations using an elastic parameter, $\alpha \in [0, 1]$ with the form

$$
argmin_{\hat\beta, \alpha, \lambda} (X\hat\beta - y)^{2} + \lambda \left ( {1-\alpha \over 2} (\hat\beta)^{2} + \alpha ||\hat\beta|| \right )
$$

**Note that the effect of the intercept is not penalised**. These models can be solved using iterative algorithms, i.e. iteratively testing various values of $\hat\beta$, $\lambda$, and $\alpha$ which minimise the cost functions 
with the next values based on how well moving across the parameter space reduces the cost functions. This method is computationally expensive, thankfully closed-form solutions exist. 
Ridge regression has the closed-form solution,

$$
\hat\beta = (X^{T} X + \lambda I)^{-1} X^{T}y
$$

This lends to less expensive iterative solutions which will only need to find the best $\lambda$ using k-fold cross-validation, 
i.e. using a set of $\lambda$, divide the $n$ observations into $k$ groups or folds, 
(1) set aside one group for validation and using the rest of the groups to solve for $\hat\beta$ (measure the performance of the model using the validation group - cross-validation), 
(2) repeat for the rest of the folds,
(3) reshuffle, generate a new grouping, and repeat for k-fold cross-validation, 
(4) repeat the k-fold cross-validation $r$ times.

On the otherhand, a closed-form solution for Lasso only exists if $X$ is orthonormal ($X^{T} = X^{-1}$ and $\|\|X\|\|=1$ hence $\hat\beta = X^{T}y$), i.e.

$$
\hat\beta_j = sign(\hat\beta^{OLS}_j) max(\{ \hat\beta^{OLS}_j - \lambda, 0 \})
$$

and by extension the closed-form solution of elasticnet if $X$ is orthonormal is

$$
\hat\beta_j = sign(\hat\beta^{OLS}_j) { max \left( { \hat\beta^{OLS}_j - \alpha \lambda \over 1 + 0.5 (1-\alpha)\lambda}, ~~ 0  \right) }
$$

For derivation details see [lasso closed-form solution](https://stats.stackexchange.com/questions/17781/derivation-of-closed-form-lasso-solution), and [elasticnet closed-form solution](https://myweb.uiowa.edu/pbreheny/7600/s16/notes/3-28.pdf).

## Attempting to find a more efficient algorithm for elasticnet when $X$ is not orthonormal

My idea is to use a similar soft-thresholding solution to the closed-form solution to lasso/elasticnet 
followed by corrections for non-orthonormality by redistributing the penalised effects into the non-penalised variables 
so we end up with the correct $X\hat\beta$.

The algorithm in development can be described as follows:

1. Define the parameter paths, i.e. $\alpha \in [0, 1]$ and $\lambda \in [0, 1]$
2. Define the k-folds and number of shuffling and k-fold cross-validation replications
3. Solve for the OLS solution (using the more efficient alternative form when $n << p$)

$$
\hat\beta^{OLS} = X^{T} (XX^{T})^{+} y
$$

4. Compute the rige and lasso norms of the OLS estimates

$$
b^{ridge} = \|\|\hat\beta^{OLS}\|\|^{2}
$$

and

$$
b^{lasso} = \|\|\hat\beta^{OLS}\|\|
$$

5. Implement the elasticnet parameter

$$
b_j = (1-\alpha) b^{ridge} + \alpha b^{lasso}
$$

6. Map these elasticnet norms between zero and one

$$
z_j = { b_j - min(b) \over max(b) - min(b) }
$$

7. Penalise:

$$
\hat\beta_j = sign(\hat\beta^{OLS}_j) max \left ( || \hat\beta^{OLS}_j || - b_j, ~~ 0 \right ) ~, ~~ if ~~ z_j < \lambda
$$

8. Account for the total magnitude of the penalised effects ($\gamma$), i.e. total the penalised positive effects ($\gamma_p$) minus the total penalised negative effects ($\gamma_n$)

$$
\gamma = \gamma_p - \gamma_n
$$
  
10. Distribute the resulting total positive or negative effects ($\gamma$) into the unpenalised effects according to their magnitudes relative to the total unpenalised effects ($B$)

$$
\hat\beta_j = sign(\hat\beta^{OLS}_j) \left ( || \hat\beta^{OLS}_j || + \gamma {\hat\beta^{OLS}_j \over B} \right ) ~, ~~ if ~~ z_j \ge \lambda
$$

## Next steps

- Create a self-standing implementation with an R API. See [poolgen/src/gp/penalise.rs](https://github.com/jeffersonfparil/poolgen/blob/main/src/gp/penalise.rs) for the current implementaion used with pool sequencing data.
- Improve the algorithm for better computational efficiency and accuracy (specifically account for overfitting in some cross-fold validation runs which seems to be reducing the performance of elasticnet, e.g. filter out parameter values ($\lambda$ and $\alpha$) which resulted in overfitting, or probably use the mode of the parameters across the replicated k-fold cross-validation runs)
- Determine how well this performs compared with ridge, Lasso, and elasticnet using the [glmnet package](https://glmnet.stanford.edu/articles/glmnet.html)
- Layout the mathematical basis if it turns out to be a good penalisation alternative
