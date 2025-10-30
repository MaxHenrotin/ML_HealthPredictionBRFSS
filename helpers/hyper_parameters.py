import numpy as np
from helpers.evaluations import k_fold_logistic


def reg_logistic_grid_search(
    y, x, gammas, lambdas, k=5, max_iters=1000, seed=12, verbose=False, newton=False
):
    """
    Performs a grid search over combinations of regularization strengths (lambda)
    and learning rates (gamma) for regularized logistic regression.

    Parameters:
        y (np.ndarray): Target vector of shape (N,), with values in {0, 1}.
        x (np.ndarray): Feature matrix of shape (N, D).
        gammas (iterable of float): Learning rates to evaluate.
        lambdas (iterable of float): Regularization parameters to evaluate.
        k (int): Number of folds for stratified cross-validation (default: 5).
        max_iters (int): Maximum number of iterations (default: 1000).
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, prints testing progress.
        newton (bool): If True, use Newton’s method instead of gradient descent.

    Returns:
        Tuple[float, float]: The best (lambda, gamma) pair yielding the highest mean F1 score.
    """
    best = (-1.0, None, None)  # (mean_f1, best_lambda, best_gamma)

    for lam in lambdas:
        for gam in gammas:
            if verbose:
                print(f"Testing λ={lam:.3g}, γ={gam:.3g}...")
            mean_f1 = k_fold_logistic(
                y, x, gam, lam, k, max_iters, seed, verbose, newton
            )[0]
            if mean_f1 > best[0]:
                best = (mean_f1, lam, gam)

    best_f1, best_lambda, best_gamma = best
    if verbose:
        print(f"\nBest: λ={best_lambda:.6g}, γ={best_gamma:.6g}, F1={best_f1:.4f}")
    return best_lambda, best_gamma


def logistic_grid_search(y, x, gammas, k=5, max_iters=1000, seed=12, verbose=False):
    """
    Performs grid search over different learning rates (gamma) for logistic regression
    using stratified k-fold cross-validation.

    Parameters:
        y (np.ndarray): Target vector of shape (N,), with values in {0, 1}.
        x (np.ndarray): Feature matrix of shape (N, D).
        gammas (iterable of float): Learning rates to evaluate.
        k (int): Number of folds for cross-validation (default: 5).
        max_iters (int): Maximum number of iterations for gradient descent (default: 1000).
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, prints progress and performance.

    Returns:
        best_gamma (float): Gamma value that yielded the best average F1 score.
    """
    best = (-1.0, None)  # (mean_f1, best_gamma)

    for gam in gammas:
        if verbose:
            print(f"Testing γ={gam:.3g}...")
        mean_f1 = k_fold_logistic(y, x, gam, None, k, max_iters, seed, verbose)[0]
        if mean_f1 > best[0]:
            best = (mean_f1, gam)

    best_f1, best_gamma = best
    if verbose:
        print(f"\nBest: γ={best_gamma:.6g}, F1={best_f1:.4f}")
    return best_gamma
