import numpy as np
from helpers.math import *
from helpers.data_processing import *
from implementations import *


# ---------- evaluation helpers ----------
def get_linear_pred(tx, w, threshold=0.5):
    """Predictions for linear models."""
    dummy_y = np.zeros(tx.shape[0])
    check_shapes(dummy_y, tx, w)

    y_pred = tx @ w
    return (y_pred >= threshold).astype(int)


def get_logistic_pred(tx, w, threshold=0.5):
    """Predictions for logistic models using sigmoid."""
    dummy_y = np.zeros(tx.shape[0])
    check_shapes(dummy_y, tx, w)

    y_pred = sigmoid(tx @ w)
    return (y_pred >= threshold).astype(int)


def get_accuracy(y_true, y_pred):
    """Compute accuracy (ratio of correct predictions)."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return np.mean(y_true.ravel() == y_pred.ravel())


def get_f1(y_true, y_pred):
    """Compute F1-score manually."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel().astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def pick_threshold_for_f1(y_val, p_val, num_steps=501):
    ts = np.linspace(0.0, 1.0, num_steps)
    best_t, best_f1 = 0.0, 0.0
    for t in ts:
        f1 = get_f1(y_val, (p_val >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def print_results_block(name, loss, acc, f1):
    """Nicely formatted printout for each method."""
    print("─" * 60)
    print(f"{name}")
    print(
        f"- Loss:      {loss:.6f}" if isinstance(loss, float) else f"-Loss:      {loss}"
    )
    print(f"- Accuracy:  {acc:.4f}")
    print(f"- F1-score:  {f1:.4f}")
    print("─" * 60 + "\n")


def k_fold_logistic(
    y,
    x,
    gam,
    lam=None,
    k=5,
    max_iters=1000,
    seed=12,
    verbose=False,
    newton=False,
    final_x_test=None,
):
    """
    Performs stratified k-fold cross-validation for (regularized) logistic regression.

    Parameters:
        y (np.ndarray): Target vector of shape (N,), with values in {0, 1}.
        x (np.ndarray): Feature matrix of shape (N, D).
        gam (float): Learning rate (gamma).
        lam (float, optional): Regularization strength (lambda). If None, unregularized logistic regression is used.
        k (int): Number of folds for cross-validation.
        max_iters (int): Maximum number of iterations for gradient descent or Newton's method.
        seed (int): Random seed to ensure reproducibility in fold splitting.
        verbose (bool): If True, prints progress and performance per fold.
        newton (bool): If True, uses Newton's method for regularized logistic regression.
        final_x_test (np.ndarray, optional): If provided, averages predictions on this test set across folds.

    Returns:
        mean_f1 (float): Average F1 score across folds.
        mean_threshold (float): Average best threshold chosen per fold.
        mean_prediction (np.ndarray or None): Ensemble test prediction if final_x_test is given, else None.
        mean_loss (float): Average loss across folds.
        mean_accuracy (float): Average accuracy across folds.
    """
    check_shapes(y, x)
    n, d = x.shape
    folds = stratified_k_indices(y, k, seed=seed)
    f1s = []
    losses = []
    accuracies = []
    thresholds = []
    k_predictions = []

    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_te, y_te = x[val_idx], y[val_idx]

        w0 = np.zeros(d)
        if lam is None:
            w, loss = logistic_regression(y_tr, x_tr, w0, max_iters, gam)
        elif not newton:
            w, loss = reg_logistic_regression(y_tr, x_tr, lam, w0, max_iters, gam)
        else:
            w, loss = regularized_logistic_regression_newton(
                y_tr, x_tr, lam, w0, max_iters, gam
            )

        # Find best threshold for this fold
        p_val = sigmoid(x_te @ w)
        best_threshold, f1 = pick_threshold_for_f1(y_te, p_val)
        (
            print(
                "for this fold : best threashold = ",
                best_threshold,
                ", giving F1 : ",
                f1,
            )
            if verbose
            else None
        )
        f1s.append(f1)
        thresholds.append(best_threshold)
        losses.append(loss)
        accuracies.append(
            get_accuracy(y_te, get_logistic_pred(x_te, w, best_threshold))
        )
        if final_x_test is not None:
            y_pred = get_logistic_pred(final_x_test, w, threshold=best_threshold)
            k_predictions.append(y_pred)

    mean_f1 = float(np.mean(f1s))
    mean_loss = float(np.mean(losses))
    mean_accuracy = float(np.mean(accuracies))
    mean_threashold = float(np.mean(thresholds))

    if final_x_test is not None:
        mean_prediction = (np.mean(np.vstack(k_predictions), axis=0) >= 0.5).astype(
            int
        )  # unfolding the k-prediction and getting the most present value per entry
    else:
        mean_prediction = None

    if verbose:
        if lam is None:
            print(f"γ={gam:.3g} → mean F1={mean_f1:.4f}")
        else:
            print(f"λ={lam:.3g}, γ={gam:.3g} → mean F1={mean_f1:.4f}")

    return mean_f1, mean_threashold, mean_prediction, mean_loss, mean_accuracy
