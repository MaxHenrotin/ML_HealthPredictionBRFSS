import numpy as np
from helpers.data_processing import *
from helpers.math import *

verbose = False  # Set to True to enable detailed print statements (Regularized Logistic Regression only)


# ---------------------Core functions -----------------------
def data_processing(
    x_train,
    y_train,
    x_test,
    normalize=False,
    remove_outliers=False,
    poly_degree=1,
    add_interactions=False,
    add_bias=False,
    verbose=True,
):
    """
    Process the raw dataset (e.g., handle missing values, normalization, feature expansion, etc.)
    """
    # Load and apply feature info
    feature_info = load_feature_info()
    keep_indices = feature_info["keep_indices"]
    nan_values = feature_info["nan_values"]
    zero_values = feature_info["zero_values"]

    # Align nan_values and zero_values with selected features
    nan_values = [nan_values[i] for i in keep_indices]
    zero_values = [zero_values[i] for i in keep_indices]
    # Keep only relevant + uncertain columns
    x_train = np.asarray(x_train[:, keep_indices], dtype=float)
    x_test = np.asarray(x_test[:, keep_indices], dtype=float)

    # Transform data from (-1/1) to (0,1)
    y_train = (y_train + 1) / 2

    # Replace NaN and 0 equivalents
    x_train, x_test = replace_equivalent_values(
        x_train, x_test, nan_values, zero_values
    )
    print("Replaced NaN and zero-equivalent values.")

    # Drop columns with too many missing values
    (
        print("Shape before dropping high missingness cols:", x_train.shape)
        if verbose
        else None
    )
    x_train, x_test = drop_high_missingness_cols(x_train, x_test, threshold=0.6)
    print("Dropped columns with >60% missing values.")
    (
        print("Shape after dropping high missingness cols:", x_train.shape)
        if verbose
        else None
    )

    # Detect feature types
    feature_types = detect_feature_types(x_train, categorical_threshold=10)
    categorical_idx = feature_types["categorical"]
    continuous_idx = feature_types["continuous"]

    # Impute missing values
    x_train = impute_data(x_train, categorical_idx, continuous_idx)
    x_test = impute_data(x_test, categorical_idx, continuous_idx)
    print("Imputed missing values.")

    # Optionally clip outliers (continuous features only)
    if remove_outliers:
        x_train, x_test = clip_outliers(x_train, x_test, continuous_idx)
        print("Clipped outliers in continuous features.")

    # Polynomial expansion (continuous features only)
    if poly_degree > 1 and continuous_idx:
        x_train, x_test = polynomial_expand(
            x_train, x_test, continuous_idx, degree=poly_degree
        )
        print(f"Added polynomial features (degree={poly_degree}).")
        print("New shape: {x_train.shape}") if verbose else None

    # Add pairwise interactions (optional)
    if add_interactions and continuous_idx:
        x_train, x_test = add_interaction_features(x_train, x_test, continuous_idx)
        print(f"Added interaction terms.")
        print(f"New shape: {x_train.shape}") if verbose else None

    # Optionally normalize only continuous features (z-score)
    if normalize:
        d_final = x_train.shape[1]
        cat_set = set(categorical_idx)
        continuous_idx_final = [i for i in range(d_final) if i not in cat_set]

        if continuous_idx_final:
            # Standardize only continuous columns
            x_train_cont = x_train[:, continuous_idx_final]
            x_train_cont, mean, std = standardize(x_train_cont)

            # Replace continuous columns in x_train
            x_train[:, continuous_idx_final] = x_train_cont

            # Apply same normalization to test set
            x_test_cont = x_test[:, continuous_idx_final]
            x_test_cont = (x_test_cont - mean) / std
            x_test[:, continuous_idx_final] = x_test_cont
            print("Standardized continuous features.")

    # Add bias column
    if add_bias:
        x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
        x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        print("Added bias.")

    # Step 3: Return processed arrays
    return x_train, y_train, x_test


# ------------------- 1 : Gradient Descent -------------------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using (full-batch) gradient descent.

    Args:
        y: target vector of shape (N,)
        tx: input matrix of shape (N, D)
        initial_w: initial weights (D,)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: optimal weights
        loss: final MSE loss
    """
    check_shapes(y, tx, initial_w)
    w = initial_w.copy()
    loss = np.inf

    for iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    loss = compute_loss(y, tx, w)
    return w, loss


# ------------------- 2 : Stochastic Gradient Descent -------------------
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent (batch size = 1).

    Args:
        y: target vector of shape (N,)
        tx: input matrix of shape (N, D)
        initial_w: initial weights (D,)
        max_iters: number of SGD updates
        gamma: learning rate

    Returns:
        w: optimal weights
        loss: final MSE loss
    """
    check_shapes(y, tx, initial_w)
    w = initial_w.copy()
    loss = np.inf

    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1, shuffle=True):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w -= gamma * grad

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    loss = compute_loss(y, tx, w)
    return w, loss


# ------------------- 3 : Least Squares -------------------
def least_squares(y, tx):
    """
    Linear regression using the normal equations.

    Args:
        y: target vector (N,)
        tx: input matrix (N,D)

    Returns:
        w: optimal weights
        loss: 0.5 * MSE loss
    """
    check_shapes(y, tx)
    # SVD: X = U @ diag(s) @ Vt   with U (N,r), s (r,), Vt (r,D)
    U, s, Vt = np.linalg.svd(tx, full_matrices=False)

    # Moore–Penrose via thresholded inverse of s
    # same tolerance style as numpy does internally
    tol = np.finfo(float).eps * max(tx.shape) * (s[0] if s.size else 0.0)
    s_inv = np.where(s > tol, 1.0 / s, 0.0)

    # w = V @ diag(s_inv) @ U^T @ y
    w = (Vt.T * s_inv) @ (U.T @ y)

    loss = compute_loss(y, tx, w)
    return w, loss


# ------------------- 4 : Ridge Regression -------------------
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using the normal equations.

    Args:
        y: target vector (N,)
        tx: input matrix (N, D)
        lambda_: regularization parameter

    Returns:
        w: optimal weights
        loss: 0.5 * MSE loss (without penalty term)
    """
    check_shapes(y, tx)
    N, D = tx.shape
    A = tx.T @ tx + (2.0 * lambda_ * N) * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


# ------------------- 5 : Logistic Regression -------------------
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y: binary target vector in {0,1} of shape (N,)
        tx: input matrix (N, D)
        initial_w: initial weights (D,)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: optimal weights
        loss: negative log-likelihood (average) -> should not include penalty term
    """
    check_shapes(y, tx, initial_w)
    w = initial_w.copy()
    loss = np.inf

    # gradient descent
    # we could also add a treashold to stop the loop when we estimate that it is not changing enough
    for iter in range(max_iters):
        loss = calculate_logistic_loss(y, tx, w)
        w -= gamma * calculate_logistic_gradient(y, tx, w)

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    loss = calculate_logistic_loss(y, tx, w)
    return w, loss


# ------------------- 6 : Regularized Logistic Regression -------------------
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Args:
        y: binary target vector in {0,1} of shape (N,)
        tx: input matrix (N, D)
        lambda_: L2 regularization parameter
        initial_w: initial weights (D,)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: optimal weights
        loss: negative log-likelihood (average, without reg term) -> should not include penalty term
    """
    check_shapes(y, tx, initial_w)
    w = initial_w.copy()
    loss = np.inf
    n, d = tx.shape

    # don't penalize bias if first column is ones
    reg_mask = np.ones(d, dtype=float)
    if np.allclose(tx[:, 0], 1.0):
        reg_mask[0] = 0.0

    prev_loss = np.inf
    tol = 1e-8

    # gradient descent
    # we could also add a treashold to stop the loop when we estimate that it is not changing enough
    for iter in range(max_iters):
        loss = calculate_logistic_loss(y, tx, w)
        grad = calculate_logistic_gradient(y, tx, w) + 2.0 * lambda_ * (reg_mask * w)
        w -= gamma * grad

        if verbose and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    loss = calculate_logistic_loss(y, tx, w)
    return w, loss


# ------------------- 7 : Regularized Logistic Regression using newton's method-------------------
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a hessian matrix of shape=(D, D)
    """
    check_shapes(y, tx, w)
    n = tx.shape[0]
    s = sigmoid(tx @ w)
    r = s * (1.0 - s)

    H = (tx.T * r) @ tx / n
    return H


# we had to put the regularizer because the hessian was ill conditionned
def regularized_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y: binary target vector in {0,1} of shape (N,)
        tx: input matrix (N, D)
        initial_w: initial weights (D,)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: optimal weights
        loss: negative log-likelihood (average) -> should not include penalty term
    """
    check_shapes(y, tx, initial_w)
    w = initial_w.copy()
    n, d = tx.shape

    reg_mask = np.ones(d, dtype=float)
    if np.allclose(tx[:, 0], 1.0):
        reg_mask[0] = 0.0

    prev_loss = np.inf
    tol = 1e-8
    # gradient descent
    # we could also add a treashold to stop the loop when we estimate that it is not changing enough
    for iter in range(max_iters):
        loss = calculate_logistic_loss(y, tx, w)
        grad = calculate_logistic_gradient(y, tx, w) + 2.0 * lambda_ * (reg_mask * w)
        hessian = calculate_hessian(y, tx, w) + 2.0 * lambda_ * np.diag(
            reg_mask
        )  # prevents ill conditionning
        # print(np.linalg.cond(hessian)) #to check if the hessian is ill conditionned (>10¹⁰ means ill conditionned)

        w -= gamma * np.linalg.solve(hessian, grad)

        if verbose and iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    loss = calculate_logistic_loss(y, tx, w)
    return w, loss
