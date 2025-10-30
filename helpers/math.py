import numpy as np


def check_shapes(y, tx, w=None):
    """
    Validates the shapes and types of input arrays for consistency.

    Args:
        y (np.ndarray): Target vector of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        w (np.ndarray, optional): Weight vector of shape (D,). Defaults to None.

    Raises:
        ValueError: If any of the following conditions are not met:
            - tx is a 2D NumPy array
            - y is a 1D NumPy array
            - w (if provided) is a 1D NumPy array
            - y and tx have the same number of samples (N)
            - w (if provided) has the same number of features as tx (D)
    """
    if not isinstance(tx, np.ndarray) or tx.ndim != 2:
        raise ValueError(
            f"tx must be 2D (N, D). Got {None if not isinstance(tx, np.ndarray) else tx.shape}."
        )
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(
            f"y must be 1D (N,). Got {None if not isinstance(y, np.ndarray) else y.shape}."
        )
    if w is not None and (not isinstance(w, np.ndarray) or w.ndim != 1):
        raise ValueError(
            f"w must be 1D (D,). Got {None if not isinstance(w, np.ndarray) else w.shape}."
        )
    N, D = tx.shape
    if y.shape[0] != N:
        raise ValueError(f"len(y)={y.shape[0]} but tx has N={N}.")
    if w is not None and w.shape[0] != D:
        raise ValueError(f"len(w)={w.shape[0]} but tx has D={D}.")


def sigmoid(z):
    """
    Numerically stable sigmoid function.
    Notes:
        - Uses a numerically stable implementation to avoid overflow when z is large or small.
        - Formula:
            sigmoid(z) = 1 / (1 + exp(-z))  for z >= 0
                       = exp(z) / (1 + exp(z))  for z < 0
    """
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def compute_loss(y, tx, w, MAE=False):
    """
    Calculate the loss using either MSE or MAE.
    Args:
    y:  (N,)
    tx: (N,D)
    w:  (D,)
    Returns:
    loss: scalar
    """
    check_shapes(y, tx, w)
    e = y - tx @ w
    if MAE:
        return float(np.mean(np.abs(e)))
    return (e.T @ e) / (2 * y.shape[0])


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    check_shapes(y, tx, w)

    N = y.shape[0]
    e = y - tx @ w
    return -(tx.T @ e) / N


def calculate_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss
    """
    check_shapes(y, tx, w)
    n = y.shape[0]

    epsilon = 1e-15  # to ensure no log(0)
    sig = sigmoid(tx @ w)
    loss = (
        -1 / n * np.sum(y * np.log(sig + epsilon) + (1 - y) * np.log(1 - sig + epsilon))
    )
    return loss


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss (negative log likelyhood).

    Args:
        y:  shape=(N,)
        tx: shape=(N,D)
        w:  shape=(D,)

    Returns:
        grad: shape=(D,)
    """
    check_shapes(y, tx, w)

    n = y.shape[0]
    sig = sigmoid(tx @ w)
    grad = ((1 / n) * tx.T) @ (sig - y)
    return grad


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
    x: numpy array of shape (N,)
    degree: integer.

    Returns:
    poly: numpy array of shape (N,d+1)

    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D (N,). Got shape {x.shape}.")
    if degree < 0 or int(degree) != degree:
        raise ValueError("degree must be a non-negative integer.")
    return np.vander(x, degree + 1, increasing=True)


def standardize(x):
    """Standardize per feature (column-wise)."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    x_std = (x - mean) / std
    return x_std, mean, std
