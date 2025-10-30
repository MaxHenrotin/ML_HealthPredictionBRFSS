import numpy as np
import json


def split_data(x_train, y_train, training_ratio, seed=12, stratify=True):
    """
    Splits the dataset into training and test sets.

    Parameters:
        x_train (np.ndarray): Feature matrix of shape (N, D).
        y_train (np.ndarray): Target vector of shape (N,) or (N, 1).
        training_ratio (float): Proportion of data to use for training (between 0 and 1).
        seed (int): Random seed for reproducibility.
        stratify (bool): If True, preserves the class distribution in both splits.

    Returns:
        x_tr (np.ndarray): Training features.
        y_tr (np.ndarray): Training labels.
        x_te (np.ndarray): Test features.
        y_te (np.ndarray): Test labels.
    """
    N = x_train.shape[0]
    rng = np.random.default_rng(seed)

    if not stratify:
        indices = rng.permutation(N)
        split = int(training_ratio * N)
        train_idx = indices[:split]
        test_idx = indices[split:]
    else:
        y = y_train.ravel().astype(int)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        split_pos = int(training_ratio * len(pos_idx))
        split_neg = int(training_ratio * len(neg_idx))

        train_idx = np.concatenate([pos_idx[:split_pos], neg_idx[:split_neg]])
        test_idx = np.concatenate([pos_idx[split_pos:], neg_idx[split_neg:]])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    x_tr, x_te = x_train[train_idx], x_train[test_idx]
    y_tr, y_te = y_train[train_idx], y_train[test_idx]
    return x_tr, y_tr, x_te, y_te


def drop_high_missingness_cols(x_train, x_test, threshold=0.5):
    """
    Drop features (columns) with too many missing values.
    """
    missing_fraction = np.mean(np.isnan(x_train), axis=0)
    keep_mask = missing_fraction <= threshold
    return x_train[:, keep_mask], x_test[:, keep_mask]


def impute_data(x, categorical_idx, continuous_idx):
    """
    Fill missing (NaN) values in the dataset based on feature type.

    - Continuous features: replaced with the median (robust to outliers)
    - Categorical features (including binary): replaced with the most frequent value (mode)

    Args:
        x (np.ndarray): Feature matrix of shape (N, D)
        categorical_idx (list[int]): indices of categorical features
        continuous_idx (list[int]): indices of continuous features

    Returns:
        np.ndarray: Array with missing values replaced.
    """
    # Impute continuous features (median)
    for j in continuous_idx:
        col = x[:, j]
        if np.isnan(col).any():
            median_val = np.nanmedian(col)
            x[np.isnan(col), j] = median_val

    # Impute categorical features (most frequent value)
    for j in categorical_idx:
        col = x[:, j]
        if np.isnan(col).any():
            valid_values = col[~np.isnan(col)]
            if len(valid_values) == 0:
                continue  # skip if all are NaN
            values, counts = np.unique(valid_values, return_counts=True)
            mode_val = values[np.argmax(counts)]
            x[np.isnan(col), j] = mode_val

    return x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """

    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def detect_feature_types(X: np.ndarray, categorical_threshold: int = 10):
    """
    Detect categorical and continuous feature indices in a numerical dataset.
    - Categorical: integer-valued columns with few unique values (<= categorical_threshold).
    - Continuous: columns with many unique or non-integer values.
    """
    categorical_idx = []
    continuous_idx = []

    for j in range(X.shape[1]):
        col = X[:, j]
        col_nonan = col[~np.isnan(col)]
        if col_nonan.size == 0:
            continue

        unique_vals = np.unique(col_nonan)
        n_unique = len(unique_vals)

        # Binary or few integer categories → categorical
        if n_unique <= categorical_threshold and np.allclose(
            unique_vals, np.round(unique_vals)
        ):
            categorical_idx.append(j)
        else:
            continuous_idx.append(j)

    return {"categorical": categorical_idx, "continuous": continuous_idx}


def stratified_k_indices(y, k, seed=12):
    """
    Splits the dataset into k folds for stratified k-fold cross-validation.
    Ensures that each fold has approximately the same proportion of positive and negative labels.

    Parameters:
        y (np.ndarray): Target vector of shape (N,) or (N, 1), with binary labels {0,1}.
        k (int): Number of folds to create.
        seed (int): Random seed for reproducibility.

    Returns:
        folds (List[np.ndarray]): List of k arrays, each containing indices for one fold.
    """
    rng = np.random.default_rng(seed)
    y = y.astype(int).ravel()
    pos = rng.permutation(np.where(y == 1)[0])
    neg = rng.permutation(np.where(y == 0)[0])
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    folds = []
    for i in range(k):
        fold = np.concatenate([pos_folds[i], neg_folds[i]])
        rng.shuffle(fold)
        folds.append(fold)

    return folds


def load_feature_info(feature_info_path="feature_info.json"):
    """
    Load feature info (relevant indices, uncertain indices, NaN and zero equivalents)
    from a JSON file created by excel_reader.py.
    """
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)
    relevant_idx = feature_info["relevant_indices"]
    uncertain_idx = feature_info["uncertain_indices"]
    nan_values = feature_info["nan_values"]
    zero_values = feature_info["zero_values"]

    keep_indices = sorted(set(relevant_idx + uncertain_idx))

    return {
        "keep_indices": keep_indices,
        "nan_values": nan_values,
        "zero_values": zero_values,
    }


def replace_equivalent_values(x_train, x_test, nan_values, zero_values):
    """
    Replace listed NaN and 0 equivalent values in the dataset.
    """
    n_features = x_train.shape[1]
    for j in range(n_features):
        nan_list = set(nan_values[j])
        zero_list = set(zero_values[j])

        # Replace NaN equivalents
        if nan_list:
            mask_train_nan = np.isin(x_train[:, j], list(nan_list))
            x_train[mask_train_nan, j] = np.nan
            mask_test_nan = np.isin(x_test[:, j], list(nan_list))
            x_test[mask_test_nan, j] = np.nan

        # Replace 0 equivalents
        if zero_list:
            mask_train_zero = np.isin(x_train[:, j], list(zero_list))
            x_train[mask_train_zero, j] = 0
            mask_test_zero = np.isin(x_test[:, j], list(zero_list))
            x_test[mask_test_zero, j] = 0

    return x_train, x_test


def clip_outliers(x_train, x_test, continuous_idx):
    """
    Clip outliers in continuous features using 5th/95th percentiles.
    """
    if not continuous_idx:
        return x_train, x_test

    for j in continuous_idx:
        col_train = x_train[:, j]

        # Compute percentiles robustly (ignore NaNs)
        q1, q3 = np.nanpercentile(col_train, [5, 95])
        iqr = q3 - q1

        # Define bounds
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Clip training values
        x_train[:, j] = np.clip(col_train, lower, upper)

        # Clip test values using same bounds
        if x_test is not None:
            x_test[:, j] = np.clip(x_test[:, j], lower, upper)

    return x_train, x_test


def polynomial_expand(x_train, x_test, continuous_idx, degree=2):
    """
    Expand only continuous features to the given polynomial degree.
    """
    x_train_cont = x_train[:, continuous_idx]
    x_test_cont = x_test[:, continuous_idx]

    poly_train = np.column_stack([x_train_cont**d for d in range(2, degree + 1)])
    poly_test = np.column_stack([x_test_cont**d for d in range(2, degree + 1)])

    # Appent augmented continuous features to the original arrays
    x_train_out = np.hstack([x_train, poly_train])
    x_test_out = np.hstack([x_test, poly_test])
    return x_train_out, x_test_out


def add_interaction_features(x_train, x_test, continuous_idx):
    """
    Add pairwise interaction features (x_i * x_j) for continuous columns only.
    """
    cont_train = x_train[:, continuous_idx]
    cont_test = x_test[:, continuous_idx]
    n = cont_train.shape[1]

    interactions_train = []
    interactions_test = []
    for i in range(n):
        for j in range(i + 1, n):
            interactions_train.append(cont_train[:, i] * cont_train[:, j])
            interactions_test.append(cont_test[:, i] * cont_test[:, j])

    if interactions_train:
        x_train = np.column_stack([x_train] + interactions_train)
        x_test = np.column_stack([x_test] + interactions_test)

    return x_train, x_test


def sanity_check_no_nan(x_train, y_train=None, x_test=None):
    """Verify that there are no NaN or Inf values in the datasets."""
    assert np.isfinite(x_train).all(), "x_train contains NaN or Inf values!"
    if y_train is not None:
        assert np.isfinite(y_train).all(), "y_train contains NaN or Inf values!"
    if x_test is not None:
        assert np.isfinite(x_test).all(), "x_test contains NaN or Inf values!"
    print("Sanity check passed — no NaN or Inf in data.")
