import numpy as np
import time
from implementations import *
from helpers.data_processing import *
from helpers.math import *
from helpers.evaluations import *
from helpers.csv_helpers import *
from helpers.hyper_parameters import *


# ------------- Configuration Parameters ------------------
DATA_PATH = "dataset/"  # Path to the dataset folder
submission = True  # If True, generate the submission CSV file
do_cross_validation = False  # If True, perform k-fold cross-validation
sub_sample = False  # If True, use a smaller data subset for faster testing
seed = 16  # Random seed for reproducibility


# ------------- Data Processing Parameters ------------------
normalize = True  # Standardize continuous features (mean=0, std=1)
remove_outliers = True  # Clip extreme values to reduce noise
poly_degree = 3  # Add polynomial features up to degree 3
add_interactions = True  # Do not add pairwise interaction terms
add_bias = True  # Add a bias column (constant 1) to features


def main():
    # -------- Load raw data --------
    print("Loading data...")
    t0 = time.time()
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = load_csv_data(
        DATA_PATH, sub_sample=sub_sample
    )
    print(f"Loaded in {time.time() - t0:.2f}s")
    print("x_train shape (raw):", x_train_raw.shape)

    # -------- Process data --------
    print("Processing data...")
    x_train, y_train, x_test = data_processing(
        x_train_raw,
        y_train=y_train_raw,
        x_test=x_test_raw,
        normalize=normalize,
        remove_outliers=remove_outliers,
        poly_degree=poly_degree,
        add_interactions=add_interactions,
        add_bias=add_bias,
    )
    print("x_train shape (processed):", x_train.shape)

    # split the data for testing (and thus detect overfitting)
    x_train_split, y_train_split, x_test_split, y_test_split = split_data(
        x_train, y_train, 0.8, seed=seed, stratify=True
    )
    print("x_train shape (after splitting data):", x_train_split.shape)

    n_split, d_split = x_train_split.shape
    sanity_check_no_nan(x_train_split, y_train_split, x_test)
    n, d = x_train.shape
    sanity_check_no_nan(x_train, y_train, x_test)

    # -------- Run methods --------
    results = []

    # 1 Mean Squared Error - Gradient Descent
    try:
        print("Running mean_squared_error_gd...")
        max_iters = 500
        gamma = 1e-3

        w0 = np.zeros(d_split)
        w, loss = mean_squared_error_gd(
            y_train_split, x_train_split, w0, max_iters, gamma
        )
        y_pred = get_linear_pred(x_test_split, w)
        acc, f1 = get_accuracy(y_test_split, y_pred), get_f1(y_test_split, y_pred)
        results.append(("MSE Gradient Descent", loss, acc, f1))
    except Exception as e:
        results.append(("MSE Gradient Descent", f"ERROR: {e}", 0, 0))

    # 2 Mean Squared Error - SGD
    try:
        print("Running mean_squared_error_sgd...")
        max_iters = 500
        gamma = 1e-3

        w0 = np.zeros(d_split)
        w, loss = mean_squared_error_sgd(
            y_train_split, x_train_split, w0, max_iters, gamma
        )
        y_pred = get_linear_pred(x_test_split, w)
        acc, f1 = get_accuracy(y_test_split, y_pred), get_f1(y_test_split, y_pred)
        results.append(("MSE Stochastic GD", loss, acc, f1))
    except Exception as e:
        results.append(("MSE Stochastic GD", f"ERROR: {e}", 0, 0))

    # 3 Least Squares
    try:
        print("Running least_squares...")
        w, loss = least_squares(y_train_split, x_train_split)
        y_pred = get_linear_pred(x_test_split, w)
        acc, f1 = get_accuracy(y_test_split, y_pred), get_f1(y_test_split, y_pred)
        results.append(("Least Squares", loss, acc, f1))
    except Exception as e:
        results.append(("Least Squares", f"ERROR: {e}", 0, 0))

    # 4 Ridge Regression
    try:
        print("Running ridge_regression...")
        lam = 1e-3

        w, loss = ridge_regression(y_train_split, x_train_split, lam)
        y_pred = get_linear_pred(x_test_split, w)
        acc, f1 = get_accuracy(y_test_split, y_pred), get_f1(y_test_split, y_pred)
        results.append((f"Ridge Regression (λ={lam})", loss, acc, f1))
    except Exception as e:
        results.append(("Ridge Regression", f"ERROR: {e}", 0, 0))

    # 5 Logistic Regression
    try:
        max_iters = 40
        k_fold_nbr = 3

        if do_cross_validation:
            print("Performing cross-validation for regularized logistic regression...")
            # Define grids (tune these ranges as needed)
            gammas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
            best_gamma = logistic_grid_search(
                y_train,
                x_train,
                gammas,
                k=5,
                max_iters=max_iters,
                seed=seed,
                verbose=True,
            )
            print(f"Using best γ={best_gamma}")
            gamma = best_gamma
        else:
            print("Running logistic_regression...")
            gamma = 0.3  # found during cross validation

        f1, threshold_log, prediction_log, loss, acc = k_fold_logistic(
            y_train,
            x_train,
            gamma,
            k=k_fold_nbr,
            max_iters=max_iters,
            seed=seed,
            verbose=True,
            final_x_test=x_test,
        )

        results.append((f"Logistic Regression (γ={gamma})", loss, acc, f1))
    except Exception as e:
        results.append(("Logistic Regression", f"ERROR: {e}", 0, 0))

    # 6 Regularized Logistic Regression
    try:
        max_iters = 40
        k_fold_nbr = 3

        if do_cross_validation:
            print("Performing cross-validation for regularized logistic regression...")
            # Define grids (tune these ranges as needed)
            lambdas = np.logspace(-6, 0, 8)
            gammas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]

            best_lambda, best_gamma = reg_logistic_grid_search(
                y_train,
                x_train,
                gammas,
                lambdas,
                k=5,
                max_iters=max_iters,
                seed=seed,
                verbose=True,
            )
            print(f"Using best λ={best_lambda}, γ={best_gamma}")
            lambda_, gamma = best_lambda, best_gamma
        else:
            print("Running reg_logistic_regression...")
            lambda_, gamma = 1e-4, 0.3  # found during cross validation

        f1, threshold_reg_log, prediction_reg_log, loss, acc = k_fold_logistic(
            y_train,
            x_train,
            gamma,
            lambda_,
            k=k_fold_nbr,
            max_iters=max_iters,
            seed=seed,
            verbose=True,
            newton=False,
            final_x_test=x_test,
        )

        results.append(
            (f"Reg. Logistic Regression (λ={lambda_}, γ={gamma})", loss, acc, f1)
        )

    except Exception as e:
        results.append(("Reg. Logistic Regression", f"ERROR: {e}", 0, 0))

    # 7 Regularized Logistic Regression (Newton)
    try:
        print("Running regularized_logistic_regression_newton...")
        max_iters = 300
        k_fold_nbr = 15

        if do_cross_validation:
            print(
                "Performing cross-validation for regularized logistic regression with Newton..."
            )
            # Define grids (tune these ranges as needed)
            lambdas = np.logspace(-6, 0, 8)
            gammas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
            best_lambda, best_gamma = reg_logistic_grid_search(
                y_train,
                x_train,
                gammas,
                lambdas,
                k=5,
                max_iters=max_iters,
                seed=seed,
                verbose=True,
                newton=True,
            )
            print(f"Using best λ={best_lambda}, γ={best_gamma}")
            lambda_, gamma = best_lambda, best_gamma
        else:
            print("Running reg_logistic_regression Newton...")
            lambda_, gamma = 1e-4, 0.3  # found during cross validation

        f1, threshold_reg_log_newton, prediction_reg_log_newton, loss, acc = (
            k_fold_logistic(
                y_train,
                x_train,
                gamma,
                lambda_,
                k=k_fold_nbr,
                max_iters=max_iters,
                seed=seed,
                verbose=True,
                newton=True,
                final_x_test=x_test,
            )
        )
        results.append(
            (
                f"Reg. Logistic Regression (Newton, λ={lambda_}, γ={gamma})",
                loss,
                acc,
                f1,
            )
        )
    except Exception as e:
        results.append(("Reg. Logistic Regression (Newton)", f"ERROR: {e}", 0, 0))

    # -------- Summary output --------
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS (Loss, Accuracy, F1-score)")
    print("=" * 60 + "\n")

    for name, loss, acc, f1 in results:
        print_results_block(name, loss, acc, f1)

    # === FINAL PREDICTIONS & SUBMISSION ===
    if submission:
        try:
            print("\nGenerating final submission...")
            print("Class balance in y_train:", np.mean(y_train))

            print("\n-----logistic regression----------")
            print("mean threashold over the k-folds : ", threshold_log)
            print("Class balance in predictions:", np.mean(prediction_log))
            prediction_log = 2 * prediction_log - 1  # convert to -1/1

            print("\n-----regularized logistic regression----------")
            print("mean threashold over the k-folds : ", threshold_reg_log)
            print("Class balance in predictions:", np.mean(prediction_reg_log))
            prediction_reg_log = 2 * prediction_reg_log - 1  # convert to -1/1

            print("\n-----regularized logistic regression with newton----------")
            print("mean threashold over the k-folds : ", threshold_reg_log_newton)
            print("Class balance in predictions:", np.mean(prediction_reg_log_newton))
            prediction_reg_log_newton = (
                2 * prediction_reg_log_newton - 1
            )  # convert to -1/1

            # Create submission file using reg log

            create_csv_submission(test_ids, prediction_reg_log_newton, "submission.csv")
            print(
                "Created 'submission.csv' using regularized logistic regression successfully."
            )
        except Exception as e:
            print(f"Submission error: {e}")


if __name__ == "__main__":
    main()
