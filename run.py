import numpy as np
from implementations import *
from helpers.data_processing import *
from helpers.math import *
from helpers.evaluations import *
from helpers.csv_helpers import *
from helpers.hyper_parameters import *

# We achieved a f1-score of 0.429 (accuracy=0.867) on AIcrowd with this submission

# ------------- Configuration Parameters ------------------
DATA_PATH = "dataset/"  # Path to the dataset folder
submission = True  # If True, generate the submission CSV file
do_cross_validation = False  # If True, perform k-fold cross-validation
sub_sample = False  # If True, use a smaller data subset for faster testing
seed = 5  # Random seed for reproducibility


# ------------- Data Processing Parameters ------------------
normalize = True  # Standardize continuous features (mean=0, std=1)
remove_outliers = True  # Clip extreme values to reduce noise
poly_degree = 3  # Add polynomial features up to degree 3
add_interactions = True  # Do not add pairwise interaction terms
add_bias = True  # Add a bias column (constant 1) to features


def run():
    # -------- Load raw data --------
    print("\nLoading data...")
    x_train_raw, x_test_raw, y_train_raw, _, test_ids = load_csv_data(
        DATA_PATH, sub_sample=sub_sample
    )

    # -------- Process data --------
    print("\nProcessing data...")
    x_train, y_train, x_test = data_processing(
        x_train_raw,
        y_train=y_train_raw,
        x_test=x_test_raw,
        normalize=normalize,
        remove_outliers=remove_outliers,
        poly_degree=poly_degree,
        add_interactions=add_interactions,
        add_bias=add_bias,
        verbose=False,
    )

    # -------- Run Reg. Logistic Regression (Newton) --------
    results = []

    try:
        print("\nRunning Newton regularized logistic regression...")
        max_iters = 300
        k_fold_nbr = 20

        if do_cross_validation:
            print(
                "\nPerforming cross-validation for regularized logistic regression with Newton..."
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
                verbose=False,
                newton=True,
            )
            lambda_, gamma = best_lambda, best_gamma
        else:
            # Set best hyperparameters found in previously performed cross validation
            lambda_, gamma = 1e-4, 0.3

        print(f"Using best λ={lambda_:.4f}, γ={gamma}.")

        f1, threshold_reg_log_newton, prediction_reg_log_newton, loss, acc = (
            k_fold_logistic(
                y_train,
                x_train,
                gamma,
                lambda_,
                k=k_fold_nbr,
                max_iters=max_iters,
                seed=seed,
                verbose=False,
                newton=True,
                final_x_test=x_test,
            )
        )
        print(
            f"Using mean threashold over the k-folds: {threshold_reg_log_newton:.4f}."
        )

        results.append(
            (
                f"Reg. Logistic Regression (Newton, λ={lambda_:.4f}, γ={gamma})",
                loss,
                acc,
                f1,
            )
        )

    except Exception as e:
        results.append(("Reg. Logistic Regression (Newton)", f"ERROR: {e}", 0, 0))

    # -------- Summary output --------
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60 + "\n")

    for name, loss, acc, f1 in results:
        print_results_block(name, loss, acc, f1)

    # === FINAL PREDICTIONS & SUBMISSION ===
    if submission:
        try:
            print("\nGenerating final submission...")
            print("Class balance in predictions:", np.mean(prediction_reg_log_newton))
            # Convert predictions from {0,1} to {-1,1}
            prediction_reg_log_newton = 2 * prediction_reg_log_newton - 1

            # Create submission file
            create_csv_submission(test_ids, prediction_reg_log_newton, "submission.csv")
            print("Created 'submission.csv' successfully.")

        except Exception as e:
            print(f"Submission error: {e}")


if __name__ == "__main__":
    run()
