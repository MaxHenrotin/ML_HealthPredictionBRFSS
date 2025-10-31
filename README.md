# ML Project 1: Health Outcome Prediction
## Predicting Health Outcomes with Logistic Regression 
The goal is to predict a binary health outcome from the 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset.  
We focus on data preprocessing, feature engineering, and model selection using logistic regression.
More information about the data at https://www.cdc.gov/brfss/annual_data/annual_2015.html.
To have a complete overview of the project and our results please read ML_project_report.pdf.

## Folder Structure
```
ML_HealthPredictionBRFSS/
├── dataset/
│   ├── sample_submission.csv
│   ├── x_test.csv
│   ├── x_train.csv
│   └── y_train.csv
├── helpers/
│   ├── csv_helpers.py
│   ├── data_processing.py
│   ├── evaluations.py
│   ├── hyper_parameters.py
│   └── math.py
├── feature_info.json
├── implementations.py
├── main.py
├── README.md
├── requirements.txt
└── run.py
```

## How to Run

### 1. Install dependencies (if not installed already)

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Make sure the following CSV files are placed in the `dataset/` directory, that need to be created by extracting the dataset.zip file:

- `x_train.csv`
- `y_train.csv`
- `x_test.csv`
- `sample_submission.csv`


(Make sure that the DATA_PATH variable in run.py and main.py is set to the path to this folder)

### 3. Run training and submission
To train the model and generate the final submission file:

```bash
python run.py
```

### 4. (Optional) Generate performance results
You can evaluate the effects of feature engineering by running:

```bash
python main.py
```

This will generate and display the comparison of F1-score for different strategies.

## Data Processing Steps

Our preprocessing pipeline includes the following modular steps:

1. **Feature Selection**  
   Manual selection of relevant features using BRFSS documentation.

2. **Replacement of Special NaN/Zero Values**  
   Mapping known placeholders (e.g., `999.99`) to actual `np.nan` or 0.

3. **Dropping Features with Too Many Missing Values**  
   Columns with more than 60% missing values are discarded.

4. **Missing Value Imputation**  
   - Mean for continuous features  
   - Most frequent value for categorical features (≤10 unique values)

5. **Outlier Clipping** *(optional)*  
   Clip extreme values in continuous features using the IQR method.

6. **Polynomial Feature Expansion** *(optional)*  
   Adds powers of continuous features up to a specified degree.

7. **Pairwise Interaction Terms** *(optional)*  
   Adds cross-products of continuous features to capture interactions.

8. **Normalization**  
   Applies z-score normalization **only** to continuous features.

9. **Bias Term Addition**  
   Adds a column of ones to include an intercept in linear models.

10. **Target Transformation**  
    Converts target values from `{−1, 1}` to `{0, 1}` to suit classification models.

## Models Implemented

We implemented the following machine learning models:

- **Least Squares Regression**  
  Solves via the normal equation.

- **Ridge Regression**  
  Regularized version of least squares using L2 penalty.

- **Gradient Descent for Least Squares**  
  Iterative minimization of MSE loss.

- **Stochastic Gradient Descent (SGD)**  
  Faster, approximate version of gradient descent using random batches.

- **Logistic Regression**  
  Binary classification using sigmoid and gradient descent.

- **Regularized Logistic Regression**  
  Adds L2 penalty to logistic loss; supports:
  - Gradient Descent  
  - **Newton's Method** (second-order optimization for faster convergence)


## Model Evaluation

To evaluate our models, we rely on **stratified k-fold cross-validation** to ensure consistent and fair performance metrics across different splits of the data. Stratification preserves the original class proportions in each fold, which is crucial in our case due to class imbalance.

### Metrics Used

- **F1-score** *(main metric)*  
  The F1-score is the harmonic mean of precision and recall. It is especially important in our context because the dataset is imbalanced. F1 gives a better picture of model performance than accuracy.

- **Accuracy**  
  We also report accuracy for completeness, although it is less informative in imbalanced settings. A model could have high accuracy simply by predicting the majority class.

### Threshold Optimization

For each fold, we find the **optimal classification threshold** that maximizes the F1-score on the validation set. This often performs better than the standard 0.5 threshold.

### Aggregated Results

We report the **mean** across all folds for:
- F1-score
- Accuracy
- Prediction vector

This cross-validation setup provides a robust and fair assessment of our models, and helps mitigate overfitting to a single train-test split.

### Best Score on AIcrowd

Our best submission to the [AIcrowd leaderboard](https://www.aicrowd.com/challenges/epfl-machine-learning-2025) achieved the following result:

- **F1-score:** `0.429` 
- **Model:** Regularized Logistic Regression (γ=0.05, λ=1e-6)
- **Preprocessing:** Normalization, Polynomial Expansion (degree=3), Outlier Clipping, Bias Term added

This score reflects the final performance of our best pipeline, evaluated against the hidden test labels on the competition server.

