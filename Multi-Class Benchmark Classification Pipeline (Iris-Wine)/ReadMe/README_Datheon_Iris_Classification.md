# Multi-Class Benchmark Classification Pipeline — Iris Dataset

A supervised machine learning project completed as part of my **Datheon Machine Learning internship training**. The project builds and benchmarks two Scikit-Learn classifiers to predict Iris flower species from sepal and petal measurements.

## Project objective

Predict one of three Iris species — `setosa`, `versicolor`, or `virginica` — using four numerical features:

- Sepal length
- Sepal width
- Petal length
- Petal width

This is a **supervised multi-class classification** task because the dataset includes labelled examples for the model to learn from.

## Tools and libraries

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook / Databricks
- Joblib (for saving trained pipelines)

## Dataset

The project uses the Iris dataset supplied for the case study. It contains 150 records and five columns (four numeric measurements and one species label).

During data cleaning:
- No missing values were found.
- One exact duplicate row was identified and removed.
- The cleaned dataset contained 149 rows: 50 `setosa`, 50 `versicolor`, and 49 `virginica`.

## Methodology

The notebook follows the 12-step Machine Learning Process taught in BrightLearn, organised into four phases:

### Phase 1 — Frame
1. Define the problem as predicting Iris species from flower measurements.
2. Load the supplied dataset into the analysis environment.

### Phase 2 — Prepare
3. Inspect data types, missing values and duplicate records; remove the exact duplicate.
4. Perform exploratory data analysis using summary statistics, species counts, histograms, a Pandas scatter matrix and a correlation matrix.
5. Separate the four input features (`X`) from the target (`y`) and standardise features using `StandardScaler`.
6. Split the data into 80% training and 20% testing sets, using stratification to preserve class proportions and `random_state=42` for reproducibility.

### Phase 3 — Model
7. Select Logistic Regression and Decision Tree classifiers.
8. Train both models using Scikit-Learn pipelines.
9. Evaluate the models using accuracy, weighted precision, weighted recall, weighted F1-score, classification reports and confusion matrices.
10. Compare model performance and consider further tuning.

### Phase 4 — Operate
11. Save the trained pipelines as Joblib files to demonstrate a deployment-ready artefact.
12. Outline monitoring considerations, including feature drift, prediction patterns and performance once true labels become available.

## Modelling approach

Two pipelines were trained:

- **Logistic Regression:** `StandardScaler` followed by `LogisticRegression(max_iter=1000, random_state=42)`.
- **Decision Tree:** `StandardScaler` followed by `DecisionTreeClassifier(max_depth=4, random_state=42)`.

Scaling is especially useful for Logistic Regression. It is not generally required by Decision Trees, but it was included in both pipelines for consistency. Because the scaler is inside each pipeline, it is fitted on the training data only, helping prevent data leakage.

The notebook also includes two-dimensional decision-boundary illustrations using only petal length and petal width. These are supplementary visualisations; the main benchmark models use all four features.

## Results

Both models achieved identical performance on the held-out test set of 30 observations:

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 0.9333 (93.33%) | 0.9333 (93.33%) |
| Weighted precision | 0.9333 (93.33%) | 0.9333 (93.33%) |
| Weighted recall | 0.9333 (93.33%) | 0.9333 (93.33%) |
| Weighted F1-score | 0.9333 (93.33%) | 0.9333 (93.33%) |

The classification reports show the same per-class results for both models:

| Species | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor | 0.90 | 0.90 | 0.90 | 10 |
| Virginica | 0.90 | 0.90 | 0.90 | 10 |

Both models correctly classified 28 of the 30 test observations. They correctly identified all 10 Setosa examples, and each made one error between Versicolor and Virginica. The confusion matrices in the notebook show these results.

## Conclusion

The results indicate that both Logistic Regression and Decision Tree performed equally well on this particular train/test split, achieving 93.33% accuracy and weighted F1-score. Neither model can be declared the clear winner based on these metrics alone. Logistic Regression is a reasonable choice as a simpler baseline, while the Decision Tree offers a rule-based model that can be easier to explain. Further validation using cross-validation and hyperparameter tuning would be useful before selecting a final model for a real application.

## How to run

1. Clone or download this repository.
2. Open the notebook in Jupyter Notebook, JupyterLab or Databricks.
3. Upload the supplied Iris CSV dataset to your environment.
4. Update the dataset-loading cell to point to your file or table. The submitted Databricks version reads from a workspace table; if running locally, replace that cell with `pd.read_csv("Task_1.csv")`.
5. Run the notebook from top to bottom.

## Repository contents

- `Datheon_Iris_Classification_Guided_Notebook.ipynb` — guided notebook with code, explanations, visualisations and evaluation.
- `Task_1.csv` — include the supplied dataset in the repository if its distribution is permitted by the course/provider.

## Scope and limitations

- Results are based on one stratified 80/20 split with `random_state=42`; performance may vary with a different split.
- The dataset is small, so the results should be treated as an educational benchmark rather than proof of real-world generalisation.
- The notebook saves the trained model pipelines but does not deploy a live prediction service or implement automated monitoring.
- The 2D decision-boundary plots use two features for visualisation; the main benchmark uses all four.

## Acknowledgements

Completed as part of the Datheon Machine Learning training programme, applying the 12-step workflow from BrightLearn's *The Machine Learning Process* learning notes.

Training programme: https://www.datheon.in/training

---

*Learning by building, evaluating and explaining — one machine learning project at a time.*
