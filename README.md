Week 1: Boston House Price Prediction

Dates: 02/08/2026 – 09/08/2026
Domain / Track: Machine Learning / Data Science
Status: Completed

Weekly Objectives:
[x] Perform Exploratory Data Analysis (EDA) and resolve missing values across all features in HousingData.csv.

[x] Build and standardize the data preprocessing workflow while preventing data leakage.

[x] Train, benchmark, and evaluate multiple linear, regularized, and ensemble regression models.

[x] Complete model diagnostics through residual analysis and feature importance visualization.

Tasks Completed & Deliverables
1. Exploratory Data Analysis & Data Cleaning
Description: Inspected the dataset distribution, audited missing values across numerical/categorical features, and analyzed inter-feature correlations with MEDV

Key Actions:Identified 20 missing entries across CRIM, ZN, INDUS, CHAS, AGE, and LSTAT.Imputed skewed continuous variables using the feature median and binary categorical feature CHAS using mode imputation via direct column assignment.Plotted a correlation heatmap to identify primary predictors (e.g., RM with positive correlation $\approx +0.70$ and LSTAT with negative correlation $\approx -0.74$).

Artifacts / Links:
Notebook: notebooks/01_eda_and_cleaning.ipynb
Output/Visualization: assets/plots/correlation_heatmap.png

2. Model Training, Benchmarking & Evaluation
Description: Structured an 80/20 train-test split, standardized continuous features using StandardScaler, and trained 5 baseline and ensemble regressors.

Key Actions:Fitted the scaler strictly on X_train and transformed X_test to prevent data leakage.Benchmarked Ordinary Least Squares (OLS), Ridge, Lasso, Random Forest, and Gradient Boosting Regressors.Evaluated all models on unseen test data using $R^2$, RMSE, and MAE metrics.Generated residual distribution plots and feature importance rankings.

Artifacts / Links:
Code: src/train_and_evaluate.py
Output/Visualization: assets/plots/feature_importance_rf.png, assets/plots/residual_plot.png

Key Learnings & Takeaways:
Handling Data Leakage in Preprocessing: Learned why scaling parameters ($\mu, \sigma$) must be computed strictly on the training partition before transforming the test partition.

Linear vs. Tree-Based Non-Linearity: Observed firsthand why ensemble methods (Random Forest, Gradient Boosting) outperform linear models on housing data by effectively modeling non-linear interactions and thresholds in features like LSTAT and RM.

Pandas 3.0 Chained Assignment: Understood the mechanics behind FutureWarning deprecations regarding inplace=True on chained series slicing and transitioned to explicit column reassignment.

Tools & Libraries Explored:
scikit-learn: Utilized StandardScaler, train_test_split, regression estimators, and performance metrics (r2_score, mean_squared_error, mean_absolute_error).
pandas & seaborn: Built statistical summaries, correlation heatmaps, and clean data frames for metric comparisons.
matplotlib.pyplot: Visualized residual scatter plots and horizontal bar charts for feature importance.

References & Resources:
Scikit-Learn Regression Metrics Documentation – Reference for calculating and interpreting MAE, RMSE, and $R^2$.
Pandas Missing Data Guide – Best practices for series imputation without chained indexing issues.
Scikit-Learn Ensemble Methods Guide – Theoretical understanding of tree bagging and gradient boosting mechanisms.
