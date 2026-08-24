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

Key Actions: 
Identified 20 missing entries across CRIM, ZN, INDUS, CHAS, AGE, and LSTAT.
Imputed skewed continuous variables using the feature median and binary categorical feature CHAS using mode imputation via direct column assignment.
Plotted a correlation heatmap to identify primary predictors (e.g., RM with positive correlation $\approx +0.70$ and LSTAT with negative correlation $\approx -0.74$).

Artifacts / Links:
Notebook: notebooks/01_eda_and_cleaning.ipynb
Output/Visualization: assets/plots/correlation_heatmap.png

2. Model Training, Benchmarking & Evaluation
Description: Structured an 80/20 train-test split, standardized continuous features using Standard Scaler, and trained 5 baseline and ensemble regressors.

Key Actions: 
Fitted the scaler strictly on X_train and transformed X_test to prevent data leakage.
Benchmarked Ordinary Least Squares (OLS), Ridge, Lasso, Random Forest, and Gradient Boosting Regressors.
Evaluated all models on unseen test data using $R^2$, RMSE, and MAE metrics.
Generated residual distribution plots and feature importance rankings.

Artifacts / Links:
Code: src/train_and_evaluate.py
Output/Visualization: assets/plots/feature_importance_rf.png, assets/plots/residual_plot.png

Key Learnings & Takeaways:
Handling Data Leakage in Preprocessing: Learned why scaling parameters ($\mu, \sigma$) must be computed strictly on the training partition before transforming the test partition.

Linear vs. Tree-Based Non-Linearity: Observed firsthand why ensemble methods (Random Forest, Gradient Boosting) outperform linear models on housing data by effectively modeling non-linear interactions and thresholds in features like LSTAT and RM.

Pandas 3.0 Chained Assignment: Understood the mechanics behind Future Warning deprecations regarding inplace=True on chained series slicing and transitioned to explicit column reassignment.

Tools & Libraries Explored:
scikit-learn: Utilized Standard Scaler, train_test_split, regression estimators, and performance metrics (r2_score, mean_squared_error, mean_absolute_error).
pandas & seaborn: Built statistical summaries, correlation heatmaps, and clean data frames for metric comparisons.
matplotlib.pyplot: Visualized residual scatter plots and horizontal bar charts for feature importance.

References & Resources:
Scikit-Learn Regression Metrics Documentation – Reference for calculating and interpreting MAE, RMSE, and $R^2$.
Pandas Missing Data Guide – Best practices for series imputation without chained indexing issues.
Scikit-Learn Ensemble Methods Guide – Theoretical understanding of tree bagging and gradient boosting mechanisms.

Week 2: Loan Approval Prediction with Machine Learning

Dates: 10/08/2026 – 17/08/2026
Domain / Track: Machine Learning / Fintech & Credit Risk Modeling
Status: Completed

Weekly Objectives
[x] Clean and preprocess fintech credit history data (loan_prediction.csv) by handling missing values across numerical and categorical features.

[x] Engineer domain-relevant credit attributes including household earning capacity (TotalIncome).

[x] Implement a stratified validation workflow to maintain class balance across training and test partitions.

[x] Train, benchmark, and evaluate multiple linear and non-linear classifiers using ranking and discrimination metrics (Accuracy, F1-Score, ROC-AUC).

Tasks Completed & Deliverables
1. Data Cleaning & Feature Engineering
Description: Performed exploratory data analysis, imputed missing entries across 7 independent variables, and engineered total household income features.

Key Actions:
Imputed skewed continuous variables (LoanAmount) with the median and categorical/discrete attributes (Credit_History, Self_Employed, Dependents, Gender, Loan_Amount_Term, Married) with the mode.
Sanitized string formatted values in the Dependents field ('3+' transformed to integer 3).
Created an aggregated TotalIncome feature by combining ApplicantIncome and CoapplicantIncome to better reflect total repayment ability.
One-hot encoded categorical variables using pd.get_dummies (drop_first=True) and mapped target labels (Y $\rightarrow$ 1, N $\rightarrow$ 0)

Artifacts / Links:
Notebook: notebooks/02_loan_eda_and_preprocessing.ipynb
Output/Visualization: assets/plots/loan_distribution_plots.png

2. Classifier Benchmarking & Credit Risk Evaluation
Description: Established a stratified 80/20 train-test split, standardized numerical distributions, and trained 4 classification algorithms.

Key Actions:
Standardized feature distributions with StandardScaler to optimize gradient descent and coefficient penalization.
Benchmarked Logistic Regression against Decision Tree, Random Forest, and Gradient Boosting Classifiers.
Evaluated performance focusing on credit risk priorities: Precision (minimizing non-performing loan approvals), Recall, and ROC-AUC score.
Built feature importance rankings highlighting Credit_History as the primary approval indicator.

Artifacts / Links:
Code: src/loan_classifier_pipeline.py
Output/Visualization: assets/plots/loan_feature_importance.png, assets/plots/confusion_matrix_lr.png

Key Learnings & Takeaways
Technical Concepts:
Credit Risk Trade-offs (Precision vs. Recall): In credit underwriting, a false positive (approving a loan that defaults) incurs significantly higher financial loss than a false negative (rejecting a creditworthy applicant). Evaluating models on Precision and ROC-AUC is critical beyond general accuracy.
Stratification in Imbalanced Datasets: Learned how stratify=y ensures that the ~68.7% approval distribution is proportionally preserved across both training and test sets to prevent evaluation bias.
Dominance of Credit History: Observed how non-linear tree splits prioritize Credit_History as the primary branch, reflecting standard underwriting heuristics.

Tools & Libraries Explored:
scikit-learn: Applied StratifiedKFold, StandardScaler, and classification estimators (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier).
sklearn.metrics: Evaluated roc_auc_score, precision_score, recall_score, and visualized confusion matrices using seaborn.

References & Resources:
Scikit-Learn Classification Metrics Documentation – Technical reference for precision-recall tradeoffs and ROC-AUC curve calculations.
Pandas Categorical Encoding Documentation – Reference for dummy variable creation and column dropping strategies.
Credit Risk Modeling Best Practices – Guidance on regularized logistic regression for financial decision-making systems.

Week 3: Sentiment Analysis using DistilBERT

**Dates:** 18/08/2026 – 25/08/2026
**Domain / Track:** Natural Language Processing / Machine Learning
**Status:** Completed

## Weekly Objectives

[x] Select and implement a pretrained Language Model suitable for a Natural Language Processing task.

[x] Load and configure DistilBERT for binary sentiment classification using a Jupyter Notebook workflow.

[x] Demonstrate tokenization, text preprocessing, sentiment prediction, and confidence-score generation.

[x] Evaluate model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

[x] Analyze model behavior, inference performance, and limitations on different types of natural-language inputs.

# Tasks Completed & Deliverables

## 1. Language Model Selection & Environment Setup

**Description:**
Selected **DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)** as the pretrained Language Model for implementing a sentiment analysis application.

**Key Actions:**

* Selected DistilBERT because it provides a lightweight and computationally efficient alternative to BERT while retaining strong contextual language understanding.
* Configured the Hugging Face Transformers environment in Jupyter Notebook.
* Installed and imported the required Python libraries including `transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`.
* Loaded the pretrained tokenizer and sequence-classification model.
* Inspected model configuration including vocabulary size, hidden size, number of hidden layers, attention heads, maximum sequence length, and output classes.

**Artifacts / Links:**

* Notebook: `notebooks/03_distilbert_sentiment_analysis.ipynb`
* Model: `distilbert-base-uncased-finetuned-sst-2-english`
* Output/Visualization: Model configuration and prediction outputs generated in the notebook.

## 2. Tokenization & Sentiment Prediction

**Description:**
Implemented the complete NLP inference workflow by converting natural-language sentences into tokens and using the pretrained DistilBERT model to classify text as positive or negative sentiment.

**Key Actions:**

* Used the DistilBERT tokenizer to convert raw text into token representations.
* Examined generated tokens and corresponding token IDs to understand the preprocessing stage.
* Passed tokenized inputs through the pretrained transformer model.
* Applied sequence classification to identify sentiment polarity.
* Generated prediction confidence scores for individual text inputs.
* Implemented an interactive input section that allows users to enter a sentence and receive a sentiment prediction.

**Example Application:**

* Positive input: `"I really enjoyed this movie. It was fantastic!"`
* Negative input: `"The service was terrible and disappointing."`

The model returns the predicted sentiment label along with a confidence score.

**Artifacts / Links:**

* Code: `src/distilbert_sentiment_analysis.py`
* Notebook: `notebooks/03_distilbert_sentiment_analysis.ipynb`
* Output/Visualization: Sentiment prediction and confidence-score outputs.

## 3. Model Evaluation & Performance Analysis

**Description:**
Created a manually labelled sentiment dataset and evaluated the pretrained DistilBERT model on unseen test examples using standard classification metrics.

**Key Actions:**

* Created a binary sentiment dataset containing positive and negative text samples.
* Generated predictions for all test samples using DistilBERT.
* Compared actual sentiment labels with model-predicted labels.
* Calculated **Accuracy, Precision, Recall, and F1-Score**.
* Generated a classification report to analyze class-wise performance.
* Constructed a confusion matrix to identify correct classifications and misclassifications.
* Analyzed model confidence scores across individual test samples.

**Evaluation Metrics:**

* Accuracy – Measures the overall percentage of correctly classified sentiment samples.
* Precision – Measures the proportion of predicted positive samples that were actually positive.
* Recall – Measures the proportion of actual positive samples correctly identified.
* F1-Score – Provides a balanced measure of Precision and Recall.

**Artifacts / Links:**

* Output/Visualization: `assets/plots/distilbert_confusion_matrix.png`
* Output/Visualization: `assets/plots/model_confidence_scores.png`
* Evaluation results: Generated within the Jupyter Notebook.

# Key Learnings & Takeaways

## Technical Concepts

**Pretrained Language Models:**
Learned how pretrained transformer-based Language Models can be directly applied to NLP tasks without training a model completely from scratch.

**Transformer-Based NLP:**
Understood the role of transformer architectures in capturing contextual relationships between words and generating meaningful representations for classification tasks.

**Tokenization:**
Learned how raw natural-language text is converted into tokens and numerical token IDs before being processed by the transformer model.

**Transfer Learning:**
Understood how a model pretrained on a large text corpus can be fine-tuned for a specific NLP task such as sentiment classification.

**Confidence-Based Prediction:**
Analyzed the model's confidence scores to understand how strongly the model supports a particular sentiment prediction.

**Model Evaluation:**
Learned how Accuracy, Precision, Recall, F1-Score, and Confusion Matrix provide different perspectives on classification performance.

## Model Behavior & Limitations

* Observed that DistilBERT performs effectively on straightforward positive and negative statements.
* Identified that complex language can be more challenging for sentiment classification.
* Tested sentences involving negation, ambiguity, and context-dependent meaning.
* Observed that sentiment models may have difficulty correctly interpreting sarcasm and subtle contextual expressions.
* Compared prediction confidence across different input sentences.
* Measured inference time to understand the practical efficiency of the pretrained model.

# Tools & Libraries Explored

**Hugging Face Transformers:**
Used `AutoTokenizer`, `AutoModelForSequenceClassification`, and the sentiment-analysis pipeline to load and utilize the pretrained DistilBERT model.

**PyTorch:**
Used for model inference and tensor-based processing.

**scikit-learn:**
Applied `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, and `confusion_matrix` for model evaluation.

**Pandas:**
Created and manipulated the sentiment evaluation dataset and compared actual versus predicted labels.

**Matplotlib & Seaborn:**
Created confidence-score visualizations and confusion-matrix plots for performance analysis.

**Jupyter Notebook:**
Used as the primary development environment for implementing and demonstrating the complete NLP workflow.

# References & Resources

**Hugging Face Transformers Documentation** – Reference for pretrained transformer models, tokenizers, and sequence classification.

**DistilBERT Model Documentation** – Reference for the selected pretrained `distilbert-base-uncased-finetuned-sst-2-english` model.

**Scikit-Learn Classification Metrics Documentation** – Reference for Accuracy, Precision, Recall, F1-Score, classification reports, and confusion matrices.

**PyTorch Documentation** – Reference for tensor operations and model inference.
