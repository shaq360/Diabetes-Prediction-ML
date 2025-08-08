🩺 Diabetes Prediction using Machine Learning
This project focuses on the prediction of diabetes using supervised machine learning models. The goal is to evaluate model performance under different data preprocessing strategies and experimental setups, using multiple datasets. The project involves rigorous cross-validation, feature selection, and ablation studies to analyze model behavior in real-world conditions.

⚠️ This work is currently being prepared for publication in a research paper.

✅ Objectives
Predict the likelihood of diabetes using ML models

Evaluate and compare model performance across multiple datasets

Perform ablation studies to measure the impact of preprocessing steps

Analyze robustness using dataset-based cross-validation

📊 Datasets Used
A total of 4 datasets were used in this project

🧠 Models Evaluated
Logistic Regression

Random Forest

Support Vector Machine (SVM)

⚙️ Methodology
1. Preprocessing
Handling Missing Values: Imputation with mean

Outlier Treatment: Replaced using IQR method

Data Imbalance Handling: Using SMOTETomek

Standardization: Using StandardScaler

Feature Selection: Using f_classif from sklearn.feature_selection

2. Training & Validation
Train-Test Split: Initial split of the dataset into 85/15

10-Fold Cross-Validation: To ensure stable performance

Dataset-Based Cross-Validation:

Trained on two datasets and tested on a third


🔬 Ablation Study
To evaluate the impact of various preprocessing steps, the following ablation experiments were performed:

Ablation Setup	Description
Ablation 1	Replaced missing values with mean
Ablation 2	Without replacing outliers
Ablation 3	Without handling data imbalance
Ablation 4	Used simple 80/20 train-test split
Ablation 5	Without feature selection
Ablation 6	Evaluated models on a new, larger dataset

📈 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

Confusion Matrix

📄 Publication
This project is part of an ongoing research paper which is currently under review. The work demonstrates the impact of preprocessing and data validation strategies on diabetes prediction models using real-world healthcare datasets.
