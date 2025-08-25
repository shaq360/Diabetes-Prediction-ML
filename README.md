Diabetes Risk Prediction with ML: Cross-Validation and Ablation
===============================================================

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/shaq360/Diabetes-Prediction-ML/releases)

[![cross-validation](https://img.shields.io/badge/cross--validation-green)](https://github.com/shaq360/Diabetes-Prediction-ML)
[![logistic-regression](https://img.shields.io/badge/logistic--regression-orange)](https://github.com/shaq360/Diabetes-Prediction-ML)
[![random-forest](https://img.shields.io/badge/random--forest-red)](https://github.com/shaq360/Diabetes-Prediction-ML)
[![svm](https://img.shields.io/badge/svm-purple)](https://github.com/shaq360/Diabetes-Prediction-ML)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-brightgreen)](https://github.com/shaq360/Diabetes-Prediction-ML)

Banner image
------------
![ML health banner](https://images.unsplash.com/photo-1581091870621-3c6bb9a3e0a7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80)

About
-----
This repository implements supervised models to predict diabetes risk. It emphasizes solid preprocessing, dataset-aware cross-validation, and ablation studies. The code compares Logistic Regression, Support Vector Machine (SVM), and Random Forest. The repo includes experiment scripts, plotting helpers, and reporting tools for model comparison.

Why this repo
--------------
- Reproduce experiments that compare model families on the same pipeline.  
- Run dataset-based cross-validation that respects patient grouping and class balance.  
- Run ablation studies to measure the impact of preprocessing and feature choices.  
- Generate ROC, PR, calibration, and feature-importance plots.

Topics
------
cross-validation, logistic-regression, machine-learning, matplotlib, numpy, pandas, random-forest, scikit-learn, seaborn, support-vector-machine

Table of contents
-----------------
- Features
- Folder layout
- Quick start
- Releases (must download and execute)
- Reproducible experiments
- Data preprocessing
- Model training and evaluation
- Ablation studies
- Visualizations
- Configuration
- Contributing
- License

Features
--------
- End-to-end pipeline: raw CSV -> cleaned data -> models -> metrics -> plots.  
- Multiple models: Logistic Regression, SVM, Random Forest.  
- Proper cross-validation: stratified, grouped, or time-based folds.  
- Ablation scripts to drop features or change preprocessing steps.  
- Exportable reports: CSV metrics, PNG plots, serialized models (joblib).  
- Clear utilities for ROC AUC, precision/recall, calibration, and confusion matrices.

Folder layout
-------------
- data/ - raw and processed datasets (not in repo).  
- notebooks/ - EDA and example notebooks.  
- src/  
  - preprocess.py - cleaning and feature engineering.  
  - model.py - model definitions and wrappers.  
  - cv.py - cross-validation utilities.  
  - ablation.py - ablation automation.  
  - eval.py - metrics and plotting functions.  
- experiments/ - experiment configs and results.  
- releases/ - packaged release assets (see Releases section).  
- README.md - this file.

Quick start
-----------
1. Clone the repo
   ```bash
   git clone https://github.com/shaq360/Diabetes-Prediction-ML.git
   cd Diabetes-Prediction-ML
   ```
2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Place your dataset (for example, Pima Indians Diabetes CSV) in data/raw/ as data.csv. The repo expects a target column named "Outcome" or "diabetes".

4. Run a quick experiment
   ```bash
   python src/model.py --config experiments/default.yaml
   ```

Releases (download and execute)
-------------------------------
Download and execute the release asset file from the Releases page. Download and run the packaged executable or scripts available at:
https://github.com/shaq360/Diabetes-Prediction-ML/releases

Example:
- Download the release asset named diabetes_prediction_release.zip from the Releases page.
- Unzip:
  ```bash
  unzip diabetes_prediction_release.zip -d release_run
  cd release_run
  ```
- Run:
  ```bash
  python run_release.py --data ../data/raw/data.csv
  ```
The release contains a self-contained runner that executes a standard experiment, generates plots in outputs/, and saves metrics to outputs/metrics.csv.

Reproducible experiments
------------------------
Each experiment uses a YAML config in experiments/*.yaml. The config covers:
- model: type (logreg, svm, rf)
- preprocessing: impute_strategy, scaling, feature_set
- cross_validation: type (stratified, group, time), n_splits
- metrics: list of metrics to compute
- output: output directory

Example config snippet:
```yaml
model:
  type: rf
  params:
    n_estimators: 200
    max_depth: 10

preprocessing:
  impute_strategy: median
  scale: True
  features: baseline

cross_validation:
  type: stratified
  n_splits: 5
```
Run with:
```bash
python src/model.py --config experiments/rf_baseline.yaml
```

Data preprocessing
------------------
The pipeline includes:
- Type casting and column normalization.  
- Missing value handling: median imputation for numeric features, constant for categorical.  
- Outlier handling: winsorize or clip based on percentiles.  
- Scaling: StandardScaler or RobustScaler.  
- Categorical encoding: one-hot or ordinal as configured.  
- Feature engineering: BMI categories, age bins, and ratio features.

The preprocess module exposes a Pipeline object. Use it to reproduce preprocessing across CV folds.

Cross-validation details
------------------------
Use dataset-aware folds. Options:
- Stratified K-Fold for class balance.  
- Group K-Fold when samples share patient IDs.  
- TimeSeriesSplit for temporal validation.

The repo uses stratified repeated CV by default. Each fold uses the same preprocessing fit on the training split only. Use cv.py to produce fold indices and to save them for later reproducibility.

Model training and evaluation
-----------------------------
Supported classifiers:
- Logistic Regression (L2, liblinear or saga solver)  
- Support Vector Machine (RBF, linear kernels)  
- Random Forest (feature importance via Gini)

Evaluation metrics:
- Accuracy  
- Precision, Recall, F1  
- ROC AUC  
- PR AUC  
- Calibration (Brier score, calibration curve)

Each run saves:
- metrics.csv with fold-level metrics  
- aggregated_metrics.json with mean and std  
- model_fold_X.joblib for each fold  
- plots/ with ROC, PR, and calibration curves

Use eval.py to calculate thresholds and to produce tables for each model.

Ablation studies
----------------
Ablation scripts let you:
- Remove groups of features (demographics, labs, vitals).  
- Switch preprocessing options (no scaling, different imputation).  
- Change hyperparameters while keeping the pipeline fixed.

The ablation module runs a grid of experiments and writes a matrix of metric deltas. Use the ablation report to identify which components improve generalization.

Example ablation run:
```bash
python src/ablation.py --config experiments/ablation.yaml --out outputs/ablation
```

Visualizations
--------------
The repo provides standard plots:
- ROC curves per fold with mean and bands.  
- Precision-Recall curves.  
- Confusion matrices per fold.  
- Calibration plots and reliability diagrams.  
- Feature importance bar plots (Random Forest) and coefficient plots (Logistic Regression).

Plotting uses seaborn and matplotlib. Save formats: PNG and PDF.

Example: generate ROC plots after a run
```bash
python src/eval.py --results outputs/metrics.csv --plot-dir outputs/plots
```

Configuration and hyperparameter tuning
---------------------------------------
You can tune hyperparameters in two ways:
- Use the config file to set a grid and call the built-in grid search (CV-aware).  
- Use scikit-learn wrappers for randomized search with custom scoring functions.

The grid search respects the same CV splitter used for final evaluation to avoid leakage. Use the same random_state across runs to improve reproducibility.

CLI reference
-------------
- python src/model.py --config <path> --seed <int>  
- python src/preprocess.py --input data/raw/data.csv --output data/processed/data.pkl  
- python src/cv.py --type stratified --n_splits 5 --save folds.pkl  
- python src/ablation.py --config experiments/ablation.yaml --out outputs/ablation

Outputs and artifacts
--------------------
- outputs/models/ - saved model files (joblib)  
- outputs/plots/ - PNG and PDF plots  
- outputs/metrics.csv - fold-level metrics  
- outputs/report.html - quick HTML report (if enabled)

Example output snippet (metrics.csv)
| model | fold | accuracy | precision | recall | f1 | roc_auc |
|-------|------|----------|-----------|--------|----|---------|
| rf    | 1    | 0.78     | 0.71      | 0.67   |0.69| 0.84    |

Best practices
--------------
- Fit preprocessing only on training folds.  
- Use group-aware CV when samples come from the same subject.  
- Report mean and std for all metrics.  
- Run ablation to check each preprocessing and feature choice.

Common pitfalls
---------------
- Leaking test information by fitting scalers on full data.  
- Using accuracy in imbalanced settings as only metric.  
- Ignoring calibration for clinical decision support.

Examples and notebooks
----------------------
Open the notebooks/ folder for:
- EDA.ipynb — exploratory analysis and feature distributions.  
- ModelComparison.ipynb — side-by-side model runs and plots.  
- AblationAnalysis.ipynb — visual summary of ablation results.

Dependencies
------------
Key packages:
- python >= 3.8  
- numpy  
- pandas  
- scikit-learn  
- seaborn  
- matplotlib  
- joblib  
Install with:
```bash
pip install -r requirements.txt
```

Testing
-------
Unit tests live in tests/. Run them with pytest:
```bash
pytest -q
```

Logging and reproducibility
---------------------------
All experiments log parameters, seed, and git commit SHA. Save the random_state in outputs/experiment_meta.json.

Contributing
------------
- Fork the repository.  
- Create a feature branch.  
- Add tests for new functions.  
- Open a pull request with a clear description of the change.

Cite
----
If you use this repo in research, cite the repository and include experiment config details in your methods.

License
-------
This project uses the MIT License. See LICENSE for details.