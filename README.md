# Cybersecurity-Threat-Classification


## Overview
This project involves the classification of cybersecurity threats using machine learning models. The workflow includes data preprocessing, feature selection, model training, evaluation, and visualization of results. The models compared in this project are:
- XGBoost Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## Prerequisites
Ensure the following are installed on your system:
- Python (>= 3.10)
- Required Python libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn xgboost
  ```

## Files Included
1. Cybersecurity_Threat_Classification.ipynb: The Jupyter Notebook containing all the code for preprocessing, feature selection, model training, evaluation, and visualization.
2. Report.docx: A professional report summarizing findings, methodology, and results.
3. sampled_data.csv : Example dataset used in the project.
4. README.md: This file, explaining how to run the code and understand the outputs.

## How to Run the Code
Follow the steps below to execute the project:

### 1. Clone the Repository
If this project is hosted on a repository, clone or download it to your local system:
```bash
git clone <https://github.com/Saakshitha/Cybersecurity-Threat-Classification>
cd Cybersecurity_Threat_Classification
```

### 2. Set Up the Environment
Install the required Python packages using the `requirements.txt` file if available:
```bash
pip install -r requirements.txt
```

Alternatively, install packages manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 3. Load the Dataset
- Use the provided `sampled_data.csv` as the dataset.
- Ensure the file path in the notebook matches your local file structure.

### 4. Run the Jupyter Notebook
Launch the notebook to execute the project code:
```bash
jupyter notebook Cybersecurity_Threat_Classification.ipynb
```
Execute the cells sequentially to:
- Preprocess data
- Perform feature selection
- Train and evaluate models
- Visualize results

## Outputs
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Cohen Kappa Score, and MCC for all three models.
- Visualizations:
  - Confusion Matrices
  - ROC and Precision-Recall Curves
  - Feature Importance Graphs
- Summary Report: `Report.docx` contains a professional summary of findings and insights.

## Key Notes
1. The project identifies XGBoost as the best-performing model based on evaluation metrics and visualizations.
2. Hyperparameters can be adjusted in the model training cells to optimize performance further.
3. Results may vary depending on the dataset used.
