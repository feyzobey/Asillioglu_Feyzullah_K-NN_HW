# k-Nearest Neighbor (k-NN) Classifier

## Overview
This project implements a k-Nearest Neighbor (k-NN) Classifier from scratch in Python, without using pre-built machine learning libraries like scikit-learn. The classifier is trained and tested on the "Play Tennis" dataset.

## Dataset
The dataset contains weather-related features (Outlook, Temperature, Humidity, Wind) and a target variable (PlayTennis). The model predicts whether tennis will be played based on weather conditions.

## Features
- Handles both categorical and numerical features.
- Converts categorical features into numeric values using one-hot encoding for distance calculations.
- Supports Euclidean and Manhattan distance metrics.
- Performs Leave-One-Out Cross-Validation (LOOCV) and standard evaluation.
- Logs classification results, confusion matrix, and performance metrics to output files.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - (Optional) matplotlib, seaborn for visualization.

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Testing
Run the program with the following command:
```bash
python main.py
```

### Input
You will be prompted to provide:
1. **`k` (Number of neighbors)**: Enter the desired value of `k` (e.g., 3).
2. **Distance metric**: Choose either `euclidean` or `manhattan`.

### Output
The program performs:
1. **Leave-One-Out Cross-Validation (LOOCV)**:
   - Uses one instance as a test set while training on the rest.
   - Logs results to `knn_results_loocv.txt`.
2. **Standard Evaluation**:
   - Trains on the entire dataset and evaluates on all instances.
   - Logs results to `knn_results_standard.txt`.

## Output Files

1. **`knn_results_loocv.txt`**:
   - Logs classification results, accuracy, confusion matrix, and metrics for LOOCV.

2. **`knn_results_standard.txt`**:
   - Logs classification results, accuracy, confusion matrix, and metrics for standard evaluation.

## Example Output
The following is an example of LOOCV output logged in `knn_results_loocv.txt`:
```plaintext
===== k-NN Classification Results (loocv) =====
k: 3, Distance Metric: euclidean

--------------------------------------------------
| Instance |   Actual   |   Predicted  |  Correct |
--------------------------------------------------
|   1      |   No       |   Yes        |  False   |
|   2      |   No       |   No         |  True    |
...
--------------------------------------------------
Overall Accuracy: 0.85

--------------------------------------------------
Confusion Matrix:
--------------------------------------------------
                   Predicted
            |   Yes    |   No     |
------------|----------|----------|
Actual Yes  |   8      |   2      |
Actual No   |   2      |   2      |
--------------------------------------------------
Precision: 0.80
Recall: 0.73
F1 Score: 0.76
```