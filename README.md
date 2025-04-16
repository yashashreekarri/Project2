# Project 2

### GROUP MEMBERS:
- Samarth Rajput A20586237  
- Jenil Panchal A20598955  
- Yashashree Reddy Karri A20546825  
- Krishna Reddy A20563553

---

## Overview  
This project implements the Gradient Boosting Tree Classifier algorithm from scratch, based on Sections 10.9–10.10 of The Elements of Statistical Learning (2nd Edition). It supports binary and multi-class classification, and does not rely on any external ML libraries like sklearn.

---

## Installation & Setup

1. Clone the Repository:
```bash
git clone https://github.com/YOUR_USERNAME/Project2.git
cd Project2
```

2. Create & Activate Virtual Environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```
Includes:
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- pytest

---

##  Running Tests
To validate model correctness:
```bash
$env:PYTHONPATH="."   # PowerShell
pytest tests/
```

---

##  Run Training and Visualization
```bash
python main.py
```
- Trains binary and multi-class models
- Prints classification reports
- Visualizes decision boundary (if 2D data)

---

## Test Cases
`tests/test_gradient_boosting.py` includes:

- `test_gradient_boosting_binary` → checks binary classification accuracy > 80%
- `test_gradient_boosting_multiclass` → checks multi-class accuracy > 70%

---

## Parameters

The GradientBoostingClassifier class exposes the following parameters for tuning model performance:

- n_estimators:  
  Specifies the number of boosting rounds (i.e., the number of trees to be added sequentially).  
  Default value: 100

- learning_rate:  
  Controls the contribution of each tree in the boosting sequence. A smaller value results in slower learning and potentially better generalization.  
  Default value: 0.1

- max_depth:  
  Determines the maximum depth of each individual regression tree. Controls the model complexity and risk of overfitting.  
  Default value: 3

- min_samples_split:  
  The minimum number of samples required to split an internal node in a tree. Helps prevent the model from learning from very small subsets of the data.  
  Default value: 2

- min_samples_leaf:  
  Specifies the minimum number of samples required to be at a leaf node. Used for regularization and controls overfitting.  
  Default value: 1


---

##  Project Questions

### 1. What does the model do, and when should it be used?
Gradient Boosting is an ensemble learning method that sequentially builds decision trees on the residuals of the previous trees. It's used when you want a high-performing model on structured data, often outperforming random forests or logistic regression.

### 2. How did you test your model?
- Used synthetic datasets
- Verified accuracy scores with known separable classes
- Created pytest unit tests
- Visualized boundary performance

### 3. What parameters are exposed?
All core hyperparameters: number of trees, learning rate, tree depth, sample size thresholds

### 4. Limitations and Challenges

- Noisy Data:  
  The model may struggle to generalize well when the dataset contains significant noise. This can lead to overfitting, especially when too many trees are added.  
  Workaround: Tune the learning rate to be lower or limit the number of trees (n_estimators) to prevent overfitting.

- High-Dimensional Data:  
  When the number of features is very large, the model may become computationally expensive and harder to interpret. It may also capture spurious relationships.  
  Workaround: Apply dimensionality reduction techniques such as Principal Component Analysis (PCA) or use manual feature selection to retain only the most relevant features.

- Categorical Features:  
  This implementation is designed for numeric features and does not natively handle categorical data.  
  Workaround: Convert categorical variables into numerical format using encoding techniques like one-hot encoding or label encoding prior to training.

---