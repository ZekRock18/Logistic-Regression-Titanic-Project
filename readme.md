
# Titanic Dataset - Data Science Solution 🚢💻

## Project Overview

This project focuses on solving the famous Titanic dataset problem using data science techniques. The dataset provides information about passengers aboard the Titanic and their survival status. The goal is to predict whether a passenger survived or not based on various features such as age, sex, class, and more. 🎯

### Problem Statement

The Titanic dataset contains information about passengers on the ill-fated Titanic voyage. It includes columns like age, sex, class, and whether the passenger survived or not. The task is to build a machine learning model that can predict whether a passenger survived, given certain features.

### Solution Approach 🔧

To solve this problem, the following steps were followed:

1. **Data Loading**:
   - Loaded the Titanic training and test datasets using `pandas.read_csv()`.

2. **Data Exploration**:
   - Explored the dataset using various visualization techniques such as heatmaps and count plots to understand the data distribution and identify missing values.

3. **Handling Missing Data**:
   - Missing values in the 'Age' column were filled using an imputation strategy based on the passenger's class (`Pclass`).
   - The 'Cabin' column was dropped due to many missing values and its lack of relevance to the model.
   - Rows with missing values in the 'Embarked' column were dropped.

4. **Feature Engineering**:
   - Converted categorical columns (`Sex`, `Embarked`) into numerical format using one-hot encoding. To avoid multicollinearity, one of the dummy columns was dropped (`drop_first=True`).

5. **Modeling**:
   - A Logistic Regression model was used to predict the target variable (`Pclass`), which represents the passenger class.
   - The dataset was split into features (`X_train`) and the target variable (`y_train`).
   - A Logistic Regression model was trained on the training data, and predictions were made on the test data.

6. **Evaluation**:
   - The performance of the model was evaluated using the `classification_report` from scikit-learn, which provides precision, recall, and F1-score metrics for each class.

### Libraries Used 📚

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `seaborn`: For statistical data visualization.
- `scikit-learn`: For machine learning tasks such as model training, prediction, and evaluation.

### Results 📊

After training the Logistic Regression model, the classification report provides an evaluation of the model's performance. It includes metrics like precision, recall, and F1-score for the target classes.

### Conclusion 🎉

This solution applies standard data preprocessing and machine learning techniques to the Titanic dataset. While the model focuses on predicting the passenger class (`Pclass`), similar techniques can be used to predict survival (`Survived`) or any other features from the dataset.

### How to Run 🏃‍♂️

1. Clone or download the repository.
2. Make sure you have Python installed along with the required libraries.
3. Run the script `main.py` to train the model and see the results.

