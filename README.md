NB Ensemble
This repository contains the code for implementing an ensemble model using Naive Bayes and AdaBoost classifiers. The project aims to enhance the predictive performance of a Naive Bayes model by combining it with the AdaBoost algorithm.

Table of Contents
Introduction
Data
Requirements
Usage
Model Training
Evaluation
Results
Contributing
License
Introduction
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. This project demonstrates how to use Gaussian Naive Bayes as the base estimator in an AdaBoost classifier to improve the accuracy of predictions.

Data
The dataset used in this project is heart.csv, which contains various features related to heart health. The target variable is binary, indicating the presence or absence of heart disease.

Requirements
Python 3.6+
Jupyter Notebook
NumPy
Pandas
Scikit-learn
Seaborn
Usage
Clone the repository:

git clone https://github.com/yourusername/nb-ensemble.git
cd nb-ensemble
Install the required packages:

pip install -r requirements.txt
Open the Jupyter Notebook:

jupyter notebook NB\ Ensemble.ipynb
Model Training
The notebook contains code for training a Gaussian Naive Bayes model combined with an AdaBoost classifier. The key steps are:

Load and preprocess the data:

python
Copy code
import pandas as pd

# Load the dataset
df = pd.read_csv('heart.csv')

# Data preprocessing steps
df.fillna(method='ffill', inplace=True)  # Handling missing values
x = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
Train the models:
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Initialize the base estimator
nb = GaussianNB()

# Initialize the AdaBoost classifier with GaussianNB as the base estimator
model = AdaBoostClassifier(base_estimator=nb, n_estimators=10)

# Train the model
model.fit(x_train, y_train)
Evaluation
Model performance is evaluated using accuracy and confusion matrix:

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions
ypred = model.predict(x_test)

# Print accuracy
print(f'Accuracy: {accuracy_score(y_test, ypred)}')

# Compute confusion matrix
cmat = confusion_matrix(y_test, ypred)

# Plot confusion matrix
sns.heatmap(cmat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
Results
The combination of Gaussian Naive Bayes with AdaBoost improves the model's accuracy. The confusion matrix provides insight into the model's performance across different classes.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.
