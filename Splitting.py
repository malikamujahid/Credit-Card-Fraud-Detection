import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load your dataset (assuming you have it in a CSV file)
data = pd.read_csv("CleanedCreditCardData1.csv")

# Drop non-numeric columns
non_numeric_columns = ['Transaction ID', 'Day of Week', 'Shipping Address', 'Country of Residence',
                       'Gender', 'Transaction DateTime']
data = data.drop(columns=non_numeric_columns)

# 2. Define your features (X) and target (y)
# "Fraud" is the column indicating whether a transaction is fraudulent or not
X = data.drop("Fraud", axis=1)
y = data["Fraud"]

# 3. Split the dataset into training (70%) and test (30%) sets
# You can adjust test_size and random_state as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Remove currency symbols from 'Amount' column
X_train['Amount'] = X_train['Amount'].str.replace('£', '').astype(float)
X_test['Amount'] = X_test['Amount'].str.replace('£', '').astype(float)
from sklearn.impute import SimpleImputer

#  filling missing values with the median
imputer = SimpleImputer(strategy='median')

#  training data and transform both training and test data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create and train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on test data
logistic_predictions = logistic_model.predict(X_test)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test, logistic_predictions)
report_lr = classification_report(y_test, logistic_predictions)

# Create and train Random Forest Classifier model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Make predictions on test data
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate Random Forest Classifier model
accuracy_rf = accuracy_score(y_test, random_forest_predictions)
report_rf = classification_report(y_test, random_forest_predictions)

# Print evaluation results
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Classification Report:\n", report_lr)

print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:\n", report_rf)
