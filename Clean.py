import pandas as pd

# Extract - Load the dataset
data = pd.read_csv("CreditCardData.csv")

# Transforming the data
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

data['Gender'].fillna('Unknown', inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)

# Assuming 'Date' column represents the date in the format '14-Oct-20'
data['Transaction DateTime'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data.drop(['Date'], axis=1, inplace=True)

data = pd.get_dummies(data, columns=['Type of Card', 'Entry Mode', 'Type of Transaction',
                                     'Merchant Group', 'Country of Transaction', 'Bank'])

# Load
data.to_csv("CleanedCreditCardData1.csv", index=False)

