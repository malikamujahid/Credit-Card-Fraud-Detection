import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load your dataset into a DataFrame
data = pd.read_csv("CleanedCreditCardData1.csv")

# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Now you can use correlation_matrix for analysis or visualization
# Create a heatmap to visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
