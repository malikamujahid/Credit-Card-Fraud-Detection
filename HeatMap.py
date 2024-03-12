import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
data = pd.read_csv("CleanedCreditCardData1.csv")

# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

# Add labels and title
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Heatmap")

# Show the heatmap
plt.show()
