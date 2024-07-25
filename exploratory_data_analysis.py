#exploratotary data analysis
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(df.head())

# Display basic information about the dataset
print(df.info())

# Descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Distribution of numerical features
df.hist(bins=30, figsize=(20, 15), color='deepskyblue')
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Distribution of categorical features
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='pclass', hue='survived')
plt.title('Survival Count by Passenger Class')
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='embarked', hue='survived')
plt.title('Survival Count by Embarkation Point')
plt.show()

# Scatter plot of Age vs Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6)
plt.title('Scatter Plot of Age vs Fare')
plt.show()

# Pairplot
sns.pairplot(df.dropna(), hue='survived', diag_kind='kde')
plt.suptitle('Pairplot of Titanic Dataset', y=1.02)
plt.show()
