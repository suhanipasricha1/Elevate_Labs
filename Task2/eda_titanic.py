import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df.hist(figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.show()

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='Survived', vars=['Age', 'Fare', 'SibSp', 'Parch'])
plt.suptitle('Pairplot - Feature Relationships', y=1.02)
plt.show()

categorical = ['Pclass', 'Sex', 'Embarked']
for col in categorical:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='Survived', data=df)
    plt.title(f'Survival Count by {col}')
    plt.show()

print("\nObservations:")
print("- Higher survival in Pclass 1 passengers.")
print("- Females have higher survival rates compared to males.")
print("- Passengers embarked from 'C' had slightly better survival.")
print("- Fare and Age distributions show presence of outliers.")
print("- Fare is right-skewed.")

df.describe().to_csv('eda_summary.csv')

