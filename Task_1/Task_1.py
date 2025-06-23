import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'Titanic-Dataset.csv'
data = pd.read_csv(file_path)

#basic information
info = data.info()
describe = data.describe()

Age = data["Age"]
cabin = data["Cabin"]
Embarked = data["Embarked"]

data.replace([""," ","NA", "N/A", "na", "null"], np.nan, inplace=True)
missing_values_age = Age.isnull().sum()
missing_values_cabin = cabin.isnull().sum()
missing_values_Embarked = Embarked.isnull().sum()


# print(info)
# print(describe)
print("Missing values in age column: ",missing_values_age)
print("Missing values in cabin column: ",missing_values_cabin)
print("Missing values in  column Embarked: ",missing_values_Embarked)
print(data[data.isnull().any(axis=1)])

## handling missing values

# Filling missing numerical values (Age)
data["Age"].fillna(data["Age"].median(), inplace=True)
# print(data["Age"].head(10))      #to check whether replaced or not

# Filling missing categorical values (Embarked)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
# print(data["Embarked"].head(10))      #to check whether replaced or not


## Normalizing numerical features using z-score: (x - mean) / std
numerical_cols = ["Age", "Fare", "SibSp", "Parch"]
for col in numerical_cols:
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col] - mean) / std


plt.figure(figsize=(10, 5))
sns.boxplot(data=data[numerical_cols])
plt.ylabel("Standardized Value") 
plt.xlabel("Features")
plt.title("Boxplot to Identify Outliers")
plt.show()

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in ["Age", "Fare", "SibSp", "Parch"]:
    data = remove_outliers_iqr(data, col)

print("Final dataset shape after outlier removal:", data.shape)




