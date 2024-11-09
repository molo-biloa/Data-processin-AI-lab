import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Step 1: Data Collection
data = sns.load_dataset('titanic')

# Step 2: Data Cleaning
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Drop rows where 'age' or 'fare' is missing
data.dropna(subset=['age', 'fare'], inplace=True)

# Fill missing 'embarked' values with the mode
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

# Step 3: Handling Outliers
# Visualize outliers with boxplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data['age']).set_title('Age Outliers')
plt.subplot(1, 2, 2)
sns.boxplot(data['fare']).set_title('Fare Outliers')
plt.show()

# Cap outliers
def cap_outliers(df, column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

cap_outliers(data, 'age')
cap_outliers(data, 'fare')

# Step 4: Data Normalization
scaler = MinMaxScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Step 5: Feature Engineering
# Create family_size
data['family_size'] = data['sibsp'] + data['parch']

# Extract title from 'name' if available (Titanic dataset may not have 'name' in seaborn version)
if 'name' in data.columns:
    data['title'] = data['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
else:
    data['title'] = np.nan  # Placeholder if 'name' column is not present

# Step 6: Feature Selection
# Check correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Select features based on correlation (for example, 'age', 'fare', 'family_size', 'sex')
data = data[['age', 'fare', 'family_size', 'sex', 'survived']]

# Encode categorical 'sex' column
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Step 7: Model Building
# Split data into train/test sets
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# Build and evaluate a Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
