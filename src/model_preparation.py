import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the preprocessed data
kickstarter_df = pd.read_csv('data/preprocessed_kickstarter.csv')

# Load encoded features
region_encoded = pd.read_csv('data/region_encoded.csv')
category_encoded = pd.read_csv('data/category_encoded.csv')
main_category_encoded = pd.read_csv('data/main_category_encoded.csv')
deadline_weekday_encoded = pd.read_csv('data/deadline_weekday_encoded.csv')
launched_at_weekday_encoded = pd.read_csv('data/launched_at_weekday_encoded.csv')
created_at_weekday_encoded = pd.read_csv('data/created_at_weekday_encoded.csv')

## Constructing X and y
categorical_X = kickstarter_df[['country', 'category',
            'main_category','deadline_weekday', 
            'created_at_weekday','launched_at_weekday', 
            'show_feature_image' , 'video']]

numerical_X = kickstarter_df[['usd_goal','name_len_clean', 
        'blurb_len_clean','project_duration', 
        'creation_to_launch','deadline_yr', 
        'deadline_month', 'deadline_day', 
        'deadline_hr',
        'created_at_month','created_at_day', 
        'created_at_hr',
        'launched_at_month', 'launched_at_day', 
        'launched_at_hr']]

categorical_X = pd.concat([category_encoded, main_category_encoded, region_encoded, deadline_weekday_encoded,
    launched_at_weekday_encoded,created_at_weekday_encoded,
    kickstarter_df[['show_feature_image' , 'video']]], axis = 1)

# Log transforming outliers in numerical_X
numerical_X.loc[:, 'creation_to_launch'] = np.log(numerical_X['creation_to_launch']+1).astype('float64')
numerical_X.loc[:, 'usd_goal'] = np.log(numerical_X['usd_goal']+1).astype('float64')

# Creating X and y
X = pd.concat([numerical_X, categorical_X], axis = 1)
y = kickstarter_df['state']

# Data Validation
print("Feature matrix shape:", X.shape)
print("Target variable distribution:\n", y.value_counts(normalize=True))
print("\nMissing values:\n", X.isnull().sum())
print("\nFeature types:\n", X.dtypes)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = numerical_X.columns
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save prepared data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
pd.Series(y_train).to_csv('data/y_train.csv', index=False)
pd.Series(y_test).to_csv('data/y_test.csv', index=False)

# Save transformers
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

# Print summary statistics
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\nClass distribution in training set:\n", pd.Series(y_train).value_counts(normalize=True))
print("\nClass distribution in test set:\n", pd.Series(y_test).value_counts(normalize=True)) 