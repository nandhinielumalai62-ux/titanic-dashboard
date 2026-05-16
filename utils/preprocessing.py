import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def clean_data(df):
    data = df.copy()
    # Duplicate removal
    data.drop_duplicates(inplace=True)
    
    # Drop irrelevant columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True, errors='ignore')
    
    return data

def handle_missing_values(df):
    data = df.copy()
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].median())
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    if 'Fare' in data.columns:
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    return data

def handle_outliers(df, cols=['Age', 'Fare']):
    data = df.copy()
    for col in cols:
        if col in data.columns:
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = np.clip(data[col], lower_bound, upper_bound)
    return data

def encode_and_scale(df, save_encoders=True):
    data = df.copy()
    encoders = {}
    
    # Label encoding
    if 'Sex' in data.columns:
        le = LabelEncoder()
        data['Sex'] = le.fit_transform(data['Sex'])
        encoders['Sex'] = le
        
    if 'Embarked' in data.columns:
        le = LabelEncoder()
        data['Embarked'] = le.fit_transform(data['Embarked'].astype(str))
        encoders['Embarked'] = le
        
    # Feature scaling
    numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    scaler = StandardScaler()
    cols_to_scale = [c for c in numeric_cols if c in data.columns]
    if cols_to_scale:
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        encoders['scaler'] = scaler
        
    if save_encoders and not os.path.exists('model'):
        os.makedirs('model')
    
    if save_encoders:
        joblib.dump(encoders, 'model/encoders.pkl')
        
    return data, encoders

def full_preprocessing_pipeline(df, save_encoders=True):
    df = clean_data(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df, encoders = encode_and_scale(df, save_encoders=save_encoders)
    return df, encoders
