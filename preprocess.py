import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_and_save():
    # Load the dataset
    df = pd.read_csv('train.csv')
    print("Original dataset shape:", df.shape)
    
    # Create a copy for preprocessing
    data = df.copy()
    
    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    if 'Fare' in data.columns:
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        
    # Drop columns that are less useful
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True, errors='ignore')
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    if 'Sex' in data.columns:
        data['Sex'] = le_sex.fit_transform(data['Sex'])
    if 'Embarked' in data.columns:
        data['Embarked'] = le_embarked.fit_transform(data['Embarked'])
        
    # Save the preprocessed dataset
    output_file = 'preprocessed_train.csv'
    data.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
    print("New dataset shape:", data.shape)
    print("First few rows:")
    print(data.head())

if __name__ == '__main__':
    preprocess_and_save()
