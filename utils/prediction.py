from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import pandas as pd
import os

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    best_model = None
    best_f1 = -1
    best_name = ""
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = 0.0
            
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean Accuracy': cv_scores.mean()
        })
        
        # Track best model based on F1
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    # Save best model
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(best_model, 'model/trained_model.pkl')
    
    # Best model detailed evaluation
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cr = classification_report(y_test, y_pred_best, output_dict=True)
    
    results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
    
    # Save feature names for prediction alignment
    joblib.dump(list(X.columns), 'model/features.pkl')
    
    return results_df, best_name, best_model, cm, cr

def load_model_and_predict(input_data):
    try:
        model = joblib.load('model/trained_model.pkl')
        encoders = joblib.load('model/encoders.pkl')
        features = joblib.load('model/features.pkl')
        
        # Apply encoding and scaling
        if 'Sex' in input_data.columns and 'Sex' in encoders:
            try:
                input_data['Sex'] = encoders['Sex'].transform(input_data['Sex'])
            except:
                input_data['Sex'] = 0
                
        if 'Embarked' in input_data.columns and 'Embarked' in encoders:
            try:
                input_data['Embarked'] = encoders['Embarked'].transform(input_data['Embarked'].astype(str))
            except:
                input_data['Embarked'] = 0
                
        numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
        cols_to_scale = [c for c in numeric_cols if c in input_data.columns]
        if 'scaler' in encoders and cols_to_scale:
            input_data[cols_to_scale] = encoders['scaler'].transform(input_data[cols_to_scale])
            
        # Align columns
        input_data = input_data.reindex(columns=features, fill_value=0)
            
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return prediction, probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None
