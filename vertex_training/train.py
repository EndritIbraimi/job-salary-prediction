import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from google.cloud import storage

# Download data from Cloud Storage
def download_data():
    bucket_name = os.environ.get('BUCKET_NAME', 'job-salary-ml-data-1770491809')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('data/cleaned_jobs.csv')
    blob.download_to_filename('cleaned_jobs.csv')
    print('✅ Data downloaded from Cloud Storage')

# Load and prepare data
def prepare_data():
    df = pd.read_csv('cleaned_jobs.csv')
    df['combined_text'] = df['job_title_clean'] + ' ' + df['job_description_clean']
    
    X = df['combined_text']
    y = df['salary_category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Test samples: {len(X_test)}')
    
    return X_train, X_test, y_train, y_test

# Vectorize text
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f'✅ Text vectorized: {X_train_tfidf.shape[1]} features')
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ''
    
    for name, model in models.items():
        print(f'\nTraining {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f'Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f'\n✅ Best model: {best_name} with accuracy {best_score:.4f}')
    
    return best_model, best_name, best_score, results

# Upload model to Cloud Storage
def upload_model(model, vectorizer, model_name, accuracy):
    bucket_name = os.environ.get('BUCKET_NAME', 'job-salary-ml-data-1770491809')
    
    # Save locally first
    os.makedirs('model_output', exist_ok=True)
    joblib.dump(model, 'model_output/model.pkl')
    joblib.dump(vectorizer, 'model_output/vectorizer.pkl')
    
    with open('model_output/model_info.txt', 'w') as f:
        f.write(f'Best Model: {model_name}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Trained on: Vertex AI\n')
    
    # Upload to Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for filename in ['model.pkl', 'vectorizer.pkl', 'model_info.txt']:
        blob = bucket.blob(f'models/{filename}')
        blob.upload_from_filename(f'model_output/{filename}')
        print(f'✅ Uploaded {filename}')

# Main training pipeline
if __name__ == '__main__':
    print('=== Starting Vertex AI Training ===\n')
    
    # Step 1: Download data
    download_data()
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Step 3: Vectorize
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    
    # Step 4: Train models
    best_model, best_name, best_score, results = train_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )
    
    # Step 5: Upload to Cloud Storage
    upload_model(best_model, vectorizer, best_name, best_score)
    
    print('\n=== Training Complete ===')
    print(f'Model saved to gs://job-salary-ml-data-1770491809/models/')
