# Job Salary Range Prediction Using Machine Learning

A complete end-to-end machine learning project that predicts salary ranges (Low, Medium, High) for job positions based on job titles and descriptions using Natural Language Processing and supervised learning.

## 🎯 Project Overview

This project demonstrates a full ML workflow including:
- Data preprocessing and feature engineering
- Text vectorization using TF-IDF
- Training multiple classification models
- Cloud-based training using Google Colab
- Model deployment to Google Cloud Storage
- Cost-efficient cloud resource management

## 📊 Results

- **Best Model**: Naive Bayes
- **Test Accuracy**: 37.78%
- **Training Platform**: Google Colab (Free tier)
- **Cloud Storage**: Google Cloud Storage
- **Dataset Size**: 672 job postings

### Model Comparison

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 37.78% |
| Logistic Regression | 37.04% |
| Random Forest | 31.85% |

## 🏗️ Architecture
```
Data Collection → Data Cleaning → Feature Engineering → Model Training → Evaluation → Cloud Deployment
```

### Technology Stack

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, pandas, numpy
- **NLP**: TF-IDF Vectorization
- **Cloud Platform**: Google Cloud Platform
- **Training**: Google Colab (Free)
- **Storage**: Google Cloud Storage

## 📁 Project Structure
```
job-salary-prediction/
├── data/
│   ├── Uncleaned_DS_jobs.csv      # Original dataset
│   └── cleaned_jobs.csv            # Preprocessed data
├── models/
│   ├── model.pkl                   # Trained model
│   ├── vectorizer.pkl              # TF-IDF vectorizer
│   └── model_info.txt              # Model metadata
├── vertex_training/
│   ├── train.py                    # Vertex AI training script
│   └── requirements.txt            # Python dependencies
├── 01_data_exploration.ipynb       # Data analysis
├── 02_data_preprocessing.ipynb     # Data cleaning
├── 03_model_training.ipynb         # Local model training
├── Cloud_Training.ipynb            # Colab cloud training
└── README.md                       # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- Google Cloud account with free credits
- Git

### Local Setup
```bash
# Clone the repository
git clone https://github.com/EndritIbraimi/job-salary-prediction.git
cd job-salary-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter google-cloud-aiplatform

# Run Jupyter notebooks
jupyter notebook
```

### Google Cloud Setup
```bash
# Install Google Cloud SDK
brew install --cask google-cloud-sdk

# Authenticate
gcloud init

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage-api.googleapis.com
```

## 💡 How to Use

### Train Model Locally

1. Open `01_data_exploration.ipynb` to explore the data
2. Run `02_data_preprocessing.ipynb` to clean and prepare data
3. Execute `03_model_training.ipynb` to train models

### Train on Google Colab (Recommended)

1. Upload `Cloud_Training.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload your GCP credentials (`colab-key.json`)
3. Run all cells
4. Models automatically save to Cloud Storage

### Make Predictions
```python
import joblib

# Load model
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Predict
job_text = "senior data scientist machine learning python"
features = vectorizer.transform([job_text])
prediction = model.predict(features)[0]
print(f"Predicted Salary Range: {prediction}")
```

## 📈 Model Performance

### Classification Report (Naive Bayes)
```
              precision    recall  f1-score   support
        High       0.36      0.44      0.40        48
         Low       0.33      0.27      0.30        44
      Medium       0.44      0.42      0.43        43
```

### Key Insights

- Model performs best on Medium salary range
- 38% accuracy is reasonable for a 3-class problem with limited features
- Baseline (random guessing) would be 33%
- Room for improvement with more data and features

## ☁️ Cloud Integration

### Google Cloud Storage

- **Bucket**: `gs://job-salary-ml-data-1770491809`
- **Data**: `gs://job-salary-ml-data-1770491809/data/`
- **Models**: `gs://job-salary-ml-data-1770491809/models/`

### Cost Management

- Training: $0 (used free Google Colab)
- Storage: ~$0.02/month (minimal data)
- Total credits used: < $1 out of $226 available

## 🔄 Workflow

1. **Data Collection**: Downloaded public job salary dataset from Kaggle
2. **Data Cleaning**: Extracted salary values, cleaned text, handled missing data
3. **Feature Engineering**: Created salary categories, combined job title + description
4. **Text Vectorization**: TF-IDF with 1000 features, bigrams
5. **Model Training**: Tested 3 algorithms (Logistic Regression, Random Forest, Naive Bayes)
6. **Evaluation**: Selected best model based on accuracy
7. **Deployment**: Saved to Google Cloud Storage

## 🎓 Learning Outcomes

- End-to-end ML project development
- NLP and text classification techniques
- Cloud platform integration (GCP)
- Cost-efficient cloud resource usage
- Version control with Git/GitHub
- Professional documentation practices

## 🚧 Future Improvements

- Add more features (company size, location, years of experience)
- Try deep learning models (BERT, transformers)
- Collect more training data
- Implement a web interface for predictions
- Add model monitoring and retraining pipeline

## 📝 License

This project is for educational purposes.

## 👤 Author

**Endrit Ibraimi**
- GitHub: [@EndritIbraimi](https://github.com/EndritIbraimi)

## 🙏 Acknowledgments

- Dataset: Kaggle Data Science Job Postings
- Platform: Google Cloud Platform
- Training: Google Colab
