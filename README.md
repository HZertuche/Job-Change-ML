# Job Change Prediction using Machine Learning

## Project Overview
This project predicts whether a candidate is likely to change jobs based on demographic, educational, and professional information. The goal is to help companies identify candidates genuinely interested in working with them and optimize training efforts.

## Dataset
The dataset contains information about candidates who signed up for company training programs. Key features include:

- *enrollee_id* – Unique candidate ID  
- *city* – City code  
- *city_development_index* – Scaled development index of the city  
- *gender* – Candidate gender  
- *relevent_experience* – Whether candidate has relevant experience  
- *enrolled_university* – Type of university course enrolled  
- *education_level* – Level of education  
- *major_discipline* – Major discipline of study  
- *experience* – Total years of experience  
- *company_size* – Number of employees at current employer  
- *company_type* – Type of current employer  
- *last_new_job* – Years since previous job  
- *training_hours* – Training hours completed  
- _target_ – 0: Not looking for job change, 1: Looking for job change

**Note:** The dataset is imbalanced and contains missing values.

## Tools & Technologies
- Python 
- Amazon S3 - Storage
- AWS SageMaker – For Jupyter notebooks and model training  
- AWS Glue – For data cleaning and preprocessing  
- XGBoost – Machine learning algorithm  
- Scikit-learn – For train/test split, evaluation metrics  
- Pandas / NumPy – Data manipulation  
- Matplotlib / Seaborn – Visualization


## Data Preprocessing
Steps performed in the project:

1. Missing values were imputed (Unknown or numeric replacements).  
2. Categorical features were converted to numeric using label encoding.  
3. Columns like *experience* and *last_new_job* were cleaned to handle special values like >20 or never.  
4. Derived feature *senior_candidate* added (experience ≥10 years).  
5. Data was saved in clean Parquet format using AWS Glue.


## Machine Learning Model
**Model:** XGBoost Classifier  

**Hyperparameters:**

***First Model***
n_estimators=200,
max_depth=6,
learning_rate=0.1

***Second Model***
n_estimators=350,
max_depth=6,
learning_rate=0.07,
subsample=0.9,
colsample_bytree=0.8,
scale_pos_weight=scale

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

## Visual Results
### Feature Importance
![Feature Importance](screenshot/first_model_feature_importance.png)

### Confusion Matrix
![Confusion Matrix](screenshot/first_model_confusion_matrix.png)
![Confusion Matrix](screenshot/second_model_confusion_matrix.png)

### Accuracy, Precision, Recall and F1-Score
![Accuracy, Precision, Recall and F1-Score](screenshot/first_model_stats.png)
![Accuracy, Precision, Recall and F1-Score](screenshot/second_model_stats.png)