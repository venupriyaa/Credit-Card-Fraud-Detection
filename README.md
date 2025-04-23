# Credit Card Fraud Detection

## Project Overview
This project develops a machine learning model to identify fraudulent credit card transactions. Using a real-world dataset of European credit card transactions, built and evaluated multiple classification models while addressing the significant class imbalance problem (only 0.172% of transactions are fraudulent).

## Objectives
- Analyze patterns in credit card transaction data to identify fraud indicators
- Address the severe class imbalance using resampling techniques
- Engineer meaningful features to improve detection capabilities
- Develop and compare multiple classification models
- Optimize model performance for high fraud detection accuracy

## Dataset
The dataset contains transactions made by European cardholders in September 2013:
- 284,807 transactions with only 492 frauds (0.172%)
- Features V1-V28 are PCA transformed for confidentiality
- 'Time' represents seconds elapsed since the first transaction
- 'Amount' is the transaction value
- 'Class' is the target variable (1 for fraud, 0 for legitimate)

Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Structure
```
├── credit_card_fraud_detection.py # Main project code 
├── credit_card_fraud_model.pkl    # Saved best model
├── README.md                      # Project documentation
├── requirements.txt               # Required packages
├── plots/                        # Generated visualizations
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   ├── feature_importance.png
│   ├── original_distribution.png
│   ├── pr_curves.png
│   ├── roc_curves.png
│   └── time_amount_distribution.png
```

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib

Install requirements with:
```
pip install -r requirements.txt
```

## Implementation Steps

### 1. Exploratory Data Analysis
- Analysis showed no missing values in the dataset
- Amount statistics differed between normal and fraudulent transactions:
  - Normal transactions: Mean = $88.29, Median = $22.00
  - Fraudulent transactions: Mean = $122.21, Median = $9.25
- Several PCA components showed clear separation between fraud and normal transactions

### 2. Data Preprocessing
- Feature engineering:
  - Created hour of day from Time feature
  - Applied log transformation to Amount feature
  - Created interaction features between important variables
- Split data into training (80%) and testing (20%) sets
- Applied StandardScaler to normalize features

### 3. Handling Class Imbalance
- Implemented SMOTE to increase fraud cases from 394 to 22,745 in training data
- Combined SMOTE with random undersampling to create a more balanced dataset
- Created three different training datasets:
  - Original imbalanced data
  - SMOTE-oversampled data
  - Combined SMOTE+undersampling data

### 4. Model Development
Multiple models were implemented and compared:
- Logistic Regression with different balancing techniques
- Random Forest with SMOTE
- Gradient Boosting with SMOTE
- Hyperparameter-tuned Random Forest (our best model)

### 5. Model Evaluation
Models were evaluated using metrics appropriate for imbalanced data:
- Random Forest with SMOTE achieved outstanding performance:
  - Accuracy: 99.95%
  - Precision: 86.60%
  - Recall: 85.71%
  - F1 Score: 86.15%
  - ROC AUC: 96.86%
  - PR AUC: 87.83%

## Running the Project

1. Clone this repository:
```
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the project directory as `creditcard.csv`

4. Run the main script:
```
python credit_card_fraud_detection.py
```

5. To use the saved model for prediction:
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('credit_card_fraud_model.pkl')

# Prepare transaction data (must have same features as training data)
# Make predictions
def predict_fraud(transaction_data, model, threshold=optimal_threshold):
    fraud_probability = model.predict_proba(transaction_data)[:, 1]
    fraud_prediction = (fraud_probability >= threshold).astype(int)
    
    return {
        'fraud_probability': fraud_probability,
        'is_fraud': fraud_prediction,
        'threshold_used': threshold
    }
```

## Results and Performance

### Model Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| LR (Imbalanced) | 0.9745 | 0.0586 | 0.9184 | 0.1101 | 0.9713 | 0.7634 |
| LR (SMOTE) | 0.9730 | 0.0556 | 0.9184 | 0.1049 | 0.9697 | 0.7683 |
| LR (Combined) | 0.9859 | 0.1009 | 0.9082 | 0.1816 | 0.9691 | 0.7626 |
| RF (SMOTE) | 0.9995 | 0.8660 | 0.8571 | 0.8615 | 0.9686 | 0.8783 |
| GB (SMOTE) | 0.9984 | 0.5152 | 0.8673 | 0.6464 | 0.9794 | 0.7693 |

### Random Forest Confusion Matrix
```
[[56851    13]
 [   14    84]]
```

The confusion matrix shows:
- True Negatives: 56,851 (correctly identified legitimate transactions)
- False Positives: 13 (legitimate transactions incorrectly flagged as fraud)
- False Negatives: 14 (fraudulent transactions missed)
- True Positives: 84 (correctly identified fraudulent transactions)

### Feature Importance
The Random Forest model identified the most important features for fraud detection, providing insights into fraud patterns and indicators.

## Conclusion

This project successfully developed a high-performance credit card fraud detection system with the following achievements:

1. **Exceptional Accuracy**: The Random Forest model achieved 99.95% accuracy, detecting 85.71% of all fraudulent transactions while maintaining 86.60% precision.

2. **Effective Class Imbalance Handling**: SMOTE resampling significantly improved model performance by addressing the extreme class imbalance.

3. **Minimal False Positives**: The model generated only 13 false alarms out of 56,864 legitimate transactions, minimizing customer inconvenience.

4. **Valuable Feature Engineering**: Transforming the Amount feature and creating interaction features enhanced the model's discriminative power.

5. **Cost-Effective Solution**: By minimizing both false positives and false negatives, the model provides a cost-effective fraud detection system.

6. **Production-Ready Implementation**: The project includes a reusable prediction function and a saved model file for easy deployment.

The high performance of the Random Forest model demonstrates that machine learning techniques can effectively detect credit card fraud even with highly imbalanced datasets. The model's ability to maintain high precision while achieving good recall makes it suitable for real-world implementation.

## Future Improvements
- Implement more advanced ensemble techniques
- Explore deep learning approaches for fraud detection
- Develop a real-time transaction scoring system
- Implement model monitoring and retraining pipeline
- Add explainability features to help fraud analysts understand model decisions
