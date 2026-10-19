# Credit Card Fraud Detection Using Machine Learning
<img width="647" height="331" alt="Screenshot 2026-10-19 131256" src="https://github.com/user-attachments/assets/df45f925-03d0-4c91-b003-ca169d0637ea" />

## 1. Project Overview

This project uses Machine Learning to identify fraudulent credit card transactions.

I trained a Random Forest Classifier and created a web application using Streamlit to predict whether a transaction is genuine or fraudulent.

## 2. Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib
- Streamlit

## 3. Dataset

The dataset contains 284,807 transactions.

- Genuine transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target column: `Class`

**Class values:**
- `0` = Genuine transaction
- `1` = Fraudulent transaction

## 4. Project Steps

1. Loaded the dataset using Pandas.
2. Checked the dataset and removed duplicate records.
3. Separated the input features and target column.
4. Split the data into training and testing sets.
5. Trained a Random Forest Classifier.
6. Evaluated the model's performance.
7. Saved the trained model using Joblib.
8. Built a Streamlit web application.

## 5. Model Performance

| Metric | Score |
|---|---:|
| Precision | 96.97% |
| Recall | 73.33% |
| F1-score | 83.54% |

These scores are from my earlier test-set evaluation.

## 6. Application Features

- Displays dataset information.
- Predicts genuine and fraudulent transactions.
- Shows estimated fraud probability.
- Displays a fraud distribution chart.
- Shows model performance metrics.
- Displays a confusion matrix.
- Maintains prediction history during the current session.

## 7. How to Run the Project

Install the required packages:

```bash
python -m pip install streamlit pandas scikit-learn joblib
```

Run the application:

```bash
python -m streamlit run app.py
```

## 8. What I Learned

- Data cleaning and preprocessing
- Training a Machine Learning model
- Classification and model evaluation
- Precision, recall, and F1-score
- Saving and loading a trained model
- Building a web application with Streamlit

## 9. Conclusion

This project helped me understand how Machine Learning can be used to detect potentially fraudulent credit card transactions. I also learned how to deploy a trained model in an interactive web application.

**Note:** This is an educational project and is not intended for real-world financial decisions.

## Author

Balaga Jagadish

Aspiring AI/ML Engineer

<img width="647" height="331" alt="Screenshot 2026-10-19 131256" src="https://github.com/user-attachments/assets/8cda7370-817b-4608-ade3-08f5ea495ea3" />
<img width="688" height="380" alt="Screenshot 2026-10-19 132106" src="https://github.com/user-attachments/assets/7dcadd55-b65c-4f5c-9f38-984db38f5917" />
<img width="706" height="346" alt="Screenshot 2026-10-19 132056" src="https://github.com/user-attachments/assets/8973e2e6-2138-419d-a941-5d42f4a5b4c7" />

<img width="737" height="202" alt="Screenshot 2026-10-19 132117" src="https://github.com/user-attachments/assets/ae5e4b2b-0273-4943-8fe7-a674af986e38" />



