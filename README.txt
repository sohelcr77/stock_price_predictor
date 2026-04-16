Stock Close Price Prediction App
================================

Overview
--------
This project is a Streamlit-based web application for stock close price forecasting using a trained LSTM model.

The app allows a user to:
- upload a stock CSV file
- visualize selected stock features
- predict the next close price
- see a forecast plot based on historical close prices

The model uses the last 14 rows of stock data to predict the next closing price.

----------------------------------------
Required Files
----------------------------------------
Make sure these files are in the same project folder:

1. app.py
   The main Streamlit application file.

2. Stock_Model.h5
   The trained deep learning model.

3. scaler.pkl
   The saved StandardScaler used during training.

4. requirements.txt
   The dependency file for installation.

----------------------------------------
Expected CSV Format
----------------------------------------
The uploaded CSV file must contain the following columns:

- Open
- High
- Low
- Close
- Change %
- Volume

Optional:
- Date

Example:

Date,Open,High,Low,Close,Change %,Volume
2026-01-01,120.5,122.1,119.8,121.2,0.58%,300000
2026-01-02,121.0,123.0,120.2,122.4,0.99%,320000
2026-01-03,122.3,124.1,121.0,123.1,0.57%,310000

Notes:
- "Change %" can be stored like 0.58% or 0.58
- The CSV must contain at least 14 valid rows
- Missing or invalid values in required columns may be dropped during preprocessing

----------------------------------------
How the App Works
----------------------------------------
1. The user uploads a CSV file
2. The app checks whether all required columns exist
3. The numeric columns are cleaned
4. The "Change %" column is converted into numeric format
5. The last 14 rows are selected
6. The scaler transforms the selected rows
7. The reshaped input is passed into the LSTM model
8. The app predicts the next close price
9. The predicted close price is inverse-transformed
10. A forecast plot is displayed

----------------------------------------
Model Input Details
----------------------------------------
Number of past rows used:
14

Feature columns used:
- Open
- High
- Low
- Close
- Change %
- Volume

Target column:
- Close

Target index used in inverse transform:
3

----------------------------------------
Install Dependencies
----------------------------------------
Run:

pip install -r requirements.txt

----------------------------------------
Run the App
----------------------------------------
Use this command:

streamlit run app.py

After running, Streamlit will provide a local URL in the terminal, usually something like:

http://localhost:8501

Open that link in your browser.

----------------------------------------
requirements.txt Example
----------------------------------------
streamlit
tensorflow
pandas
numpy
scikit-learn
joblib
matplotlib

----------------------------------------
Project Structure
----------------------------------------
project_folder/
│
├── app.py
├── Stock_Model.h5
├── scaler.pkl
├── requirements.txt
└── README.txt

----------------------------------------
Common Errors and Fixes
----------------------------------------

1. Error:
X has 4 features, but StandardScaler is expecting 6 features as input.

Reason:
The scaler was trained on 6 columns, but the app was using only 4.

Fix:
Make sure the app uses:
Open, High, Low, Close, Change %, Volume

2. Error:
Missing required columns

Reason:
The uploaded CSV does not contain all required feature columns.

Fix:
Check the column names carefully and match them exactly.

3. Error:
Need at least 14 rows to make prediction.

Reason:
The LSTM requires 14 past rows.

Fix:
Upload a CSV with at least 14 valid rows.

4. Error:
No valid rows remain after cleaning the data.

Reason:
Some required columns contain invalid or non-numeric data.

Fix:
Clean the CSV values and ensure proper formatting.

5. Error loading model or scaler

Reason:
The model file or scaler file is missing or incorrectly named.

Fix:
Check that:
- Stock_Model.h5 exists
- scaler.pkl exists
- both are in the same folder as app.py

----------------------------------------
Dummy Data Generation
----------------------------------------
You can generate multiple test CSV files using a Python script.

The generated files should contain:
- Date
- Open
- High
- Low
- Close
- Change %
- Volume

This is useful for testing upload and prediction behavior before using real stock data.

----------------------------------------
Deployment
----------------------------------------
You can deploy this project using Streamlit Community Cloud.

Steps:
1. Upload all project files to a GitHub repository
2. Go to Streamlit Community Cloud
3. Connect your GitHub account
4. Select your repository
5. Choose app.py as the main file
6. Deploy

----------------------------------------
Important Notes
----------------------------------------
- This app assumes the scaler and model match the same training pipeline
- If you retrain the model, save the updated scaler again
- The inverse transform logic assumes Close is at index 3
- If your training pipeline changes, update FEATURE_COLUMNS and TARGET_INDEX in app.py

----------------------------------------
Author Notes
----------------------------------------
This app is built for simple stock close price forecasting with an LSTM model and a CSV upload workflow.
It is intended for educational, testing, and prototype deployment purposes.
