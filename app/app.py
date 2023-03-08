import ssl
ssl._create_default_https_context= ssl._create_unverified_context
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the data
data = pd.read_csv('bank.csv',sep= ';')

#treatment exposure definition
import numpy as np
col         = 'duration'
conditions  = [ data[col] <=150, (data[col] >150)  ]
choices     = [ 0, 1 ]
    
data["treatment_tag"] = np.select(conditions, choices, default=np.nan)

#treatment exposure definition
import numpy as np
col         = 'duration'
conditions  = [ data[col] <=150, (data[col] >150)  ]
choices     = [ 0, 1 ]
    
data["treatment_tag"] = np.select(conditions, choices, default=np.nan)

# Define the categorical and numerical columns
# Preprocess data
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
treatment_tag = 'treatment_tag'
conversion = 'y'

X = data.drop([conversion], axis=1)
y = data[conversion]

# Preprocess the data
encoder = OneHotEncoder(sparse=False)
X_cat = encoder.fit_transform(data[cat_cols])
scaler = StandardScaler()
X_num = scaler.fit_transform(data[num_cols])
X = np.hstack((X_cat, X_num))
y = data['y']
treatment = data['treatment_tag']

# Split the data into treatment and control groups
X_treat = X[treatment == 1]
y_treat = y[treatment == 1]
X_ctrl = X[treatment == 0]
y_ctrl = y[treatment == 0]

# Fit a classification model for each group
model_treat = LogisticRegression()
model_treat.fit(X_treat, y_treat)
model_ctrl = LogisticRegression()
model_ctrl.fit(X_ctrl, y_ctrl)

# Calculate the uplift score for each observation
uplift_scores = model_treat.predict_proba(X)[:, 1] - model_ctrl.predict_proba(X)[:, 1]

# Rank the observations and recommend products to the top-ranked individuals
recommendations = data.loc[uplift_scores.argsort()[::-1], ['age', 'job', 'marital', 'education', 'balance']]

# Define the Streamlit app
def app():
    st.title('Uplift Modeling App')
    
    # Create a slider for selecting the number of recommendations
    num_rec = st.slider('Number of Recommendations', 1, 100, 10)
    
    # Show the top recommended individuals and their features
    st.subheader('Top Recommended Individuals')
    st.table(recommendations.head(num_rec))
    
    # Show the model performance for each group
    st.subheader('Model Performance')
    st.write('Treatment Group:')
    y_treat_pred = model_treat.predict(X_treat)
    st.write('Accuracy:', accuracy_score(y_treat, y_treat_pred))
    st.write('F1 Score:', f1_score(y_treat, y_treat_pred, pos_label='yes'))
    st.write('Precision:', precision_score(y_treat, y_treat_pred, pos_label='yes'))
    st.write('Recall:', recall_score(y_treat, y_treat_pred, pos_label='yes'))
    
    st.write('Control Group:')
    y_ctrl_pred = model_ctrl.predict(X_ctrl)
    st.write('Accuracy:', accuracy_score(y_ctrl, y_ctrl_pred))
    st.write('F1 Score:', f1_score(y_ctrl, y_ctrl_pred, pos_label='yes'))
    st.write('Precision:', precision_score(y_ctrl, y_ctrl_pred, pos_label='yes'))
    st.write('Recall:', recall_score(y_ctrl, y_ctrl_pred, pos_label='yes'))
    
if __name__ == '__main__':
    app()
