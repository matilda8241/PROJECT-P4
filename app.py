import streamlit as st
import pandas as pd
import os
import sklearn 
import scipy
import numpy as np
import sklearn.externals as extjoblib
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#loading the trained model and components 
num_imputer = joblib.load('numerical_imputer.joblib')
cat_imputer = joblib.load('categorical_imputer.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
dt_model = joblib.load('Final_model.joblib')

st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales predictor app")

st.caption("This app predicts sales patterns of Cooperation Favorita over time in different stores in Ecuador.")

# Create the input fields
input_df = {}
col1,col2 = st.columns(2)
with col1:
    input_df["year"] = st.number_input("year",step=1)
    input_df["month"] = st.slider("month", min_value=1, max_value=12, step=1)
    input_df["day"] = st.slider("day", min_value=1, max_value=31, step=1)
    input_df["dayofweek"] = st.number_input("dayofweek,0=Sun and 6=Sat",step=1)
    input_df["end_month"] = st.selectbox("end_month",['True','False'])
    input_df["onpromotion"] = st.slider("Enter the no of item on promotion",min_value=1.00,max_value=30.0,step=0.1)
with col2:   
    input_df["cluster"] = st.number_input("cluster", min_value=1,max_value=17,step=1)
    input_df["product"] = st.selectbox("product", ["AUTOMOTIVE", "CLEANING", "BEAUTY", "FOODS", "GROCERY", "STATIONERY", 
                                                   "CELEBRATION", "HOME", "HARDWARE", "LADIESWEAR", "LAWN AND GARDEN", "CLOTHING", "LIQUOR,WINE,BEER", "PET SUPPLIES"], index=2)
    input_df["oil_price"] =st.slider("Enter the current oil price",min_value=1.00,max_value=100.00,step=0.1)  
    input_df["store_nbr"] = st.slider("store_nbr",0,54)
    input_df["state"] = st.selectbox("state", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
       'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
       'El Oro', 'Esmeraldas', 'Manabi'])
    input_df["store_type"] = st.selectbox("store_type",['D', 'C', 'B', 'E', 'A'])
   

  # Create a button to make a prediction

if st.button("Predict"):
    # Convert the input data to a pandas DataFrame
        input_data = pd.DataFrame([input_df])


# Selecting categorical and numerical columns separately
        cat_col = [col for col in input_data.columns if input_data[col].dtype == "object"]
        num_col = [col for col in input_data.columns if input_data[col].dtype != "object"]


 # Apply the imputers
        input_data_cat = cat_imputer.transform(input_data[cat_col])
        input_data_num = num_imputer.transform(input_data[num_col])


 # Encode the categorical columns
        input_encoded_df = pd.DataFrame(encoder.transform(input_data_cat).toarray(),
                                   columns=encoder.get_feature_names(cat_col))

# Scale the numerical columns
        input_df_scaled = scaler.transform(input_data_num)
        input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_col)

#joining the cat encoded and num scaled
        final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)

# Make a prediction
        prediction =dt_model.predict(final_df)[0]
        

# Display the prediction
        st.write(f"The predicted sales are: {prediction}.")
        input_df.to_csv("data.csv", index=False)
        st.table(input_df)
st.balloons()