
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open('cus_segmentation.pkl', 'rb'))

# Streamlit app setup
st.set_page_config(page_title='Segmentation of Customers', layout='centered')
st.title('Segmentation of Customers')

st.markdown(
  """
  <style>
  .stApp {
      # background-color: #0077b6;
       background-color: #967ea7;
          }
  </style>
  """,
  unsafe_allow_html=True
)


st.write()

def user_input():
  Year_Birth = st.sidebar.number_input('Input Year of Birth',value=1940,step=1,format="%d")
  Education = st.sidebar.selectbox('Education', ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'])
  Marital_Status = st.sidebar.selectbox('Marital Status', ['Single', 'Together', 'Married', 'Divorced'])
  Income = st.sidebar.number_input('Input Household Income',value=0,step=1,format="%d")
  Kidhome = st.sidebar.radio('Select Number Of Kids In Household', [0, 1, 2])
  Teenhome = st.sidebar.radio('Select Number Of Teens In Household', [0, 1, 2])
  MntWines = st.sidebar.number_input('Amount spent on Wine in last 2 years',value=0,step=1,format="%d")
  MntFruits = st.sidebar.number_input('Amount spent on Fruits in last 2 years',value=0,step=1,format="%d")
  MntMeatProducts = st.sidebar.number_input('Amount spent on Meat in last 2 years',value=0,step=1,format="%d")
  MntFishProducts = st.sidebar.number_input('Amount spent on Fish in last 2 years',value=0,step=1,format="%d")
  MntSweetProducts = st.sidebar.number_input('Amount spent on Sweet in last 2 years',value=0,step=1,format="%d")
  MntGoldProds = st.sidebar.number_input('Amount spent on Gold in last 2 years',value=0,step=1,format="%d")
  NumWebPurchases = st.sidebar.number_input('Number of purchases made through Company website',value=0,step=1,format="%d")
  NumCatalogPurchases = st.sidebar.number_input('Number of purchases made using a catalogue',value=0,step=1,format="%d")
  NumStorePurchases = st.sidebar.number_input('Number of purchases made directly in store',value=0,step=1,format="%d")

  
  data = {
      'Year_Birth': Year_Birth,
      'Education': Education,
      'Marital_Status': Marital_Status,
      'Income': Income,
      'Kidhome': Kidhome,
      'Teenhome': Teenhome,
      'MntWines': MntWines,
      'MntFruits': MntFruits,
      'MntMeatProducts': MntMeatProducts,
      'MntFishProducts': MntFishProducts,
      'MntSweetProducts': MntSweetProducts,
      'MntGoldProds': MntGoldProds,
      'NumWebPurchases': NumWebPurchases,
      'NumCatalogPurchases': NumCatalogPurchases,
      'NumStorePurchases': NumStorePurchases
  }
  
  features = pd.DataFrame(data, index=[0])
  return features

def preprocess_input(features):
  # Encode categorical features
  education_mapping = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
  marital_status_mapping = {'Single': 0, 'Together': 1, 'Married': 2, 'Divorced': 3}
  
  features['Education'] = features['Education'].map(education_mapping)
  features['Marital_Status'] = features['Marital_Status'].map(marital_status_mapping)
  
      # Ensure the order of columns is consistent with the training data
  columns_order = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 
                   'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                   'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
  
  features = features[columns_order]
  return features

def cust_segment(features):
  prediction = model.predict(features)
  cluster = ''
  if prediction.item() == 0:
      cluster = 'Cluster 1'
  elif prediction.item() == 1:
      cluster = 'Cluster 2'
  elif prediction.item() == 2:
      cluster = 'Cluster 3'
  return cluster

# Get user input
sub_df = user_input()
processed_sub_df = preprocess_input(sub_df)

# Predict customer segment
if st.button("Segment Customer"):
  segment = cust_segment(processed_sub_df)
  st.write(f'The customer belongs to {segment}')
  
