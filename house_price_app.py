import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ── Load the model 
model = pickle.load(open('house_price_model.pkl', 'rb'))

# ── Page config 
st.set_page_config(page_title='House Price Prediction', page_icon='🏠', layout='centered')

st.title('🏠 House Price Prediction App')
st.markdown('Enter the house details below and click **Predict Price** to get an estimate.')
st.divider()

# ── Inputs 
col1, col2 = st.columns(2)

with col1:
    st.subheader('🏗️ House Details')
    square_footage       = st.number_input('Square Footage (sq ft)',  min_value=200,  max_value=10000, value=1500)
    num_bedrooms         = st.number_input('Number of Bedrooms',      min_value=1,    max_value=10,    value=3)
    num_bathrooms        = st.number_input('Number of Bathrooms',     min_value=1,    max_value=10,    value=2)
    year_built           = st.number_input('Year Built',              min_value=1900, max_value=2024,  value=2000)

with col2:
    st.subheader('📐 Property Details')
    lot_size             = st.number_input('Lot Size (sq ft)',        min_value=500,  max_value=50000, value=5000)
    garage_size          = st.number_input('Garage Size (cars)',      min_value=0,    max_value=5,     value=2)
    neighborhood_quality = st.slider('Neighborhood Quality (1 = Poor → 10 = Excellent)',
                                      min_value=1, max_value=10, value=7)

st.divider()

# ── Predict 
if st.button('🔍 Predict Price', use_container_width=True):

    # Build input dataframe — same column order as notebook's X
    input_data = pd.DataFrame({
        'Square_Footage'       : [square_footage],
        'Num_Bedrooms'         : [num_bedrooms],
        'Num_Bathrooms'        : [num_bathrooms],
        'Lot_Size'             : [lot_size],
        'Garage_Size'          : [garage_size],
        'Neighborhood_Quality' : [neighborhood_quality],
        'Year_Built'           : [year_built],
    })


    # Scale using StandardScaler (fit on single row — matches inference pattern)
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Predict (model outputs log price → reverse with exp)
    log_prediction = model.predict(input_scaled)[0]
    predicted_price = round(np.exp(log_prediction), 2)

    st.divider()
    st.success(f'🏷️ **Estimated House Price: ${predicted_price:,.2f}**')

    # Price range band (±10%)
    low  = round(predicted_price * 0.90, 2)
    high = round(predicted_price * 1.10, 2)
    st.info(f'📊 **Estimated Range:** ${low:,.2f} — ${high:,.2f}  *(±10% confidence band)*')

    # Key feature summary
    st.divider()
    st.subheader('📋 Input Summary')
    summary = pd.DataFrame({
        'Feature': ['Square Footage', 'Bedrooms', 'Bathrooms',
                    'Lot Size', 'Garage Size', 'Neighborhood Quality', 'Year Built'],
        'Value': [f'{square_footage} sq ft', num_bedrooms, num_bathrooms,
                  f'{lot_size} sq ft', garage_size, f'{neighborhood_quality}/10', year_built]
    })
    st.table(summary)

    st.caption('💡 Tip: Square footage and neighborhood quality have the strongest influence on price.')
