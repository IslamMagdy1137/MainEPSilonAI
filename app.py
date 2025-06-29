import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer,KNNImputer
from datasist.structdata import detect_outliers
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer
import joblib
import streamlit as st

# Load the saved model and inputs
# Make sure the file paths are correct
inputs = joblib.load("input_dep.h5")
model = joblib.load("final_model.h5")

# NOTE: For a robust deployment, you need to save and load the fitted transformers (like the binary encoder and
# power transformer) and apply them to the input data in this predict function before passing it to the model.
# The current implementation creates a DataFrame with original column names, but the model expects transformed features.

def predict(Name, Location, Year, Kilometers_Driven, Fuel_Type,Transmission, Owner_Type, Mileage, Engine, Power, Seats,Brand):
    # Create a DataFrame with the original column names for initial data capture
    input_data = pd.DataFrame({
        'Name': [Name],
        'Location': [Location],
        'Year': [Year],
        'Kilometers_Driven': [Kilometers_Driven],
        'Fuel_Type': [Fuel_Type],
        'Transmission': [Transmission],
        'Owner_Type': [Owner_Type],
        'Mileage': [Mileage],
        'Engine': [Engine],
        'Power': [Power],
        'Seats': [Seats],
        'Brand': [Brand]
    })

    # --- Transformation steps (REQUIRED for correct prediction) ---
    # You need to load your fitted transformers here and apply them to input_data
    # Example (requires saving/loading the actual transformers):
    # input_data['Owner_Type'] = input_data['Owner_Type'].map(transformation) # assuming transformation is saved/loaded
    # input_data = binaryencoder.transform(input_data) # assuming binaryencoder is saved/loaded
    # numerical_cols_for_transform = ['Year','Kilometers_Driven','Mileage','Engine','Power','Seats'] # Adjust based on your training
    # input_data[numerical_cols_for_transform] = numerical_transformer.transform(input_data[numerical_cols_for_transform]) # assuming numerical_transformer is saved/loaded
    # Ensure column order and names match the training data (using 'inputs' from joblib.load is a good approach here)
    # transformed_input_data = input_data[inputs.tolist()] # Reorder and select columns based on saved inputs

    # --- Placeholder for prediction with UNTRANSFORMED data (Will likely fail) ---
    # This part needs to be replaced with prediction on transformed_input_data
    print("Input data before transformation (transformation missing):")
    print(input_data)

    # Attempting prediction directly on untransformed data for now (will likely raise error)
    # Replace this with: result = model.predict(transformed_input_data)[0]
    try:
        # As a temporary measure to get the app running, create a DataFrame with the columns the model expects
        # and try to populate it. This is still not the correct way as transformations are missing.
        # A proper fix requires applying the transformations.
        test_df_for_model = pd.DataFrame(columns=inputs)
        # This part is complex as you need to map the original inputs to the transformed columns.
        # For example, 'Owner_Type' needs to be mapped to its numerical value.
        # Binary encoded columns need to be generated.
        # Numerical features need to be scaled/transformed.
        # This temporary approach will likely still fail or give incorrect predictions.
        # It's here to show that the model expects different columns.

        # Example (highly simplified and likely incorrect without actual transformers):
        # Assuming 'inputs' lists the final column names the model expects
        # You would need logic here to populate test_df_for_model based on input_data after transformations.
        # For now, let's try to create a structure that might match if no binary encoding happened (which is not the case here)
        # This is just to prevent immediate column mismatch errors, but the values will be wrong.

        # A better temporary fix is to acknowledge the missing transformation and return an error message
        # rather than attempting prediction on incorrect data format.

        # Let's return an error message indicating missing transformation
        return "Prediction Error: Input data not transformed correctly for the model."

    except Exception as e:
        print(f"Prediction failed during temporary DataFrame creation or prediction attempt: {e}")
        return f"Prediction Error: {e}"


def main():
    st.title("Know your car price")

    # Assuming the selectbox options cover the categories seen during training
    Name = st.selectbox('Name',('Maruti Wagon','Hyundai Creta', 'Honda Jazz','Maruti Ertiga','Audi A4'
                                ,'Hyundai EON','Nissan Micra','Toyota Innova','Volkswagen Vento','Tata Indica','Ford Fusion',
                                'Mercedes-Benz SL-Class','BMW Z4','Toyota Prius','Force One','Maruti Versa','Honda WR-V',
                                'Bentley Continental',   'Lamborghini Gallardo','Jaguar F'))
    Location = st.selectbox('Location',('Mumbai', 'Hyderabad', 'Kochi', 'Coimbatore', 'Pune', 'Delhi',
       'Kolkata', 'Chennai', 'Jaipur', 'Bangalore', 'Ahmedabad'))
    Year = st.slider("Year" , min_value=1980, max_value=2040, value=2000, step=1)
    Kilometers_Driven = st.slider("Kilometers_Driven" , min_value=1, max_value=999999, value=1, step=100)
    Fuel_Type = st.selectbox("Fuel_Type",('CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'))
    Transmission = st.selectbox("Transmission",('Manual', 'Automatic'))
    Owner_Type = st.selectbox("Owner_Type",('First', 'Second', 'Fourth & Above', 'Third'))
    Mileage =  st.slider("Mileage" , min_value=1, max_value=999999, value=1, step=100)
    Engine = st.slider("Engine" , min_value=0, max_value=6000, value=100, step=100)
    Power = st.slider("Power" , min_value=10, max_value=800, value=10, step=10)
    Seats = st.slider("Seats" , min_value=1, max_value=20, value=1, step=1)
    Brand = st.selectbox("Brand",('Maruti','Hyundai','Honda','Audi',
        'Nissan',        'Toyota',    'Volkswagen',          'Tata',
          'Land',    'Mitsubishi',       'Renault', 'Mercedes-Benz',
           'BMW',      'Mahindra',          'Ford',       'Porsche',
        'Datsun',        'Jaguar',         'Volvo',     'Chevrolet',
         'Skoda',          'Mini',          'Fiat',          'Jeep',
    'Ambassador',         'Isuzu',         'ISUZU',         'Force',
       'Bentley',   'Lamborghini'))


    if st.button("test"):
        # This test button logic is separate from the predict function
        print(Name, Location, int(Year), int(Kilometers_Driven), Fuel_Type,Transmission, Owner_Type, int(Mileage), int(Engine), int(Power), int(Seats) , Brand)
        # You might want to add a print or display to the streamlit app for the test button
        st.write("Test button clicked with values:")
        st.write(f"Name: {Name}, Location: {Location}, Year: {Year}, Kilometers_Driven: {Kilometers_Driven}, Fuel_Type: {Fuel_Type}, Transmission: {Transmission}, Owner_Type: {Owner_Type}, Mileage: {Mileage}, Engine: {Engine}, Power: {Power}, Seats: {Seats}, Brand: {Brand}")


    if st.button("predict"):
        # Call the predict function and display the result
        result = predict(Name, Location, int(Year), int(Kilometers_Driven), Fuel_Type,Transmission, Owner_Type, int(Mileage), int(Engine), int(Power), int(Seats),Brand)
        st.text("The predicted price is: {}".format(result))

if __name__ == '__main__':
    main()
