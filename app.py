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

st.title("Know your car price")

inputs = joblib.load("input_dep.h5")
model = joblib.load("final_model.h5")

def predict(Name, Location, Year, Kilometers_Driven, Fuel_Type,Transmission, Owner_Type, Mileage, Engine, Power, Seats,Brand):
    test_df = pd.DataFrame(columns = inputs)
    test_df.at[0,'Name'] = Name
    test_df.at[0,'Location'] = Location
    test_df.at[0,'Year'] = Year
    test_df.at[0,'Kilometers_Driven'] = Kilometers_Driven
    test_df.at[0,'Fuel_Type'] = Fuel_Type
    test_df.at[0,'Transmission'] = Transmission
    test_df.at[0, 'Owner_Type'] = Owner_Type
    test_df.at[0, 'Mileage'] = Mileage
    test_df.at[0, 'Engine'] = Engine
    test_df.at[0, 'Power'] = Power
    test_df.at[0, 'Seats'] = Seats
    test_df.at[0, 'Brand'] = Brand

    print(test_df)
    result = model.predict(test_df)[0]
    return result

def main():
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
        print(Name, Location, int(Year), int(Kilometers_Driven), Fuel_Type,Transmission, Owner_Type, int(Mileage), int(Engine), int(Power), int(Seats) , Brand)
    if st.button("predict"):
        restult = predict(Name, Location, int(Year), int(Kilometers_Driven), Fuel_Type,Transmission, Owner_Type, int(Mileage), int(Engine), int(Power), int(Seats),Brand)

        st.text("The price is {}".format(restult))
if __name__ == '__main__':
    main()
