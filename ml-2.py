import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using a model built on the Palmer's Penguin's dataset. Use the form belowto get started!")

penguin_file = st.file_uploader('Upload your own penguin data')

if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    model_rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
else:
    df = pd.read_csv(penguin_file)
    df = df.dropna()
    target = df['species']
    features = df.drop(columns=['year', 'species'])
    features = pd.get_dummies(features)
    target, unique_penguin_mapping = pd.factorize(target)
    X_train, X_test, y_train, y_test = train_test_split( features, target, test_size=.8, random_state=99)
    model_rfc = RandomForestClassifier(random_state=99)
    model_rfc.fit(X_train, y_train)
    score = model_rfc.score(X_test, y_test)
    st.write('Random Forest - Accuracy: {}'.format(score))

with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
    
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

# prediction
prediction = model_rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])
prediciton_species = unique_penguin_mapping[prediction][0]
st.write('We predict your penguin is of the {} species'.format(prediciton_species))
st.write('We used a machine learning (Random Forest) model to predict the species, the features used in this prediction are ranked by relative importance below.')
st.image('features_importance.jpeg')
