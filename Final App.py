
import pandas as pd
import streamlit as st
from pickle import load
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.image as mp


st.set_page_config(layout='wide')
st.title("***HEART DISEASE IDENTIFICATION***")

image = mp.imread(r"C:\Users\sumit\deployement_3pm\Heart.png")
st.image(image)
st.header("What is Heart disease?")
st.subheader("The term **heart diseas** refers to several types of heart conditions. The most common type of heart disease in the United States is coronary artery disease (CAD), which affects the blood flow to the heart. Decreased blood flow can cause a heart attack.")
st.subheader("**Key risk factors for heart disease     :** High blood pressure, high blood cholesterol and smoking. ")
st.subheader("**Signs and symptoms of Heart Disease    :** Chest pain,Shortness of breath,Dizziness and Fainting")
st.sidebar.header('User Inpuut Parameters')

BMI = st.sidebar.number_input('Enter Your Body Mass Index',0.0,100.0)
Smoking = st.sidebar.selectbox('Are you Smoker?',['No','Yes'])
AlcoholDrinking = st.sidebar.selectbox('Are you Alcoholic?',['No','Yes'])
Stroke = st.sidebar.selectbox('Ever had heart stoke before?',['No','Yes'])
PhysicalHealth = st.sidebar.number_input('What is your physical health condition?',0.0,30.0)
MentalHealth = st.sidebar.number_input('What is your mental health condition?',0.0,30.0)
DiffWalking = st.sidebar.selectbox('Do you face difficulty while walking?',['No','Yes'])
Sex = st.sidebar.selectbox('Sex',['Female','Male'])
AgeCategory = st.sidebar.selectbox('Select your age range',('70-74', '75-79', '55-59', '60-64', '80 or older', '65-69','40-44', '50-54', '25-29', '45-49', '30-34', '35-39', '18-24'))
Race = st.sidebar.selectbox('Select your Race', ['White', 'Black', 'Hispanic', 'Other', 'Asian','American Indian/Alaskan Native'])
Diabetic = st.sidebar.selectbox('Do you have Diabetics?', ['No','Yes'])
PhysicalActivity = st.sidebar.selectbox('Are you able to do physical activities?', ('No','Yes'))
GenHealth = st.sidebar.selectbox('How is your general health condition?',['Very good', 'Good', 'Poor', 'Fair', 'Excellent'])
SleepTime = st.sidebar.number_input('How many hours you sleep in a day?',0.0,24.0)
Asthma = st.sidebar.selectbox('Do you have Asthma?',['No','Yes'])
KidneyDisease = st.sidebar.selectbox('Do you have any Kidney disease?',['No','Yes'])
SkinCancer = st.sidebar.selectbox('Do you have Skin cancer?',['No','Yes'])
submit = st.sidebar.button('Submit')


data = {'BMI':BMI,
        'Smoking':Smoking,
        'AlcoholDrinking':AlcoholDrinking,
        'Stroke':Stroke,
        'PhysicalHealth':PhysicalHealth,
        'MentalHealth':MentalHealth,
        'DiffWalking':DiffWalking,
        'Sex':Sex,
        'AgeCategory':AgeCategory,
        'Race':Race,
        'Diabetic':Diabetic,
        'PhysicalActivity':PhysicalActivity,
        'GenHealth':GenHealth,
        'SleepTime':SleepTime,
        'Asthma':Asthma,
        'KidneyDisease':KidneyDisease,
        'SkinCancer':SkinCancer}

data= pd.DataFrame(data,index=[0])
st.header('User Input parameters :')
st.write(data)


heart = pd.read_csv(r"C:\Users\sumit\P122 -- Heart Disease\heart_2020_cleaned.csv")
heart = heart.drop(columns=['Unnamed: 0','HeartDisease'])
df = pd.concat([data,heart],axis=0)


encode = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
df_non_numeric =df.select_dtypes(['object'])
non_numeric_cols = df_non_numeric.columns.values
for col in non_numeric_cols:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
scelar = StandardScaler()
scelar = scelar.fit_transform(df) 
df = pd.DataFrame(data=scelar,columns=df.columns)   
df = df[:1]


rf_model = load(open(r"C:\Users\sumit\P122 -- Heart Disease\rf_model.pkl",'rb'))

st.subheader('Predicted Result :')

def result():
    prediction = rf_model.predict(df)
    if prediction == 0:
        results = st.header("You don't have heart disease")
    else:
        results = st.header("You have a Heart disease")
        results = st.write("**for more information :**   Youtube link : https://www.youtube.com/watch?v=afYCN3Upy_w" )
        resuls = st.write("**web link :** https://www.cdc.gov/heartdisease/about.htm#:~:text=The%20term%20%E2%80%9Cheart%20disease%E2%80%9D%20refers,can%20cause%20a%20heart%20attack.")

    return results
        
results = result()
        
if submit is True:
    st.write(results)

