import numpy as np
from joblib import load
import streamlit as st
from sklearn.preprocessing import StandardScaler



clf = load('model.joblib')
sc = StandardScaler()

def prediction(CreditScore,Geography, Gender, Age, Tenure ,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    prediction = clf.predict(sc.transform(np.array([[CreditScore,Geography, 
    Gender, Age, Tenure ,Balance, NumOfProducts, HasCrCard, 
    IsActiveMember, EstimatedSalary]])))

    print(prediction)
    return prediction


def main():
    st.title("Bank Customer Artificial Neural Network")

    st.markdown(
        '''
        <div style ="background-color:grey;padding:13px">
        <h3 style ="color:black;text-align:center;"> Bank Customer Artificial Neural Network </h3>
        </div>
        '''
        ,
        unsafe_allow_html = True
    )

       # defined in the above code

    # CreditScore	Geography	Gender	Age	Tenure Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary
    CreditScore = st.text_input('CreditScore')
    Geography = st.text_input('Geography')
    Gender = st.text_input('Gender')
    Age = st.text_input('Age')
    Tenure = st.text_input('Tenure')
    Balance = st.text_input('Balance')
    NumOfProducts = st.text_input('NumOfProducts')
    HasCrCard = st.text_input('HasCrCard')
    IsActiveMember = st.text_input('IsActiveMember')
    EstimatedSalary = st.text_input('EstimatedSalary')

    result = ''
    if st.button("Predict"):
        result = prediction(CreditScore,Geography, Gender, Age, Tenure ,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        if (result > 0.5):
            st.success('The customer would exit')
        else:
            st.success('The customer would not exit')
     
if __name__=='__main__':
    main()

