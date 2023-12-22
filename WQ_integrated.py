import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

st.title("Wine Quality Index")

st.text("Please select your option")

option = st.radio("Select an option:", ("Dataset Validation", "Execution"))

if option == "Dataset Validation":
    st.subheader("Dataset")
    st.write("[Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)")

    st.subheader("Shape of the dataset")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://i.imgur.com/BtNoNKY.png" />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("First 5 data values")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://i.imgur.com/WjqNfPt.png" />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Null values")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/AmslgTo.png" />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Data Summary")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/Z5gFoG6.png", width = 700 /> <br>
                <img src="https://imgur.com/j35J34u.png" />             
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Count")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/Tg2lnyN.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Fixed Acidity")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/hRsr88a.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Volatile Acidity")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/raTZ2BA.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Citric Acid")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/OZgYpqS.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Residual Sugar")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/dVS0jyN.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Chlorides")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/psq49fb.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Free Sulphur Dioxides")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/KT8Nq1q.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Total Sulphur Dioxides")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/AIc0tSN.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )

    st.subheader("Quality vs Density")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/d9hKuQG.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs pH")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/abaHaQO.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Sulphates")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/0taKuIR.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Quality vs Alcohol")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/WPvtg0i.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Correlation")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/sVvJMJc.png", width = 900 />
    </a>''',
    unsafe_allow_html=True
            )

    st.subheader("Quality Classification")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/x4mw3xD.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Total, Training, Testing")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/WXjjJr1.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )
    
    st.subheader("Accuracy")
    st.markdown('''
    <a href="https://docs.streamlit.io">
        <img src="https://imgur.com/G7hXfDg.png", width = 300 />
    </a>''',
    unsafe_allow_html=True
            )

elif option == "Execution":

    fa = st.text_input("Please enter the fixed acidity value")
    va = st.text_input("Please enter the volatile acidity value")
    ca = st.text_input("Please enter the citric acid value")
    rs = st.text_input("Please enter the residual sugar value")
    ch = st.text_input("Please enter the chlorides value")
    fsd = st.text_input("Please enter the free sulphur dioxides value")
    tsd = st.text_input("Please enter the total sulphur dioxides value")
    den = st.text_input("Please enter the density value")
    pH = st.text_input("Please enter the pH value")
    sul = st.text_input("Please enter the sulphates value")
    al = st.text_input("Please enter the alcohol value")

    try:
        fa_f = float(fa)
        va_f = float(va)
        ca_f = float(ca)
        rs_f = float(rs)
        cl_f = float(ch)
        fsd_f = float(fsd)
        tsd_f = float(tsd)
        den_f = float(den)
        pH_f = float(pH)
        sul_f = float(sul)
        al_f = float(al)
    
    except ValueError as e:
        st.warning("Enter all the values")

    if st.button("Generate Quality"):
        wine_ds = pd.read_csv('E:\\Ekanth\\Python\\Wine_Quality\\WQ_Dataset.csv')
        x = wine_ds.drop("quality", axis = 1)
        y = wine_ds["quality"].apply(lambda y_value: "perfect" if y_value == 10 else ("good" if (y_value >= 7 and y_value < 10) else ("average" if (y_value >= 5 and y_value < 7) else ("bad" if (y_value >= 3 and y_value < 5) else "inedible"))))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        x_test_pred = model.predict(x_test)
        test_data_acc = accuracy_score(x_test_pred, y_test)

        input_data = (fa_f, va_f, ca_f, rs_f, cl_f, fsd_f, tsd_f, den_f, pH_f, sul_f, al_f)
        input_data_np_arr = np.asarray(input_data)
        input_data_reshape = input_data_np_arr.reshape(1, -1)
        pred = model.predict(input_data_reshape)
        st.write("The quality of the wine is: ")
        st.write(pred)

        st.write("The accuracy of this prediction is: ")
        st.write(test_data_acc)

