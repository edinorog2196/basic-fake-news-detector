import streamlit as st
import pickle
import io

import pandas as pd   #data storage
from pathlib import Path
#RUN:
#python3 model_generation.py [first to generate the models]
#streamlit run app.py [to launch the app]
#try also the more complex spacy

#Link for a full list of classifiers in sklearn
#https://scikit-learn.org/stable/search.html?q=classifier

from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from matplotlib import cm

# This code is different for each deployed app.

IS_DARK_THEME = True



st.set_page_config(page_title = "Basic Fake News Detector")


def training(classifier):
    #The model
  
    
    if classifier == "PAC-I" :
        pac=pickle.load(open('/app/basic-fake-news-detector/Codes/model1','rb'))
    elif classifier == "PAC-II" :
        pac=pickle.load(open('/app/basic-fake-news-detector/Codes/model2','rb'))
    elif classifier == "LSVC":
        pac=pickle.load(open('/app/basic-fake-news-detector/Codes/model3','rb'))
    elif classifier == "SGDC":
        pac=pickle.load(open('/app/basic-fake-news-detector/Codes/model4','rb'))   

    return pac
  
def generate_output(text,classifier):    
    
    #Initialize a TfidfVectorizer
    #we need to check the impact of max_df
    tfidf_vectorizer = pickle.load(open('/app/basic-fake-news-detector/Codes/tfidf1','rb'))
    text = text.replace(',', '')
    pac=training(classifier)
    data = io.StringIO(text)
    df_text = pd.read_csv(data)

    test=tfidf_vectorizer.transform(df_text)
    pred=pac.predict(test)
    if pred == 0:
         st.markdown("<h1><span style='color:red'>This is a fake news article!</span></h1>",
                     unsafe_allow_html=True)
    else:
         st.markdown("<h1><span style='color:green'> This is a real news article!</span></h1>",
                     unsafe_allow_html=True)
             
    q_text = '> '.join(text.splitlines(True))   
    q_text = '> ' + q_text
    st.markdown(q_text)

    wc = WordCloud(width = 1000, height = 600,
                random_state = 1, colormap=cm.rainbow,
                stopwords = STOP_WORDS).generate(text) 
    
    fig, ax = plt.subplots()
    ax.imshow(wc,interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)


desc = "#### This web app detects fake news written in English language."
        
st.markdown("# :mag: Basic Fake News Detector :mag_right:")
st.markdown(desc)

accuracy_PAC_I=pickle.load(open('/app/basic-fake-news-detector/Codes/accuracy1','rb'))
accuracy_PAC_II=pickle.load(open('/app/basic-fake-news-detector/Codes/accuracy2','rb'))
accuracy_LSVC=pickle.load(open('/app/basic-fake-news-detector/Codes/accuracy3','rb'))
accuracy_SGDC=pickle.load(open('/app/basic-fake-news-detector/Codes/accuracy4','rb'))
st.markdown("This app was developed with the [Streamlit](https://streamlit.io) library.")
st.markdown("We exploit the following classifiers:")
st.markdown(":red_circle: [Passive Aggressive Classifier I]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)\
        with hinge loss function (%.2f%% accuracy)" % accuracy_PAC_I)    
st.markdown(":red_circle: [Passive Aggressive Classifier II]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)\
        with squared hinge loss function (%.2f%% accuracy)" % accuracy_PAC_II) 
st.markdown(":red_circle: [Linear Support Vector Machine Classifier]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)\
    (%.2f%% accuracy)" % accuracy_LSVC) 
st.markdown(":red_circle: [Stochastic Gradient Descent Classifier]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)\
    (%.2f%% accuracy)" % accuracy_SGDC) 


st.markdown("### Choose a classifier:") 
classifier = st.radio("", ('PAC-I','PAC-II','LSVC','SGDC'))     
st.markdown("### Enter the text of a news article written in English language")
    
text = st.text_area("Text:", height=500)
if st.button("Run"):
    generate_output(text,classifier) 
    
    
