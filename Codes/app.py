import streamlit as st
import matplotlib.pyplot as plt
import pickle


import pandas as pd   #data storage


#Useful functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Preprocessors
from sklearn.feature_extraction.text import TfidfVectorizer
#try also the more complex spacy

#Link for a full list of classifiers in sklearn
#https://scikit-learn.org/stable/search.html?q=classifier

#DIFFERENT CLASSIFIERS
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier 
from sklearn.svm import LinearSVC


from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
import io
import time
from matplotlib import pyplot as plt
from matplotlib import cm



st.set_page_config(page_title = "Basic Fake News Detector")




def training(tfidf_vectorizer,classifier):
    #The model
    start = time.time()
  
    #Initialize a PassiveAggressiveClassifier
    
    if classifier == "PAC-I" :
        pac=pickle.load(open('model1','rb'))
    elif classifier == "PAC-II" :
        pac=pickle.load(open('model2','rb'))
    elif classifier == "LSVC":
        pac=pickle.load(open('model3','rb'))
    elif classifier == "SGDC":
        pac=pickle.load(open('model4','rb'))   

    return pac
  
def generate_output(text,classifier):    
    
    #Initialize a TfidfVectorizer
    #we need to check the impact of max_df
    tfidf_vectorizer = pickle.load(open('tfidf1','rb'))
    text = text.replace(',', '')
    pac=training(tfidf_vectorizer,classifier)
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


desc = "#### This web app detects fake news written in English or Russian language."
        
st.markdown("# :mag: Basic Fake News Detector :mag_right:")
st.markdown(desc)

st.markdown("This app was developed with the [Streamlit](https://streamlit.io) library.")
st.markdown("We exploit the following classifiers: [Passive Aggressive Classifier]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)\
    (with hinge and squared hinge loss function which achieved 98.04% and 98.03% accuracy on the testset respectively)\
    or the [Linear Support Vector Machine Classifier] which achieved 98.20% accuracy on the testset\
(https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)\
    or the [SGDC Classifier] which achieved 96.64% accuracy on the testset.(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)")


st.markdown("### Choose a classifier:") 
classifier = st.radio("", ('PAC-I','PAC-II','LSVC','SGDC'))     
st.markdown("### Enter the text of a news article written in English or Russian language")
    
text = st.text_area("Text:", height=500)
if st.button("Run"):
    generate_output(text,classifier) 
    
    
