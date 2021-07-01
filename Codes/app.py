import streamlit as st
import pickle
import io

import pandas as pd   #data storage
import numpy as np

#RUN:
#python3 model_generation.py [first to generate the models]
#streamlit run app.py [to launch the app]
#try also the more complex spacy

#Link for a full list of classifiers in sklearn
#https://scikit-learn.org/stable/search.html?q=classifier

#DIFFERENT CLASSIFIERS
import sklearn.feature_extraction._stop_words as sw
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from matplotlib import cm



st.set_page_config(page_title = "Basic Fake News Detector")
STOP_WORDS= frozenset({'may', 'whereupon', 'describe', 'top', 'however', 'amount',\
    'along', 'my', 'perhaps', 'couldnt', 'every', 'call', 'had', 'twelve', 'whom',\
    'are', 'against', 'after', 'why', 'wherever', 'it', 'below', 'also', 'within', 'over', 'though', 'who',\
    'even', 'whereas', 'by', 'alone', 'thereby', 'became', 'find', 'ltd', 'too', 'nobody', 'anyhow',\
    'yourself', 'those', 'interest', 'mostly', 'never', 'more', 'there', 'seems', 'not', 'on', \
    'nine', 'still', 'than', 'do', 'some', 'again', 'nothing', 'must', 'have', 'otherwise',\
    'or', 'above', 'another', 'same', 'detail', 'they', 'eleven', 'has', 'down',\
    'therein', 'hundred', 'others', 'yet', 'while', 'noone', 'anywhere', 'him', 'himself', 'always',\
    'someone', 'whoever', 'until', 'wherein', 'de', 'the', 'once', 'latterly', 'yours', 'due',\
    'sometime', 'your', 'four', 'therefore', 'hereby', 'whereafter', 'everything', 'third', 'hasnt', 'get',\
    'their', 'latter', 'he', 'for', 'although', 'should', 'these', 'thick', 'something', 'because', 'no', 'rather', 'cry', 'becomes',\
    'so', 'yourselves', 'enough', 'per', 'less', 'either', 'full', 'mill', 'bill', 'her', 'becoming', 'and', 'i', \
    're', 'neither', 'were', 'his', 'off', 'else', 'amoungst', 'side', 'whenever', 'bottom', 'beside', 'will', \
    'throughout', 'last', 'up', 'about', 'via', 'being', 'except', 'none', 'back', 'seeming', 'whither',\
    'through', 'where', 'almost', 'since', 'into', 'please', 'all', 'am',\
    'nevertheless', 'become', 'between', 'amongst', 'system', 'out', 'everywhere',\
    'us', 'anyway', 'seemed', 'next', 'somewhere', 'only', 'ourselves', 'when', 'an', 'somehow', 'whatever', \
    'of', 'afterwards', 'ten', 'this', 'six', 'any', 'hereupon', 'cant', 'go', 'is', 'me', 'thus',\
    'part', 'most', 'take', 'together', 'whether', 'but', 'own', 'behind', 'other', 'everyone',\
    'one', 'which', 'thereupon', 'formerly', 'mine', 'three', 'see', 'whole', 'to', 'seem', \
    'keep', 'sometimes', 'two', 'herein', 'move', 'few', 'themselves', 'ie', 'sixty',\
    'ours', 'hers', 'then', 'thin', 'across', 'at', 'been', 'name', 'front',\
    'give', 'fifteen', 'fire', 'was', 'among', 'moreover', 'done', 'con',\
    'both', 'found', 'cannot', 'here', 'around', 'can', 'fill',\
    'that', 'indeed', 'she', 'etc', 'from', 'would', 'well',\
    'former', 'five', 'itself', 'least', 'onto', 'very', 'empty', 'how', 'in', 'inc', 'already',\
    'nowhere', 'much', 'myself', 'now', 'show', 'towards', 'thereafter', 'anyone', 'un', 'eight', \
    'without', 'whose', 'our', 'eg', 'toward', 'we', 'nor', 'beyond', 'under', 'them', 'each',\
    'further', 'could', 'put', 'be', 'fifty', 'thence', 'during', 'forty', 'if', 'might', \
    'such', 'its', 'as', 'made', 'namely', 'whence', 'beforehand', 'meanwhile', 'sincere',\
    'elsewhere', 'thru', 'hence', 'hereafter', 'ever', 'twenty', 'several', 'serious', 'whereby',\
    'herself', 'before', 'co', 'upon', 'anything', 'besides', 'what', 'a', 'first', 'you', 'often', 'with', 'many'})


def training(classifier):
    #The model
  
    
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

accuracy_PAC_I=pickle.load(open('accuracy1','rb'))
accuracy_PAC_II=pickle.load(open('accuracy2','rb'))
accuracy_LSVC=pickle.load(open('accuracy3','rb'))
accuracy_SGDC=pickle.load(open('accuracy4','rb'))
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
    
    
