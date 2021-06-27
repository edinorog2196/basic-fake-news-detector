import streamlit as st
import matplotlib.pyplot as plt


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


rs = 7 #random state to shuffle the news. One should average over these


    

        

def read_data():
    #Read the data
    df1=pd.read_excel('../Dataset/training.xlsx')
    df2=pd.read_excel('../Dataset/test.xlsx')

    df=pd.concat([df1,df2])
    #print(df)

    #Get shape and head
    df.shape
    df.head()

    #DataFlair - Get the labels
    real_or_fake=df.value
    titles=df.title
    articles=df.article

    N_news=len(articles)
    return real_or_fake,titles,articles,N_news

def training(tfidf_vectorizer,classifier):
    real_or_fake,titles,articles,N_news=read_data()
    #The model
    start = time.time()
    #use both articles and titles
    x_train,x_test,y_train,y_test = \
        train_test_split(articles+titles,real_or_fake,train_size=ts, random_state=rs)

    

    #Fit and transform train set, transform test set
    tfidf_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U')) 
    tfidf_test=tfidf_vectorizer.transform(x_test.values.astype('U'))

    #Initialize a PassiveAggressiveClassifier
    if classifier == "PAC-I" :
        pac=PassiveAggressiveClassifier(loss='hinge',max_iter=50,n_jobs=-1)
    elif classifier == "PAC-II" :
        pac=PassiveAggressiveClassifier(loss="squared_hinge",max_iter=50,n_jobs=-1)    
    elif classifier == "LSVC":
        pac=LinearSVC(max_iter=50)
    elif classifier == "SGDC":
        pac=SGDClassifier(average=True)   
    pac.fit(tfidf_train,y_train)

    #Predict on the test set and calculate accuracy
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    end=time.time()
    accuracy= score*100.
    n_t=N_news*ts


    #Print output
    c_m=confusion_matrix(y_test,y_pred, labels=[0,1])
    tn, fp, fn, tp = c_m.ravel()/(1-ts)/N_news*100.
    st.markdown("### - Number of training news: %.f (%.2f %% of available dataset)" % (n_t,ts*100.))
    st.markdown("### - Accuracy over test news: %.2f %%" % accuracy)
    st.markdown("### - TN, FP, FN, TP: %.2f %%, %.2f %%, %.2f %%, %.2f %%" % (tn,fp,fn,tp) )
    st.markdown("### - Time employed: %.f seconds" % (end-start) )

    return pac
  
def generate_output(text,classifier):    
    
    #Initialize a TfidfVectorizer
    #we need to check the impact of max_df
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
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
    (with hinge and squared hinge loss function)\
    or the [Linear Support Vector Machine Classifier]\
(https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)\
    or the [SGDC Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)")


st.markdown("### Choose a classifier:") 
classifier = st.radio("", ('PAC-I','PAC-II','LSVC','SGDC'))     


st.markdown("### Select the training size %:") 
number = st.slider("", 1, 99)
ts=number/100
st.markdown("### Enter the text of a news article written in English or Russian language")
    
text = st.text_area("Text:", height=500)
if st.button("Run"):
    generate_output(text,classifier) 
    
    
