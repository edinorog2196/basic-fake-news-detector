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
CURRENT_THEME = "blue"
IS_DARK_THEME = True
EXPANDER_TEXT = """
    This is a custom theme. You can enable it by copying the following code
    to `.streamlit/config.toml`:
    ```python
    [theme]
    primaryColor = "#E694FF"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#C6CDD4"
    font = "sans-serif"
    ```
    """
THEMES = [
    "light",
    "dark",
    "green",
    "blue",
]
GITHUB_OWNER = "streamlit"

# Show thumbnails for available themes.
# As html img tags here, so we can add links on them.
cols = st.beta_columns(len(THEMES))
for col, theme in zip(cols, THEMES):

    # Get repo name for this theme (to link to correct deployed app)-
    if theme == "light":
        repo = "theming-showcase"
    else:
        repo = f"theming-showcase-{theme}"

    # Set border of current theme to red, otherwise black or white
    if theme == CURRENT_THEME:
        border_color = "red"
    else:
        border_color = "lightgrey" if IS_DARK_THEME else "black"

    col.markdown(
        #f'<p align=center><a href="https://share.streamlit.io/{GITHUB_OWNER}/{repo}/main"><img style="border: 1px solid {border_color}" alt="{theme}" src="https://raw.githubusercontent.com/{GITHUB_OWNER}/theming-showcase/main/thumbnails/{theme}.png" width=150></a></p>',
        f'<p align=center><a href="https://apps.streamlitusercontent.com/{GITHUB_OWNER}/{repo}/main/streamlit_app.py/+/"><img style="border: 1px solid {border_color}" alt="{theme}" src="https://raw.githubusercontent.com/{GITHUB_OWNER}/theming-showcase/main/thumbnails/{theme}.png" width=150></a></p>',
        unsafe_allow_html=True,
    )
    if theme in ["light", "dark"]:
        theme_descriptor = theme.capitalize() + " theme"
    else:
        theme_descriptor = "Custom theme"
    col.write(f"<p align=center>{theme_descriptor}</p>", unsafe_allow_html=True)


""
with st.beta_expander("Not loading?"):
    st.write(
        "You probably played around with themes before and overrode this app's theme. Go to ☰ -> Settings -> Theme and select *Custom Theme*."
    )
with st.beta_expander("How can I use this theme in my app?"):
    st.write(EXPANDER_TEXT)

""
""



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
    
    
