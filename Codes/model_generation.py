
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
import time



ts = 0.9  #training size
rs = 7 #random state to shuffle the news. One should average over these



#Read the data
df1=pd.read_excel('../Dataset/training.xlsx')
df2=pd.read_excel('../Dataset/test.xlsx')

df=pd.concat([df1,df2])

#Get shape and head
df.shape
df.head()

#DataFlair - Get the labels
real_or_fake=df.value
titles=df.title
articles=df.article

N_news=len(articles)
   


#The model
start = time.time()
#use both articles and titles
x_train,x_test,y_train,y_test = \
       train_test_split(articles+titles,real_or_fake,train_size=ts, random_state=rs)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U')) 
tfidf_test=tfidf_vectorizer.transform(x_test.values.astype('U'))


# We now train the model on different classifiers and save the trained models.
# We also test the models so we know the achieved accuracy.

#Initialize a PassiveAggressiveClassifier
    
pac1=PassiveAggressiveClassifier(loss='hinge',max_iter=50,n_jobs=-1)
      
pac1.fit(tfidf_train,y_train)

y_pred=pac1.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
accuracy_pac1= score*100.


pickle.dump(pac1,open('model1','wb'))
pickle.dump(tfidf_vectorizer,open('tfidf1','wb')) #only need to dump this once
pickle.dump(accuracy_pac1,open('accuracy1','wb'))
print('Accuracy pac1 :', accuracy_pac1)


#Initialize  PassiveAggressiveClassifier with squared hinge loss function
    
pac2=PassiveAggressiveClassifier(loss='squared_hinge',max_iter=50,n_jobs=-1)
      
pac2.fit(tfidf_train,y_train)

y_pred=pac2.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
accuracy_pac2= score*100.

pickle.dump(pac2,open('model2','wb'))
pickle.dump(accuracy_pac2,open('accuracy2','wb'))
print('Accuracy pac2 :', accuracy_pac2)


#Initialize a linear Support Vector Machine
    
lsvm=LinearSVC(max_iter=50)
      
lsvm.fit(tfidf_train,y_train)

y_pred=lsvm.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
accuracy_lsvm= score*100.

pickle.dump(lsvm,open('model3','wb'))
pickle.dump(accuracy_lsvm,open('accuracy3','wb'))
print('Accuracy lsvm :', accuracy_lsvm)



#Initialize a stochasic gradient descent classifier
    
sgd=SGDClassifier(average=True)
      
sgd.fit(tfidf_train,y_train)

y_pred=sgd.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
accuracy_sgd= score*100.

pickle.dump(sgd,open('model4','wb'))
pickle.dump(accuracy_sgd,open('accuracy4','wb'))
print('Accuracy sgd :', accuracy_sgd)


