# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[64]:


pwd


# In[65]:


#urls_data = pd.read_csv("urldata.csv")


# In[66]:


#urls_data=pd.read_csv('urldata.csv', error_bad_lines=False)
urls_data=pd.read_csv('urldata.csv', encoding = "ISO-8859-1")


# In[67]:


urls_data.head()


# In[68]:


type(urls_data)


# In[69]:


def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens


# In[70]:


y = urls_data["label"]


# In[71]:


url_list = urls_data["url"]


# In[72]:


vectorizer = TfidfVectorizer(tokenizer=makeTokens)


# In[73]:


X = vectorizer.fit_transform(url_list)


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[75]:


Logit = LogisticRegression()
Logit.fit(X_train, y_train)


# In[76]:


print("Accuracy ",Logit.score(X_test, y_test))


# In[77]:


print("Accuracy ",Logit.score(X_train, y_train))


# In[80]:


X_predict = ["google.com/search=jcharistech",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]


# In[81]:


X_predict = vectorizer.transform(X_predict)
New_predict = Logit.predict(X_predict)


# In[59]:


print(New_predict)


# In[82]:


X_predict1 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]


# In[83]:


X_predict1 = vectorizer.transform(X_predict1)
New_predict1 = Logit.predict(X_predict1)


# In[61]:


print(New_predict1)


# In[92]:


# Store vectors into X variable as Our XFeatures


# In[89]:


vectorizer = TfidfVectorizer()


# In[91]:


X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[93]:


# Model Building

logitmodel = LogisticRegression()	#using logistic regression
logitmodel.fit(X_train, y_train)


# In[94]:


print("Accuracy ",logitmodel.score(X_test, y_test))


# In[95]:


X_predict2 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]


# In[96]:


X_predict2 = vectorizer.transform(X_predict2)
New_predict2 = logitmodel.predict(X_predict2)
print(New_predict2)


# In[ ]:


#Checking Confusion Matrix
+         #Predicted
+ #Actual True_Pos  False_Neg
+        #False_Pos  True_Neg


# In[97]:


from sklearn.metrics import confusion_matrix


# In[98]:


predicted = logitmodel.predict(X_test)
matrix = confusion_matrix(y_test, predicted)


# In[99]:


print(matrix)


# In[ ]:


#Comparing with the diagram above
#The True positives have 12366 and the true negatives are 68752
#Hence it has a good performance since majority of the predictions falls in the diagonal TP and TN


# In[ ]:


#Classification Report
#Displays the precision, recall, F1-score and support for each class


# In[100]:


from sklearn.metrics import classification_report


# In[101]:


report = classification_report(y_test, predicted)


# In[102]:


print(report)


# In[ ]:


#Plotting Confusion Matrix


# In[103]:


# Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns


# In[104]:


matrix


# In[105]:


plt.figure(figsize=(20,10))


# In[106]:


# Confusion Matrix Graph With Seaborn
sns.heatmap(matrix,annot=True)
plt.show()


# In[107]:


# Setting formate to integer with "d"
sns.heatmap(matrix,annot=True,fmt="d")
plt.show()


# In[108]:


# Plot with Labels

plt.title('Confusion Matrix ')

sns.heatmap(matrix,annot=True,fmt="d")
# Set x-axis label
plt.xlabel('Predicted Class')
# Set y-axis label
plt.ylabel('Actual Class')
plt.show()
