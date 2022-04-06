import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

#loading the csv file and filling missing values with empty string
dataset = pd.read_csv('Crop_recommendation.csv')
dataset.fillna(value='', inplace=True)
dataset

#selecting specific columns according to our requirement
X = dataset.iloc[:,:].values
X
mapping = X[:,-1].copy()
mappingdf = pd.DataFrame(mapping)
mappingdf.insert(1,1,"0")
maparr=mappingdf.iloc[:,:].values
maparr


# In[5]:


#applying one hot encoder on closely related categorical data

le = LabelEncoder()
X[:, -1] = le.fit_transform(X[:, -1])
X


# In[6]:


#array to track encoded labels
maparr[:, -1] = le.fit_transform(maparr[:, 0])
maparrdf=pd.DataFrame(maparr)
maparrdf = maparrdf.drop_duplicates(0)
maparrdf


# In[7]:


#seperating the dependant attributes and the independent(result) attribute
y=X[:,-1]
X=np.delete(X,7,1)
y


# In[8]:


#changing the datatype of array to float32
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[10]:


#applying feature scaling to the arrays
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# # Naive Bayes

# In[11]:


from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)


# In[12]:


y_pred1 = classifier1.predict(X_test)


# In[13]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))


# In[14]:


new_input=[[90,13,45,27.7,81.9,6.82,197.34]]
new_input = np.asarray(new_input).astype(np.float32)


# In[15]:


new_output=classifier1.predict(new_input)
crop=maparrdf.loc[maparrdf[1] == new_output[0]].iloc[0,0]
crop


# # KNN

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)


# In[17]:


y_pred2 = classifier2.predict(X_test)


# In[18]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))


# In[19]:


new_output=classifier2.predict(new_input)
crop=maparrdf.loc[maparrdf[1] == new_output[0]].iloc[0,0]
crop


# # SVM

# In[20]:


from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, y_train)


# In[21]:


y_pred3 = classifier3.predict(X_test)


# In[22]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))


# In[23]:


new_output=classifier3.predict(new_input)
crop=maparrdf.loc[maparrdf[1] == new_output[0]].iloc[0,0]
crop


# # Random Forrest

# In[24]:


from sklearn.ensemble import RandomForestClassifier
classifier4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier4.fit(X_train, y_train)


# In[25]:


y_pred4 = classifier4.predict(X_test)


# In[26]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred4))


# In[27]:


new_output=classifier4.predict(new_input)
crop=maparrdf.loc[maparrdf[1] == new_output[0]].iloc[0,0]
crop


# # Majority Voting

# In[28]:


from sklearn.ensemble import VotingClassifier


# In[29]:


final_model = VotingClassifier(
    estimators=[('nb', classifier1), ('knn', classifier2), ('svc', classifier3), ('rf', classifier4)], voting='hard')


# In[30]:


final_model.fit(X_train, y_train)


# In[31]:


pred_final = final_model.predict(X_test)

pickle.dump(final_model,open('model.pkl','wb'),protocol=4)
# In[32]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred4))


# In[64]:

model= pickle.load(open('model.pkl','rb'))
new_output=model.predict(new_input)
crop=maparrdf.loc[maparrdf[1] == (model.predict(new_input))[0]].iloc[0,0]
print(crop)


# In[ ]:





# In[ ]:





# In[ ]:




