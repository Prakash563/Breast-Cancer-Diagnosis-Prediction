#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('breast-cancer-wisconsin-data.csv')
df.head()


# In[2]:


df.info()


# In[3]:


df.drop(columns='id',inplace=True)


# In[4]:


df.duplicated().sum()


# In[5]:


df.describe()


# In[10]:


for i in df.select_dtypes(include=['int','float']).columns:
    sns.boxplot(df[i])
    plt.show()


# In[13]:


for i in df.select_dtypes(include=['int','float']).columns:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    iqr=q3-q1
    lower_bd=q1+1.5*iqr
    upper_bd=q3-1.4*iqr
    df[i]=df[i].clip(lower_bd,upper_bd)



# In[14]:


for i in df.select_dtypes(include=['int','float']).columns:
    sns.boxplot(df[i])
    plt.show()


# In[15]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['diagnosis']=le.fit_transform(df['diagnosis'])
df


# In[17]:


feat=df.drop('diagnosis',axis=1)
tar=df['diagnosis']


# In[20]:





# In[18]:


#Selecting best 10 columns
from sklearn.feature_selection import f_classif

scores = pd.DataFrame(
    f_classif(feat, tar)[0],
    index=feat.columns,
    columns=['F_score']
)

scores.sort_values(by='F_score', ascending=False).head(10).index
f_classif(feat,tar)[0]


# In[19]:


x=df[['perimeter_worst', 'area_worst', 'radius_worst', 'concave points_worst',
       'concave points_mean', 'perimeter_mean', 'area_mean', 'radius_mean',
       'area_se', 'concavity_mean']]
y=df['diagnosis']


# In[21]:


x


# In[22]:


y


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)


# In[26]:


#Building Model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred_train=lr.predict(x_train)
pred_test=lr.predict(x_test)
from sklearn.metrics import *
print("Train Acc is ",accuracy_score(y_train,pred_train))
print(classification_report(y_train,pred_train))
print("Test Acc is ",accuracy_score(y_test,pred_test))
print(classification_report(y_test,pred_test))


# In[27]:


#Building Model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred=dt.predict(x_train)
pred
from sklearn.metrics import *
print("Train Acc is ",accuracy_score(y_train,pred_train))
print(classification_report(y_train,pred_train))
print("Test Acc is ",accuracy_score(y_test,pred_test))
print(classification_report(y_test,pred_test))


# In[28]:


#Applying Scaler & Training Model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sc = StandardScaler()

# Fit scaler only on training data, then transform
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

lr1 = LogisticRegression()
lr1.fit(x_train_scaled, y_train)
pred_train = lr1.predict(x_train_scaled)
pred_test = lr1.predict(x_test_scaled)

print("Train Acc is ",accuracy_score(y_train,pred_train))
print(classification_report(y_train,pred_train))
print("Test Acc is ",accuracy_score(y_test,pred_test))
print(classification_report(y_test,pred_test))


# In[31]:


x_train.columns


# In[32]:


#joblib ane package lo model train dhi store cheskuntaam
import joblib
joblib.dump(lr1,'model.pkl')#variable,#filename


# In[33]:


#to save the scalar one
joblib.dump(sc,'scalar.pkl')


# In[34]:


x_train


# In[35]:


y_train


# In[36]:


pd.concat([x_train,y_train],axis=1)


# In[ ]:




