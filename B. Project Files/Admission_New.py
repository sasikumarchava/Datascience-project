#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading dataset
data = pd.read_csv('Admission Data.csv')


# # Data Preprocessing

# In[3]:


#Data inspection
data.head()


# In[4]:


# checking the dataype of the parameters(Columns)
data.info()


# In[5]:


# Descriptive Analysis
data.describe()


# In[6]:


data.mean()


# In[7]:


# Checking the null values in dataset
data.isnull().any()


# In[8]:


data.isnull().sum()


# # Data Visualization

# In[9]:


#BarPlot
plt.bar(data['TOEFL_Score'],data['Chance_of_Admit'])


# In[10]:


#BarPlot
plt.bar(data['GRE_Score'],data['Chance_of_Admit'])


# In[11]:


#BarPlot
plt.bar(data['University_Rating'],data['Chance_of_Admit'])


# In[12]:


#Pie Chart
plt.pie(data['Chance_of_Admit'],autopct='%.2f')


# In[13]:


plt.plot(data['Chance_of_Admit'],data['TOEFL_Score'],marker='o')
plt.plot(data['Chance_of_Admit'],data['GRE_Score'],marker='x')
plt.xlabel('Chance_of_Admit')
plt.ylabel('TOEFL and GRE Scores')


# In[14]:


#Pair Plot
sns.pairplot(data)


# In[15]:


#Heat Map
hm=data.corr()
sns.heatmap(hm)


# In[16]:


#extracting numerical columns values
x_independent = data.iloc[:,:-1]
y_dependent=data.iloc[:,8:9]


# In[17]:


#Dropping unnecessary columns
x_independent=x_independent.drop(['Serial_No'],axis=1)


# In[18]:


x_independent


# In[19]:


#Checking Outliers
sns.boxplot(x_independent)


# In[20]:


sns.boxplot(x_independent.LOR)


# In[21]:


sns.boxplot(x_independent.CGPA)


# In[22]:


#Calculating quartiles for x_independent
quantile = x_independent.quantile(q=[0.25,0.75])
quantile


# In[23]:


#IQR
IQR = quantile.iloc[1] - quantile.iloc[0]
IQR


# In[24]:


#calculating upper extreme
upper_extreme = quantile.iloc[1] + (1.5*IQR)
upper_extreme


# In[25]:


#calculating lower extreme
lower_extreme = quantile.iloc[0] - (1.5*IQR)
lower_extreme


# In[26]:


#removing outliers from the extracted numeric columns

removed_outliers = x_independent[(x_independent >=lower_extreme)&(x_independent <=upper_extreme)]
removed_outliers


# In[65]:


removed_outliers.to_csv('file1.csv')


# In[27]:


#Finding null values after removing outliers
removed_outliers.isnull().any()


# In[28]:


#Replacing null values
removed_outliers['LOR'].fillna(removed_outliers['LOR'].mean(),inplace=True)
removed_outliers['CGPA'].fillna(removed_outliers['CGPA'].mean(),inplace=True)


# In[29]:


#Checking whether null values are removed
removed_outliers.isnull().sum()


# In[30]:


removed_outliers


# In[31]:


#Removed outliers Boxplot
sns.boxplot(removed_outliers)


# # Scaling

# In[32]:


name=removed_outliers.columns
name


# In[33]:


#Normalisation
from sklearn.preprocessing import MinMaxScaler


# In[34]:


scale=MinMaxScaler()


# In[35]:


X_scaled=scale.fit_transform(removed_outliers)


# In[36]:


X=pd.DataFrame(X_scaled,columns=name)


# In[37]:


X


# # Split the data into dependent and independent variables

# In[38]:


#Split the data into dependent and independent variables
x=X #independent values
y=y_dependent


# In[39]:


x


# In[40]:


y


# # Train-Test Split

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)


# # Build the Model

# In[43]:


#Multi Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[44]:


#importing Ridge Regression
from sklearn.linear_model import Ridge
r=Ridge()


# # Train the model

# In[45]:


#Training Linear Regression Model
model.fit(x_train.values,y_train.values)


# In[46]:


#Training Ridge Regression Model
r.fit(x_train,y_train)


# # Testing the Model

# In[47]:


#Testing Linear Regression Model
pred=model.predict(x_test)


# In[48]:


#Testing Ridge Regression Model
pred1=r.predict(x_test)


# # Measure the performance using Metrics

# In[49]:


from sklearn import metrics


# In[50]:


# R Squared for MultiLinear Regression
print(metrics.r2_score(y_test,pred))


# In[51]:


# R Squared for Ridge Regression
print(metrics.r2_score(y_test,pred1))


# In[52]:


#MSE (Mean Square Error)  for MultiLinear Regression
print(metrics.mean_squared_error(y_test,pred))


# In[53]:


#MSE (Mean Square Error) for Ridge Regression
print(metrics.mean_squared_error(y_test,pred1))


# In[54]:


#RMSE(Root Mean Square Error) for MultiLinear Regression
print(np.sqrt(metrics.mean_squared_error(y_test,pred)))


# In[55]:


#RMSE(Root Mean Square Error) for Ridge Regression
print(np.sqrt(metrics.mean_squared_error(y_test,pred1)))


# In[56]:


def calc(a,b,c,d,e,f,g):
    myarr = np.array([a,b,c,d,e,f,g])
    mydf = pd.Series(myarr, index=["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"])
    removed_outliers.columns = removed_outliers.columns.astype(str)
    frames = [removed_outliers,mydf.to_frame().T]
    result = pd.concat(frames,ignore_index=True)
    mydfs = scale.fit_transform(result)
    ans = pd.DataFrame(mydfs)
    return ans[0][400],ans[1][400],ans[2][400],ans[3][400],ans[4][400],ans[5][400],ans[6][400]
#     print(myans)
#     return myans


# In[57]:


model.predict([calc(337,118,4,4.5,4.5,9.65,1)])


# In[58]:


# def calc(a,b,c,d,e,f,g):
#     myarr = np.array([a,b,c,d,e,f,g])
#     mydf = pd.Series(myarr, index=["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"])
#     removed_outliers.columns = removed_outliers.columns.astype(str)
#     frames = [removed_outliers,mydf.to_frame().T]
#     result = pd.concat(frames,ignore_index=True)
#     mydfs = scale.fit_transform(result)
#     ans = pd.DataFrame(mydfs)
#     myans = (model.predict([[ans[0][400],ans[1][400],ans[2][400],ans[3][400],ans[4][400],ans[5][400],ans[6][400]]])*100)
#     print(myans)
#     return myans


# In[59]:


calc(337,118,4,4.5,4.5,9.65,1)


# In[60]:


model.predict([[337,118,4,4.5,4.5,9.65,1]])


# In[61]:


model.predict([[337,118,4,4.5,4.5,9.65,1]])


# In[62]:


model.predict([[0.94,0.928571,0.75,0.875,0.8571,0.900735,1]])


# In[64]:


import pickle
pickle.dump(model,open("model.pkl","wb"))


# In[ ]:




