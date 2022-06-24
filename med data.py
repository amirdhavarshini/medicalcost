#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[7]:


data = pd.read_csv(r"C:\Users\Amirdha Selvaraj\Downloads\insurance.csv")


# In[8]:


data.info()


# In[9]:


data.shape


# # charges of each attribute
# sex, children, smoker, region   - charges
# 

# In[11]:


data[["sex","charges"]].groupby(["sex"], as_index = False).mean().sort_values(by = "charges",ascending = False).style.background_gradient("Blues")


# In[12]:


data[["children","charges"]].groupby(["children"], as_index = False).mean().sort_values(by = "charges",ascending = False).style.background_gradient("Greys")


# In[13]:


data[["smoker","charges"]].groupby(["smoker"], as_index = False).mean().sort_values(by = "charges",ascending = False).style.background_gradient("Greens")


# In[18]:


data[["region","charges"]].groupby(["region"], as_index = False).mean().sort_values(by = "charges",ascending = False).style.background_gradient("Accent")


# sex, children, smoker, region - other features

# In[22]:


region = data.groupby("region", as_index=False)["age","bmi","children","charges"].mean().sort_values("age",ascending=False).style.background_gradient("rainbow")
print("Average value of other properties by region \n")
region


# In[23]:


sex = data.groupby("sex", as_index=False)["age","bmi","children","charges"].mean().sort_values("age",ascending=False).style.background_gradient("prism")
print("Average value of other properties by sex \n")
sex


# In[25]:


smoker = data.groupby("smoker", as_index=False)["age","bmi","children","charges"].mean().sort_values("age",ascending=False).style.background_gradient("jet")
print("Average value of other properties by smoker \n")
smoker


# In[27]:


children = data.groupby("children", as_index=False)["age","bmi","charges"].mean().sort_values("age",ascending=False).style.background_gradient("Spectral")
print("Average value of other properties by children \n")
children


# # missing values

# In[28]:


data.isnull().sum()


# # data visualization

# In[29]:


regions = ["southeast","northwest","southwest","northeast"]
children = [0,1,2,3,4,5]
genders = ["female", "male"]

regionAgeMean = []
sexAgeMean = []
childAgeMean = []


# In[30]:


for region in regions:
    x = data[data["region"] == region]
    ageMeanRegion = x["age"].mean()
    regionAgeMean.append(ageMeanRegion)
    


# In[31]:


for gender in genders:
    y = data[data["sex"] == gender]
    ageMeanSex = y["age"].mean()
    sexAgeMean.append(ageMeanSex)


# In[32]:


for child in children:
    z = data[data["children"] == child]
    ageMeanChild = z["age"].mean()
    childAgeMean.append(ageMeanChild)


# In[33]:


ageFirstDecember = data[data["age"].between(18,28, inclusive = True)]["charges"].mean()
ageSecondDecember = data[data["age"].between(29,39, inclusive = True)]["charges"].mean()
ageThirdDecember = data[data["age"].between(40,50, inclusive = True)]["charges"].mean()
ageFourthDecember = data[data["age"].between(51,64, inclusive = True)]["charges"].mean()

averageAge = [ageFirstDecember,ageSecondDecember,ageThirdDecember,ageFourthDecember]
ageRanges = ["18-28 Age","29-39 Age","40-50 Age","51-64 Age"]


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


plt.title("Average age by region", color = "purple")
plt.xlabel("regions")
plt.ylabel("age mean")
plt.subplot(3,2,2)
sns.barplot(x = genders, y = sexAgeMean, palette="binary")


# In[39]:


plt.title("Average age by sex", color = "orange")
plt.xlabel("genders")
plt.ylabel("age mean")
plt.subplot(3,2,3)
sns.barplot(x = children, y = childAgeMean, palette="crest")


# In[40]:


plt.title("Average age by children", color = "blue")
plt.xlabel("number of children")
plt.ylabel("age mean")
plt.subplot(3,2,4)
sns.barplot(x=ageRanges, y=averageAge, palette="Blues")


# In[42]:


from scipy.stats import norm, boxcox
from scipy import stats
plt.title("Charges by age range", color = "darkgreen")
plt.xlabel("age range")
plt.ylabel("Charges")
plt.subplot(3,2,(5,6))
sns.distplot(data["age"], fit=norm)
plt.title("Age Distplot", color = "darkred")

plt.show()


# In[48]:


bodyMassIndex = []
for i in genders:
    sex = data[data["sex"] == i]
    bmi = sex["bmi"].mean()
    bodyMassIndex.append(bmi)
    
totalNumber = data.sex.value_counts().values
genderLabel = data.sex.value_counts().index
circle = plt.Circle((0,0),0.2,color = "white") 
explode = (0, 0.1)

plt.figure(figsize=(8,10))
plt.subplot(2,2,1)
sns.barplot(x = genders, y = bodyMassIndex, palette= "rocket")

plt.title("body mass index by gender", color = "purple")
plt.xlabel("gender")
plt.ylabel("bmi")
plt.subplot(2,2,2)

plt.pie(totalNumber, labels = genderLabel,autopct='%1.2f%%', explode = explode, colors=['blue','lightpink'])
p = plt.gcf()
p.gca().add_artist(circle) 
plt.title("female/male")
plt.legend()
plt.subplot(2,2,(3,4))

sns.countplot(x = 'sex', hue = 'smoker', data = data, palette="twilight")
plt.title("smoking status by gender", color = "darkgreen")
plt.xlabel("gender")
plt.show()


# In[49]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)

plt.hist(data["bmi"], color="orange")
plt.xlabel("bmi")
plt.ylabel("Frequency")

plt.title("bmi histogram", color = "darkred")
plt.subplot(1,2,2)
sns.distplot(data["bmi"], fit=norm)
plt.title("bmi Distplot", color = "darkred")

plt.show()


# In[60]:


childNumber = []
childCharges = []

for each in children:
    child = data[data["children"] == each]
    xx = child["age"].mean()
    yy = child["charges"].mean()
    childNumber.append(xx)
    childCharges.append(yy)
    
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

sns.barplot(x = children, y = childNumber, palette= "icefire")

plt.title("Average age by child number", color = "pink")
plt.xlabel("child number")
plt.ylabel("age mean")
plt.subplot(1,2,2)
sns.barplot(x = children, y = childCharges, palette="plasma")

plt.title("Average charges by child number", color = "orange")
plt.xlabel("child number")
plt.ylabel("charges mean")

plt.show()


# In[61]:


smoker = ["yes", "no"]

smokerAge = []
smokerCharges = []

for each in smoker:
    smokerDistinction = data[data["smoker"] == each]
    xxx = smokerDistinction["age"].mean()
    yyy = smokerDistinction["charges"].mean()
    smokerAge.append(xxx)
    smokerCharges.append(yyy)

    
plt.figure(figsize=(11,10))
plt.subplot(2,2,1)
sns.barplot(x = smoker, y = smokerAge, palette= "Greens")
plt.title("average age by smoking status", color = "darkgreen")
plt.xlabel("smoker")
plt.ylabel("age mean")
plt.subplot(2,2,2)
sns.barplot(x = smoker, y = smokerCharges, palette="crest")
plt.title("Average charges by child number", color = "blue")


# In[62]:


regionCharges = []
regionBmi = []

for each in regions:
    regionn = data[data["region"] == each]
    bmiRegion = regionn["bmi"].mean()
    chargesRegion = regionn["charges"].mean()
    regionCharges.append(chargesRegion)
    regionBmi.append(bmiRegion)
    
plt.figure(figsize=(11,10))
plt.subplot(2,2,1)
sns.barplot(x = regions, y = regionCharges, palette= "Greens")
plt.title("charges by region", color = "darkgreen")
plt.xlabel("regions")
plt.ylabel("charges mean")
plt.subplot(2,2,2)
sns.barplot(x = regions, y = regionBmi, palette="crest")
plt.title("bmi by region", color = "blue")
plt.xlabel("regions")
plt.ylabel("bmi mean")

plt.show()


# In[65]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(data["charges"], color="blue")
plt.xlabel("charges")
plt.ylabel("Frequency")
plt.title("charges histogram", color = "darkred")
plt.subplot(1,2,2)
sns.distplot(data["charges"], fit=norm)
plt.title("charges Distplot", color = "darkred")

plt.show()


# changing 'male' and 'female' as 1 and 0

# In[64]:


data["sex"] = [0 if i == "female" else 1 for i in data["sex"]]
data["sex"] = data["sex"].astype("category")
data = pd.get_dummies(data, columns= ["sex"])
data.head()


# In[66]:


data["children"] = data["children"].astype("category")
data = pd.get_dummies(data, columns= ["children"])
data.head()


# In[67]:


data["smoker"] = [0 if i == "no" else 1 for i in data["smoker"]]


# In[68]:


data["smoker"] = data["smoker"].astype("category")
data = pd.get_dummies(data, columns= ["smoker"])
data.head()


# In[69]:


data["region"] = [0 if i == "southeast" else 1 if i == "southwest" else 2 if i == "northwest" else 3 for i in data["region"]]


# In[70]:


data["region"] = data["region"].astype("category")
data = pd.get_dummies(data, columns= ["region"])
data.head()


# In[71]:


(mu, sigma) = norm.fit(data["charges"])
print("mu {} : {}, sigma {} : {}".format("charges", mu, "charges", sigma))

data["charges"] = np.log1p(data["charges"])

(mu, sigma) = norm.fit(data["charges"])
print("mu {} : {}, sigma {} : {}".format("charges", mu, "charges", sigma))


# # Modelling of Data

# In[88]:


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


# In[89]:


y = data.charges
X = data.drop(["charges"], axis = 1)


# In[90]:


test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = 20)


# In[91]:


from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
result = []
multiple_linear = LinearRegression()
multiple_linear.fit(X_train, Y_train)
predict = multiple_linear.predict(X_test)
score = r2_score(Y_test,predict)
result.append(score)
print('Mean Absolute Error -->', metrics.mean_absolute_error(Y_test, predict))


# In[104]:


df_linearRegression = pd.DataFrame({'Actual': Y_test, 'Predicted': predict})
df_linearRegression.head()


# In[105]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, Y_train)
knn_predict = knn.predict(X_test)
score_knn = r2_score(Y_test,knn_predict)
result.append(score_knn)
print("r_square score --> ",score_knn)
print('Mean Absolute Error -->', metrics.mean_absolute_error(Y_test, knn_predict))


# In[106]:


df_KNNRegressor = pd.DataFrame({'Actual': Y_test, 'Predicted': knn_predict})
df_KNNRegressor.head()


# In[108]:


df_result = pd.DataFrame({"Score":result, "ML Models":["LinearRegression","KNN Regression"]})
df_result


# In[109]:


g = sns.barplot("Score", "ML Models", data = df_result)
g.set_xlabel("Score")
g.set_title("Regression Model Results", color = "violet")
plt.show()


# In[ ]:




