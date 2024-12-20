import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("titanic_test.csv")
#print(test.info())

#reading the data
#print(train.head())
#print(train.isnull())

#visualling the data
#sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Sex',data=train)
#sns.countplot(x='Survived',hue='Pclass',data=train)
#sns.countplot(x='SibSp', data=train)
#train['Fare'].hist(bins=35,figsize=(10,4))
#plt.figure(figsize=(10,7))
#sns.boxplot(x='Pclass',y='Age',data=train)


#filling out null values
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else: 
            return 24
        
    else:
        return Age    
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
#sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
#sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
#plt.show()

#now we will drop the cabin column as there are many missing values and we even dont know how to fill them and also there is no use of that column too
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

#theres one null value in embarked we will remove that too
train.dropna(inplace=True)
test.dropna(inplace=True)
#sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
#plt.show()

#now what we do is convert the string columns into 0 and 1 as our model cant take values as string. we need to dummies it using pandas
#pd.get_dummies(train['Sex'])
#now we get two columns as female(0) and male(1). but theres a problem if our model correctly predticts that ok this person is a female then the other column would be already predicited as not male. this issue is known as multi collinarity and hence to fix this we will take either of one columns and predict only one of them
sex = pd.get_dummies(train['Sex'],drop_first=True).astype(int)
embark = pd.get_dummies(train['Embarked'],drop_first=True).astype(int)

sextest = pd.get_dummies(test['Sex'],drop_first=True).astype(int)
embarktest = pd.get_dummies(test['Embarked'],drop_first=True).astype(int)

train = pd.concat([train,sex,embark],axis=1)
test = pd.concat([test,sextest,embarktest],axis=1)
#print(test.head())

train.drop(['Sex','Embarked','Name','Ticket','PassengerId','Survived'],axis=1,inplace=True)
test.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
#print(train.head())

#models and predicition
#X = train.drop('Survived',axis=1)
#y = train['Survived']
X_train = train.drop('Pclass',axis=1)
y_train = train['Pclass']
X_test = test.drop('Pclass',axis=1)
y_test = test['Pclass']

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=101)
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))