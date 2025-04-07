# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.

2.Read the csv file.

3.Make necessary preprocessing.

4.Convert all the object data type to categorical data.

5.Now convert the categorial data to numbers representation using cat.codes

6.Define the feature and target variable.

7.Split the data as training and testing data.

8.Train the model using LogisticRegression().

9.Predict and test the accuracy of the model.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ROSHAN G
RegisterNumber: 212223040176
*/
```
```
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  
  data=pd.read_csv("/content/Placement_Data.csv")
  data.info()
  
  data=data.drop('sl_no',axis=1)
  data
  
  data["gender"]=data["gender"].astype('category')
  data["ssc_b"]=data["ssc_b"].astype('category')
  data["hsc_b"]=data["hsc_b"].astype('category')
  data["hsc_s"]=data["hsc_s"].astype('category')
  data["degree_t"]=data["degree_t"].astype('category')
  data["workex"]=data["workex"].astype('category')
  data["specialisation"]=data["specialisation"].astype('category')
  data["status"]=data["status"].astype('category')
  data.dtypes
  
  data["gender"]=data["gender"].cat.codes
  data["ssc_b"]=data["ssc_b"].cat.codes
  data["hsc_b"]=data["hsc_b"].cat.codes
  data["hsc_s"]=data["hsc_s"].cat.codes
  data["degree_t"]=data["degree_t"].cat.codes
  data["workex"]=data["workex"].cat.codes
  data["specialisation"]=data["specialisation"].cat.codes
  data["status"]=data["status"].cat.codes
  data
  
  data=data.drop(['salary'],axis=1)
  data
  
  x=data.iloc[:,:-1].values
  y=data.iloc[:,-1]
  
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
  from sklearn.linear_model import LogisticRegression
  model=LogisticRegression(max_iter=1000)
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  y_pred
  
  from sklearn.metrics import accuracy_score
  acc=accuracy_score(y_pred,y_test)
  acc
  
  model.predict([[0,85,0,92,0,2,75,2,1,1,0,0]])
```

## Output:
![image](https://github.com/user-attachments/assets/dae290d9-54a9-4acb-9770-599c3d74a727)
![image](https://github.com/user-attachments/assets/1c541eb4-b004-4fe2-ad31-302c03e5cda2)
![image](https://github.com/user-attachments/assets/eca96f72-fdc3-48e3-8cc2-d5213f97f740)
![image](https://github.com/user-attachments/assets/95fb0a64-3fed-4ad8-96e1-78a792ae9192)
![image](https://github.com/user-attachments/assets/a39ab4f2-f9bc-43b1-9556-6a12f5535282)
![image](https://github.com/user-attachments/assets/d20b2ccd-2304-428a-a715-65a84b9990cf)
![image](https://github.com/user-attachments/assets/e769ca5b-8be3-473f-82b1-5c3c64c17557)   ![image](https://github.com/user-attachments/assets/1847b73e-9ca4-49ea-b179-1fbc31a341f9)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
