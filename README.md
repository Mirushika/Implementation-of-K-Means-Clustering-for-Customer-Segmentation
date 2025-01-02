# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import necessary libraries. 
2.Use pd.read_csv() to load the Mall_Customers.csv dataset into a DataFrame. 
3.Based on the Elbow Method, choose the optimal number of clusters (e.g., n_clusters=5). 
4.Create a new column "cluster" in the DataFrame and assign the predicted clusters (y_pred) to it. 
```
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: mirushika.t
RegisterNumber: 24901203 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = [] #Within-Cluster Sum of Square
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluste
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="clu
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="clust
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="clu
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="c
plt.legend()
plt.title("Customer Segments") 
```
## Ouput:

```
CustomerID 	Gender 	Age 	Annual Income (k$) 	Spending Score (1-100)
0 	1 	Male 	19 	15 	39
1 	2 	Male 	21 	15 	81
2 	3 	Female 	20 	16 	6
3 	4 	Female 	23 	16 	77
4 	5 	Female 	31 	17 	40  


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Gender                  200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 ```
dtypes: int64(4), object(1)
memory usage: 7.9+ KB


CustomerID                0
Gender                    0
Age                       0
Annual Income (k$)        0
Spending Score (1-100)    0
dtype: int64 ```

Text(0.5, 1.0, 'Elbow Method')
```
![image](https://github.com/user-attachments/assets/7733db9d-2723-4ad2-88f9-dc1bcb58f424)
```
  KMeans 
KMeans(n_clusters=5)

array([4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
       4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1,
       4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2,
       0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
       0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
       0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
       0, 2])
```
![image](https://github.com/user-attachments/assets/b742ad8e-d862-47c0-ba1e-e9706607d9e1) 




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
