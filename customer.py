import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customerData = pd.read_csv('Mall_Customers.csv')
# print(customerData.head())
# print(customerData.isnull().sum())

x = customerData.drop(columns=['CustomerID', 'Gender','Age'], axis = 1)
# print(x.head())
# print(x.values)

# Choosing the number of clusters
# * WCSS --> Within-Cluster Sum of Squares

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
# print(wcss)

# Plot an elbow graph
sns.set_theme()
plt.plot(range(1,11), wcss)
plt.title("The Elbow Point Graph")
plt.xlabel('No of Clusters')
plt.ylabel('Wcss')
# plt.show()

# Two elbow points are 3 and 5 and after 5 there is no significant drop so we will  take 5 clusters

# Training the model
model = KMeans(n_clusters=5, init='k-means++', random_state=0)
# Return a label for each datapoint based on their cluster
y = model.fit_predict(x)
# print(y)


x = x.values
# Visualising the clusters
plt.figure(figsize=(5,5))
plt.scatter(x[y ==0,0], x[y==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(x[y ==1,0], x[y==1,1], s=50, c='green', label='Cluster 2')
plt.scatter(x[y ==2,0], x[y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[y ==3,0], x[y==3,1], s=50, c='blue', label='Cluster 4')
plt.scatter(x[y ==4,0], x[y==4,1], s=50, c='violet', label='Cluster 5')


# Plotting the centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.title('Customer Clusters')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Conclusin from scatter plot
# 1) Violet(Cluster 5) are group of customers who have high spending score but have low annual income
# 2) Blue (CLuster 4) are group of customers who have low spending score as well as low Annual income
# 3) Yellow(Cluster 3) are group of customers who have low spending score and high Annual Income
# 4) Green(Cluster 2) are group of customers who have high spending score as well as high Annual income
# 5) red(Cluster 1) are group of customers who have mid spending score and mid Annual income

# Predicting the cluster of a new customer
def predict_customer_cluster(income, spending_score):
    new_data = np.array([[income, spending_score]])
    cluster = model.predict(new_data)[0]

    # Optional: Assign human-readable labels to clusters
    labels = {
        0: "Moderate Income & Spending",
        1: "High Income & High Spending",
        2: "High Income & Low Spending",
        3: "Low Income & Low Spending",
        4: "Low Income & High Spending"
    }
    
    return f"The customer belongs to Cluster {cluster}: {labels[cluster]}"

# Example usage
income_input = int(input("Enter Annual Income (k$)"))
spending_input = int(input("Enter Spending Score(1-100)"))
result = predict_customer_cluster(income_input, spending_input)
print(result)
