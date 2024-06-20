# Customer Segmentation Data Analysis

## Overview

This project focuses on customer segmentation using data from a retail dataset. The goal is to segment customers based on their purchasing behavior, which can help in targeting marketing strategies more effectively. The segmentation is performed using RFM (Recency, Frequency, Monetary) analysis and K-means clustering.

## Dataset

The dataset used for this project is a retail transaction dataset. The key attributes include:
- `CustomerID`: Unique identifier for each customer
- `InvoiceNo`: Unique identifier for each transaction
- `StockCode`: Unique identifier for each product
- `Quantity`: Number of products purchased
- `UnitPrice`: Price per unit of product
- `InvoiceDate`: Date of the transaction
- `Description`: Description of the product

## Project Steps

1. **Loading the Dataset**
    - Load the dataset using `pandas`.
    - Handle potential encoding issues and erroneous lines.

2. **Data Cleaning**
    - Correct and convert `InvoiceNo` to numeric.
    - Correct and convert `StockCode` to numeric.
    - Handle missing values.
    - Remove duplicate entries.

3. **Feature Engineering**
    - Calculate `Total_Purchase` as the product of `Quantity` and `UnitPrice`.
    - Extract day, month, and year from `InvoiceDate`.
    - Perform outlier analysis and remove outliers.

4. **RFM Analysis**
    - Calculate `Recency` as the number of days since the last purchase.
    - Calculate `Frequency` as the number of unique transactions.
    - Calculate `Monetary` as the total amount spent.

5. **Customer Segmentation using K-means Clustering**
    - Normalize the RFM features.
    - Determine the optimal number of clusters using the Elbow method.
    - Perform K-means clustering.
    - Analyze and visualize the clusters.

6. **Product Insights and Customer Persona Creation**
    - Analyze product descriptions and unit prices.
    - Create customer personas based on RFM clusters.

## Code Execution

### Dependencies

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `sklearn`

### Running the Code

1. **Load and clean the dataset:**

```python
data = pd.read_csv('customer_data.csv', error_bad_lines=False, engine='python', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
```

2. **Data Cleaning:**

```python
# Removing the 'C' prefix from InvoiceNo and converting to numeric
data['InvoiceNo'] = data['InvoiceNo'].apply(lambda x: x.replace('C', ""))
data['InvoiceNo'] = pd.to_numeric(data['InvoiceNo'])

# Cleaning StockCode
data['StockCode'] = data['StockCode'].replace(r'[A-Za-z]$', '', regex=True)
data['StockCode'] = pd.to_numeric(data['StockCode'], errors='coerce').fillna(-1).astype(int)

# Converting InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
```

3. **Handling Duplicates and Missing Values:**

```python
data.dropna(subset=['Quantity', 'UnitPrice'], inplace=True)
data.drop_duplicates(inplace=True)
data = data.dropna(subset=['CustomerID'])
data['CustomerID'] = data['CustomerID'].drop_duplicates()
```

4. **Outlier Analysis and Removal:**

```python
# Calculate IQR for Quantity and UnitPrice
Q1 = data[['Quantity', 'UnitPrice']].quantile(0.25)
Q3 = data[['Quantity', 'UnitPrice']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((data[['Quantity', 'UnitPrice']] < lower_bound) | (data[['Quantity', 'UnitPrice']] > upper_bound)).any(axis=1)
df_no_outliers = data[~outliers]
```

5. **Feature Engineering:**

```python
data['Total_Purchase'] = data['UnitPrice'] * data['Quantity']
data['Day'] = data['InvoiceDate'].dt.day
data['Month'] = data['InvoiceDate'].dt.month
data['Year'] = data['InvoiceDate'].dt.year
```

6. **RFM Analysis:**

```python
# Recency
latest_purchase_date = data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
current_date = datetime.now()
latest_purchase_date['Recency'] = (current_date - latest_purchase_date['InvoiceDate']).dt.days

# Frequency
frequency = data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency.columns = ['CustomerID', 'Frequency']

# Monetary
data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
monetary = data.groupby('CustomerID')['TotalAmount'].sum().reset_index()
monetary.columns = ['CustomerID', 'Monetary']

# Merging RFM data
rfm_df = pd.merge(latest_purchase_date[['CustomerID', 'Recency']], frequency, on='CustomerID')
rfm_df = pd.merge(rfm_df, monetary, on='CustomerID')
```

7. **K-means Clustering:**

```python
# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# Determining the optimal number of clusters using the Elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.show()

# Choosing the optimal k (e.g., 3 based on the Elbow plot)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
```

8. **Cluster Analysis and Visualization:**

```python
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Recency', 'Frequency', 'Monetary'])
print(cluster_centers)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(rfm_df['Frequency'], rfm_df['Monetary'], c=rfm_df['Cluster'], cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers['Frequency'], cluster_centers['Monetary'], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('Customer Segmentation - K-means Clustering')
plt.xlabel('Frequency')
plt.ylabel('Monetary')

legend_labels = [f'Cluster {i}' for i in range(optimal_k)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Clusters', loc='upper right')
plt.show()
```

## Conclusion

This project successfully segments customers based on their purchasing behavior using RFM analysis and K-means clustering. The resulting clusters can be used for targeted marketing strategies, enhancing customer engagement and business performance.
