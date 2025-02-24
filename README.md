# Customer Segmentation Project
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/Customer_clustering.png)

This customer segmentation project uses KMeans clustering to group customers based on their purchasing behaviors. By analyzing patterns in the data, the model identifies distinct customer segments, helping businesses tailor their marketing strategies and offerings. Segmentation enables more personalized engagement, optimized resource allocation, and improved customer retention. The insights derived from this analysis can lead to better-targeted campaigns, increased customer satisfaction, and overall business growth.
## Project Start
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Setting to make numbers easier to read on display
pd.options.display.float_format = '{:20.2f}'.format

#Show all columns on output
pd.set_option('display.max.columns',999)

# import data
df = pd.read_excel("C:/Users/PC/Desktop/Python/customer_clustering_/online_retail_II.xlsx", sheet_name=0)
```
## Explorative Data Analysis (EDA)
During my exploratory analysis, I examined the data's shape and reviewed the descriptions for both numeric and object data types. I identified several issues, including missing Customer ID values and negative Product Quantities, and concluded that it would be best to exclude this data during the next step of cleaning for further analysis. Additionally, I used regular expressions (regex) to inspect the Invoice and StockCode columns. I found discrepancies in both: the Invoice numbers, which are expected to be six-digit values, and the StockCode numbers, which should consist of five digits, showed deviations from the expected format.
```python
df.head() # taking look at the data
df.info() # checking number of columns, data types, non-null counts, memory usage
df.describe() # describes only numeric data
df.describe( include = 'O') # describes object data as well

# taking a look at data with missing customer id
df[df['Customer ID'].isna()].head(10)

# taking a look at data with negative quantity
df[df['Quantity']<0].head(10)  # C at the start of the Invoice Number means Cancellation

# pattern for Invoice - 6 digits
df['Invoice'] = df['Invoice'].astype('str') # converting into string to be able to apply regex 
df[df['Invoice'].str.match('^\\d{6}$')== False] # what data does not match this pattern?

# is C the only character that appears?
df['Invoice'].str.replace('[0-9]', '', regex = True).unique() # output : array(['', 'C', 'A'], dtype=object)
# only 3 records in our data set column Invoice start with 'A', which are 'Adjust bad debt' and should
be excluded later 
df[df['Invoice'].str.startswith('A')]

#exploring StockCode
df['StockCode'] = df['StockCode'].astype('str')
df[df['StockCode'].str.match('^\\d{5}$') == True].head(10)
df[df['StockCode'].str.match('^\\d{5}$') == False] # checking for records that does not match the pattern of 5 digits

df[(df['StockCode'].str.match('^\\d{5}$') == False) 
   & (df['StockCode'].str.match('^\\d{5}[a-zA-Z]+$') == False)]['StockCode'].unique() # neither pattern of 5 digits nor 5 digits and letters
'''
 output: array(['POST', 'D', 'DCGS0058', 'DCGS0068', 'DOT', 'M', 'DCGS0004',
       'DCGS0076', 'C2', 'BANK CHARGES', 'DCGS0003', 'TEST001',
       'gift_0001_80', 'DCGS0072', 'gift_0001_20', 'DCGS0044', 'TEST002',
       'gift_0001_10', 'gift_0001_50', 'DCGS0066N', 'gift_0001_30',
       'PADS', 'ADJUST', 'gift_0001_40', 'gift_0001_60', 'gift_0001_70',
       'gift_0001_90', 'DCGSSGIRL', 'DCGS0006', 'DCGS0016', 'DCGS0027',
       'DCGS0036', 'DCGS0039', 'DCGS0060', 'DCGS0056', 'DCGS0059', 'GIFT',
       'DCGSLBOY', 'm', 'DCGS0053', 'DCGS0062', 'DCGS0037', 'DCGSSBOY',
       'DCGSLGIRL', 'S', 'DCGS0069', 'DCGS0070', 'DCGS0075', 'B',
       'DCGS0041', 'ADJUST2', '47503J ', 'C3', 'SP1002', 'AMAZONFEE'],
      dtype=object)
'''
# StockCode is meant to follow the pattern [0-9]{5} but seems to have legit values for [0-9]{5}[a-zA-Z]+
# Records that deviate from these patterns are likely illegitimate transactions.
# Creating the DataFrame
data = {
    "Code": ["DCGS", "D", "DOT", "M or m", "C2", "C3", "BANK CHARGES or B", "S", "TESTXXX", "gift__XXX", "PADS", "SP1002", "AMAZONFEE", "ADJUSTX"],
    "Description": [
        "Looks valid, but some quantities are negative, and customer ID is null.",
        "Represents discount values.",
        "Represents postage charges.",
        "Represents manual transactions.",
        "Carriage transaction; meaning is unclear.",
        "Unclear meaning, with only one recorded transaction.",
        "Represents bank charges.",
        "Indicates samples sent to customers.",
        "Testing data, not valid for analysis.",
        "Purchases with gift cards; lacks customer data but may be useful for another analysis.",
        "Appears to be a legitimate stock code for padding.",
        "Special request item with two transactions; three appear legitimate, but one has a price of zero.",
        "Likely fees related to Amazon shipping or services.",
        "Manual account adjustments made by administrators."
    ],
    "Action": [
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude",
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude from clustering",
        "Exclude",
        "Include",
        "Exclude for now",
        "Exclude for now",
        "Exclude for now"
    ]
}

df_1 = pd.DataFrame(data)

# Display the DataFrame
df_1


```


## Data Cleaning
In this step, I cleaned and prepared customer transaction data for analysis. First, I created a new feature, **SalesLineTotal**, by multiplying `Quantity` and `Price`. Then, I aggregated the data by **Customer ID**, calculating total spending (**Monetary Value**), number of purchases (**Frequency**), and the time since the last purchase (**Recency**).  

Next, I visualized these features using **histograms** and **boxplots**, which revealed the presence of outliers. I identified and removed these outliers using the **Interquartile Range (IQR) method**, creating a cleaned dataset. After replotting boxplots to confirm better distributions, I visualized customer data with a **3D scatter plot**.

Finally, I **standardized** the data using **StandardScaler** to ensure consistency in further analysis and plotted another 3D scatter plot of the scaled features. Now, the dataset is well-structured and ready for customer segmentation🚀
```python
cleaned_df = df.copy()
cleaned_df.columns

#Cleaning Invoice column
cleaned_df['Invoice'] = cleaned_df['Invoice'].astype('str')
mask = (
    cleaned_df['Invoice'].str.match('^\\d{6}$') == True
)
cleaned_df = cleaned_df[mask]
cleaned_df.head()

# Cleaning StockCode column
cleaned_df['StockCode'] = cleaned_df['StockCode'].astype('str')

mask_1 = (
    (cleaned_df['StockCode'].str.match('^\\d{5}$')== True)
    |(cleaned_df['StockCode'].str.match('^\\d{5}[a-zA-Z]+$')== True)
    |(cleaned_df['StockCode'].str.match('^PADS$')== True)
)

cleaned_df = cleaned_df[mask_1]
cleaned_df

# dropping records with missing Customer ID
cleaned_df.dropna(subset = ['Customer ID'], inplace = True)
cleaned_df.describe() # checking statistics for a data set 

len(cleaned_df [cleaned_df['Price'] == 0])
cleaned_df= cleaned_df[cleaned_df['Price']>0]

len(cleaned_df)/len(df) # 23% of the data is lost due to cleaning
```







## Feature Engineering

```python
cleaned_df['SalesLineTotal'] = cleaned_df['Quantity']*cleaned_df['Price']
cleaned_df

aggregated_df = cleaned_df.groupby( by = 'Customer ID', as_index=False) \
.agg(
    MonetaryValue = ('SalesLineTotal', 'sum'),
    Frequency = ('Invoice', 'nunique'),
    LastInvoiceDate = ('InvoiceDate', 'max')
)
aggregated_df.head()
```

```python
max_invoice_date = aggregated_df['LastInvoiceDate'].max()
max_invoice_date

aggregated_df['Recency'] = (max_invoice_date- aggregated_df['LastInvoiceDate']).dt.days
aggregated_df
```
```python
# visualizing new features
plt.figure(figsize =(15, 5))

plt.subplot(1, 3, 1)
plt.hist(aggregated_df['MonetaryValue'], bins = 10, color = 'skyblue', edgecolor = 'black')
plt.title('Monetary Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(aggregated_df['Frequency'], bins = 10, color = 'lightgreen', edgecolor = 'black')
plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
plt.hist(aggregated_df['Recency'], bins = 10, color = 'salmon', edgecolor = 'black')
plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/images/feature%20vis.png)

```
# making boxplots
plt.figure(figsize =(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data = aggregated_df['MonetaryValue'],  color = 'skyblue')
plt.title('Monetary Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.boxplot(aggregated_df['Frequency'],  color = 'lightgreen')
plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
sns.boxplot(aggregated_df['Recency'], color = 'salmon')
plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

#  as we can see we have to deal with outliers
# we have to separate outliers for extra analysis
```
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/images/outliers.png)

```python
M_Q1 = aggregated_df['MonetaryValue'].quantile(0.25)
M_Q3 = aggregated_df['MonetaryValue'].quantile(0.75)
M_IQR = M_Q3-M_Q1 # interquartal range

monetary_outliers_df = aggregated_df[
(aggregated_df['MonetaryValue'] > (M_Q3+1.5*M_IQR))
| (aggregated_df['MonetaryValue']< (M_Q1-1.5*M_IQR))].copy()

monetary_outliers_df.describe() # high spenders

F_Q1 = aggregated_df['Frequency'].quantile(0.25)
F_Q3 = aggregated_df['Frequency'].quantile(0.75)
F_IQR = F_Q3-F_Q1 # interquartal range

frequency_outliers_df = aggregated_df[
(aggregated_df['Frequency'] > (F_Q3+1.5*F_IQR))
| (aggregated_df['Frequency']< (F_Q1-1.5*F_IQR))].copy()

frequency_outliers_df.describe() # highly frequent spenders 

non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index)) 
&(~aggregated_df.index.isin(frequency_outliers_df.index))]

non_outliers_df.describe()
```

```python
# reploting boxplots to see if we have solved the priblem with outliers
plt.figure(figsize =(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data = non_outliers_df['MonetaryValue'],  color = 'skyblue')
plt.title('Monetary Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.boxplot(non_outliers_df['Frequency'],  color = 'lightgreen')
plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
sns.boxplot(non_outliers_df['Recency'], color = 'salmon')
plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Count')

plt.tight_layout()
plt.show()  # it looks way better now
 ```
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/images/o1.png)

```python
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection = '3d')
scatter = ax.scatter( non_outliers_df['MonetaryValue'], 
                     non_outliers_df['Frequency'], 
                     non_outliers_df['Recency'])

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data')

plt.tight_layout()
plt.show()
```
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/images/3d1.png)

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(non_outliers_df[['MonetaryValue', 'Frequency', 'Recency']])
scaled_data
scaled_data_df = pd.DataFrame(scaled_data, index = non_outliers_df.index, 
                              columns = ('MonetaryValue', 'Frequency','Recency'))
scaled_data_df
```
```python
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection = '3d')
scatter = ax.scatter( scaled_data_df['MonetaryValue'], 
                     scaled_data_df['Frequency'], 
                     scaled_data_df['Recency'])

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data')

plt.tight_layout()
plt.show()
```
![](https://github.com/daria2003-h/Customer_Segmentation_Python_Project/blob/main/images/3d2.png)


## KMeans Clustering

I performed K-Means clustering to segment customers based on their purchasing behavior. First, I tested different values of k (number of clusters) using inertia and silhouette scores to find the optimal number of clusters. Based on the results, I selected k=4 and assigned each customer to a cluster.

Next, I visualized the clusters using a 3D scatter plot and violin plots to analyze their characteristics. After interpreting the clusters, I assigned meaningful labels.I also identified and analyzed outliers separately.Finally, I combined all clusters and created a summary visualization showing the customer distribution across clusters and the average values for Recency, Frequency, and MonetaryValue per cluster. This provides a clear understanding of customer segments and guides strategic decision-making for customer engagement.
```python
# number of clusters is determined by the number of centroids
max_k = 12

inertia = []
silhouette_scores = []
k_values = range(2, max_k+1)

for k in k_values: 

    kmeans = KMeans(n_clusters= k, random_state= 42, max_iter = 1000) # repeatable
    cluster_labels = kmeans.fit_predict(scaled_data_df)
    sil_score = silhouette_score(scaled_data_df, cluster_labels)
    silhouette_scores.append(sil_score)
    inertia.append(kmeans.inertia_)
```
```python
plt.figure(figsize = (14,6))

plt.subplot(1, 2, 1)
plt.plot(k_values,inertia, marker = '*')
plt.title('KMeans Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values,silhouette_scores, marker = '*', color = 'orange')
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.show()
```

```python
kmeans = KMeans( n_clusters=4, random_state =42, max_iter = 1000)
cluster_labels = kmeans.fit_predict(scaled_data_df)
cluster_labels # different clusters we got

non_outliers_df['Cluster']= cluster_labels
non_outliers_df

#Visualizing clusters 

cluster_colors = {
    0: '#1f77b4', #blue
    1: '#ff7f0e' ,#orange
    2: '#2ca02c', #green 
    3: '#d62728'  #red
}

colors = non_outliers_df['Cluster'].map(cluster_colors)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(projection = '3d')

scatter= ax.scatter( non_outliers_df['MonetaryValue'],
                    non_outliers_df['Frequency'],
                    non_outliers_df['Recency'],
                    c = colors,
                    marker='*')
ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data by Cluster')
plt.show()
```

```python
# Creating Violin Plots

plt.figure(figsize = (12,18))

plt.subplot(3,1,1)
sns.violinplot( x = non_outliers_df['Cluster'], y = non_outliers_df['MonetaryValue'], palette = cluster_colors, hue = non_outliers_df['Cluster'])
sns.violinplot(y = non_outliers_df['MonetaryValue'], color = 'gray', linewidth = 1.0)
plt.title('Monetary value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3,1,2)
sns.violinplot( x = non_outliers_df['Cluster'], y = non_outliers_df['Frequency'], palette = cluster_colors, hue = non_outliers_df['Cluster'])
sns.violinplot(y = non_outliers_df['Frequency'], color = 'gray', linewidth = 1.0)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')

plt.subplot(3,1,3)
sns.violinplot( x = non_outliers_df['Cluster'], y = non_outliers_df['Recency'], palette = cluster_colors,hue = non_outliers_df['Cluster'])
sns.violinplot(y = non_outliers_df['Recency'], color = 'gray', linewidth = 1.0)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```

### interpreting what the clusters represent and giving them meaningfull names

#### 1. Cluster 0 (Blue): 'Retain'
* Rationale : This cluster represents high-value customners who purchase regularly , though not
always very recently. The focus should be on retention efforts to maintain their loyalty and
spending levels
* Action: Implement loyalty programs, personalized offers and regular engagement to ensure they remain active

#### 2.Cluster 1 (Orange) : 'Re-Engage'
* Rationale: This group includes lower-value, infrequent buyers, who have nor purchased recently.
The focus should be on re-engagement to bring them back into active purchasing behavior.
* Action: Use targeted marketing campaigns , special discounts, or reminders to encourage them to return and purchase again

#### 3. Cluster 2 (Green): 'Nurture'
* Rationale : This cluster represents the least active and lowest-value customers, but they have made recent purchases. These customers may be new or need nurturing to increase their engagement and spending
* Action : Focus on building relationships, providing excellent customer service, and offering incentives to encourage more frequent purchases.

#### 4. Cluster 3 (Red): 'Reward'
* Rationale: This cluster includes high-value, very frequent buyers, many of whom are still actively purchasing. They are most loyal customers and rewarding their loyalty is key to maintaining their engagement 
* Action: Implement a robust loyalty program, provide exclusive offers and recognize their loyalty to keep them engaged and satisfied.


```python
monetary_outliers_df
frequency_outliers_df # overlap is possible--3 manual clusters

overlap_indices = monetary_outliers_df.index.intersection(frequency_outliers_df.index)
monetary_only_outliers = monetary_outliers_df.drop(overlap_indices)
frequency_outliers_only = frequency_outliers_df.drop(overlap_indices)
monetary_and_frequency_outliers = monetary_outliers_df.loc[overlap_indices]

monetary_only_outliers['Cluster'] = -1
frequency_outliers_only['Cluster'] = -2
monetary_and_frequency_outliers['Cluster'] = -3

outlier_clusters_df = pd.concat([monetary_only_outliers, frequency_outliers_only, monetary_and_frequency_outliers])
outlier_clusters_df
```
```python
# Creating Violin Plots
cluster_colors_2 = {
    -1: '#9467bd',
    -2: '#8c564b',
    -3: '#e377c2'
}

plt.figure(figsize = (12,18))

plt.subplot(3,1,1)
sns.violinplot( x = outlier_clusters_df['Cluster'], y = outlier_clusters_df['MonetaryValue'], palette = cluster_colors_2, hue = outlier_clusters_df['Cluster'])
sns.violinplot(y = outlier_clusters_df['MonetaryValue'], color = 'gray', linewidth = 1.0)
plt.title('Monetary value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3,1,2)
sns.violinplot( x = outlier_clusters_df['Cluster'], y = outlier_clusters_df['Frequency'], palette = cluster_colors_2, hue = outlier_clusters_df['Cluster'])
sns.violinplot(y = outlier_clusters_df['Frequency'], color = 'gray', linewidth = 1.0)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')

plt.subplot(3,1,3)
sns.violinplot( x = outlier_clusters_df['Cluster'], y = outlier_clusters_df['Recency'], palette = cluster_colors_2,hue = outlier_clusters_df['Cluster'])
sns.violinplot(y = outlier_clusters_df['Recency'], color = 'gray', linewidth = 1.0)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```
### Interpretation
1. Cluster -1 ( Monetary Outliers) PAMPER:
Characteristics : High spenders but not neccesserilyl frequent buyers. Their purchases are large but infrequent. 
Potentiol Strategy: Focus on maintaining their loyalty wit personalized offers or luxury services that cater to their high spending capacity

2. Cluster -2 ( Frequency Outliers) UPSELL:
Characteristics: Freduwnt buyers who spend less per purchase. These customers are consistently engaged but might benefit from upselling opportunities. 
Potential Strategy: Implement loyalty programs or bundle deals to encourage higher spending per visit, given their frequent engagement

3. Cluster -3 ( Monetary & Frequency Outliers) DELIGHT:
Characteristics: The most valuable outliers, with extreme spending and frequent purchases. They are likely your top-tier customers who require special attention.
Potential Strategy: Develop VIP programs or exclusive offeres to maintain their loyalty and encourage continued engagement.

```python
cluster_labels = {
    0: 'RETAIN',
    1: 'RE-ENGAGE',
    2: 'NURTURE',
    3: 'REWARD',
    -1: 'PAMPER',
    -2: 'UPSELL',
    -3: 'DELIGHT'
}

full_clustering_df = pd.concat([non_outliers_df, outlier_clusters_df])
full_clustering_df['ClusterLabel'] = full_clustering_df['Cluster'].map(cluster_labels)
full_clustering_df
```

```python
#Summarising all our findings in a single Visual

# Count of customers per cluster
cluster_counts = full_clustering_df['ClusterLabel'].value_counts()

# Mean of features per cluster
feature_means = full_clustering_df.groupby('ClusterLabel')[['Recency', 'Frequency', 'MonetaryValue']].mean()

fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot: Cluster distribution
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1, palette='viridis')
ax1.set_ylabel('Number of Customers', color='b')
ax1.set_title('Cluster Distribution with Average Feature Values')

# Second y-axis for line plot
ax2 = ax1.twinx()

# Line plot: Average Recency, Frequency, and MonetaryValue per cluster
sns.lineplot(data=feature_means, ax=ax2, marker='o', dashes=False)
ax2.set_ylabel('Average Value', color='g')

plt.show()
```

```python
# making scaling of line plot better to read

# Count of customers per cluster
cluster_counts = full_clustering_df['ClusterLabel'].value_counts()
full_clustering_df['MonetaryValue per 100 pounds'] = full_clustering_df['MonetaryValue']/100

# Mean of features per cluster
feature_means = full_clustering_df.groupby('ClusterLabel')[['Recency', 'Frequency', 'MonetaryValue per 100 pounds']].mean()

fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot: Cluster distribution
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1, palette='viridis')
ax1.set_ylabel('Number of Customers', color='b')
ax1.set_title('Cluster Distribution with Average Feature Values')

# Second y-axis for line plot
ax2 = ax1.twinx()

# Line plot: Average Recency, Frequency, and MonetaryValue per cluster
sns.lineplot(data=feature_means, ax=ax2, marker='o', dashes=False)
ax2.set_ylabel('Average Value', color='g')

plt.show()
```



