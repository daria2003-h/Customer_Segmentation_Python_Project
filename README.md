# Customer Segmentation Project
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

## Data Cleaning

## Feature Engineering

## KMeans Clustering
