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
# only 3 records in our data set column Invoice start with 'A', which are 'Adjust bad debt' and should be excluded later 
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

## KMeans Clustering
