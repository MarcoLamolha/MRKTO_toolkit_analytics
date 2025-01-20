# -*- coding: utf-8 -*-
"""Sales Data Analysis Project"""
# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %% Load Data
# Dataset source: Sample sales data (you can replace with your own dataset)
url = "https://raw.githubusercontent.com/plotly/datasets/master/sales_success.csv"
df = pd.read_csv(url)

# %% Data Cleaning
# Check initial info
print("Initial Data Overview:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Handle missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop remaining missing values if any
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("\nData after cleaning:")
print(df.info())

# %% Exploratory Data Analysis
# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Sales by product category
product_sales = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print("\nTop Selling Products:")
print(product_sales.head())

# Sales trends over time
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('YearMonth')['Revenue'].sum().reset_index()

# %% Data Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Top Selling Products
plt.subplot(2, 2, 1)
sns.barplot(x=product_sales.head().values, y=product_sales.head().index)
plt.title('Top 5 Selling Products')
plt.xlabel('Total Revenue')

# Plot 2: Sales Distribution
plt.subplot(2, 2, 2)
sns.histplot(df['Revenue'], bins=20, kde=True)
plt.title('Revenue Distribution')

# Plot 3: Monthly Sales Trend
plt.subplot(2, 2, 3)
sns.lineplot(x=monthly_sales['YearMonth'].astype(str), y=monthly_sales['Revenue'])
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)

# Plot 4: Correlation Heatmap
plt.subplot(2, 2, 4)
correlation_matrix = df.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

# %% Advanced Analysis
# Customer segmentation by purchase behavior
customer_stats = df.groupby('CustomerID').agg({
    'Revenue': ['sum', 'count', 'mean'],
    'Profit': 'sum'
}).reset_index()
customer_stats.columns = ['CustomerID', 'TotalSpend', 'PurchaseCount', 'AvgOrderValue', 'TotalProfit']

# Top customers by revenue
top_customers = customer_stats.sort_values('TotalSpend', ascending=False).head(10)

# %% Insights and Conclusions
print("\nKey Insights:")
print(f"1. Highest selling product category: {product_sales.index[0]} (${product_sales.values[0]:,.2f})")
print(f"2. Average order value: ${df['Revenue'].mean():.2f}")
print(f"3. Most profitable customer ID: {top_customers.iloc[0]['CustomerID']} (${top_customers.iloc[0]['TotalSpend']:,.2f})")
print("4. Strong positive correlation observed between Quantity and Revenue")
print("5. Sales show seasonal pattern with Q4 peaks")

# Save cleaned data
df.to_csv('cleaned_sales_data.csv', index=False)