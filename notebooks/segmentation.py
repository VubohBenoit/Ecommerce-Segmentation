# 1️⃣ Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# 2️⃣ Create output folders if they don't exist
os.makedirs("outputs/figures", exist_ok=True)

# 3️⃣ Load dataset
df = pd.read_csv("/Users/benoitvuboh/Desktop/clients segmentation/data/data.csv", encoding='ISO-8859-1')

# 4️⃣ Data cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['CustomerID'], inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# 5️⃣ Exploratory Analysis
# Sales by country
sales_country = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=sales_country.index, y=sales_country.values)
plt.xticks(rotation=90)
plt.title("Ventes par pays")
plt.tight_layout()
plt.savefig("outputs/figures/sales_by_country.png")
plt.show()

# 6️⃣ RFM Segmentation
import datetime as dt
NOW = dt.datetime(2025,10,5)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (NOW - x.max()).days,
    'InvoiceNo': 'count',
    'TotalAmount': 'sum'
})
rfm.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'TotalAmount':'Monetary'}, inplace=True)

# Standardize for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Plot segments
plt.figure(figsize=(8,6))
sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Segment'], palette='Set2')
plt.title("Segmentation clients (Recency vs Monetary)")
plt.tight_layout()
plt.savefig("outputs/figures/rfm_segments.png")
plt.show()

# 7️⃣ Product Analysis
top_products = df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 produits par chiffre d'affaires")
plt.tight_layout()
plt.savefig("outputs/figures/top_products.png")
plt.show()

# 8️⃣ Predictive model: customer purchase next month
df['Target'] = np.where(df['InvoiceDate'] > (NOW - pd.Timedelta(days=30)), 1, 0)
customer_features = rfm[['Recency','Frequency','Monetary']]
target = df.groupby('CustomerID')['Target'].max()

X_train, X_test, y_train, y_test = train_test_split(customer_features, target, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report :\n", classification_report(y_test, y_pred))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))

# 9️⃣ Save RFM segments
rfm.to_csv("outputs/rfm_segments.csv")
