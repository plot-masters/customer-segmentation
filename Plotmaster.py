#!/usr/bin/env python3
"""
Apex Project: Team Plotmasters
Customer Segmentation Analysis Script

Team Members:
1. Utkarsh Tripathi
2. Juwaria Qadri 
3. Mohammed Omar 
4. Merin Ann Cherian 

This script performs customer segmentation analysis on the Blinkit Orders dataset.
It includes data loading, validation, cleaning, EDA, feature engineering, and K-Means clustering.
"""

# Library imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ==========================================
# 1. Data Loading & Validation
# ==========================================

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def basic_exploration(df):
    """Print basic structural details of the DataFrame."""
    print("\n--- Basic Data Exploration ---")
    print(f"Shape: {df.shape}")
    print("\nTop 5 Rows:")
    print(df.head())
    print("\nBasic Info:")
    df.info()
    print("\nNull Values:")
    print(df.isnull().sum())
    print(f"\nDuplicated Rows: {df.duplicated().sum()}")

def summarize_df(df):
    """Provide descriptive statistics and check for anomalies."""
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))
    print("\nCategorical Column Summary:")
    print(df.describe(include=['object']))

    if 'order_total' in df.columns:
        invalid_orders = df[df['order_total'] <= 0]
        if not invalid_orders.empty:
            print(f"\nFound {len(invalid_orders)} rows with invalid (<=0) order totals.")

# ==========================================
# 2. Data Cleaning
# ==========================================

def drop_irrelevant_columns(df, irrelevant_cols):
    """Drop columns that are not needed for analysis."""
    cols_to_drop = [col for col in irrelevant_cols if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns: {cols_to_drop}")

def drop_high_missing_value_columns(df, threshold=0.5):
    """Drop columns with missing values exceeding the threshold."""
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() >= threshold]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped high-missing columns: {cols_to_drop}")

def impute_missing_values(df):
    """Impute missing values based on column type."""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].ffill()
    print("Missing values imputed.")

def remove_duplicates(df):
    """Remove duplicate rows."""
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - df.shape[0]} duplicate rows.")

def handle_outliers(df, column='order_total'):
    """Cap outliers using IQR method."""
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound
        print(f"Outliers handled for column: {column}")

def convert_to_datetime(df, columns):
    """Convert columns to datetime objects."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors='coerce')

def calculate_delivery_delay(df):
    """Calculate delivery delay in minutes."""
    if 'actual_delivery_time' in df.columns and 'promised_delivery_time' in df.columns:
        df['delivery_delay_minutes'] = (df['actual_delivery_time'] - 
                                        df['promised_delivery_time']).dt.total_seconds() / 60
        print("Calculated delivery_delay_minutes.")

def convert_categorical_like_columns(df, columns):
    """Convert object columns to category type."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

def clean_data(df):
    """Execute the full data cleaning pipeline."""
    print("\n--- Starting Data Cleaning ---")
    drop_irrelevant_columns(df, ['irrelevant_column']) # Example
    drop_high_missing_value_columns(df, threshold=0.5)
    impute_missing_values(df)
    remove_duplicates(df)
    handle_outliers(df, 'order_total')
    convert_to_datetime(df, ['order_date', 'actual_delivery_time', 'promised_delivery_time'])
    calculate_delivery_delay(df)
    convert_categorical_like_columns(df, ['payment_method', 'delivery_status'])
    
    # Remove outliers for delivery delay (Specific step found in EDA)
    if 'delivery_delay_minutes' in df.columns:
        Q1 = df['delivery_delay_minutes'].quantile(0.25)
        Q3 = df['delivery_delay_minutes'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['delivery_delay_minutes'] >= (Q1 - 1.5 * IQR)) & 
                (df['delivery_delay_minutes'] <= (Q3 + 1.5 * IQR))]
        print("Removed outliers from delivery_delay_minutes.")
        
    print("Data cleaning completed.")
    return df

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================

def perform_eda(df):
    """Perform basic EDA tasks (Printing summaries)."""
    print("\n--- Exploratory Data Analysis ---")
    
    print("\nNumerical Columns Summary:")
    print(df[['order_total', 'delivery_delay_minutes']].describe())
    
    print("\nCorrelation Matrix:")
    print(df[['order_total', 'delivery_delay_minutes']].corr())
    
    print("\nPayment Method Distribution:")
    print(df['payment_method'].value_counts())
    
    print("\nDelivery Status Distribution:")
    print((df['delivery_status'].value_counts(normalize=True) * 100).round(2))

# ==========================================
# 4. Feature Engineering
# ==========================================

def process_customer_data(df):
    """Transform transaction data into customer-centric features."""
    print("\n--- Feature Engineering ---")
    
    # Aggregation
    customer_df = df.groupby('customer_id').agg({
        'order_total': ['mean', 'sum', 'count'],
        'delivery_delay_minutes': 'mean',
        'payment_method': lambda x: x.mode()[0],
        'delivery_status': lambda x: x.mode()[0]
    }).reset_index()

    customer_df.columns = [
        'customer_id', 'avg_order_total', 'total_spent', 'num_orders',
        'avg_delay', 'most_used_payment', 'most_delivery_status'
    ]

    # Keep a copy for final profiling
    customer_df_original = customer_df.copy()

    # One-Hot Encoding
    customer_df = pd.get_dummies(customer_df, columns=['most_used_payment', 'most_delivery_status'], prefix=['payment', 'status'])

    # Numerical Transformations
    scaler = StandardScaler()
    customer_df['avg_order_total'] = scaler.fit_transform(customer_df[['avg_order_total']])

    customer_df['total_spent'] = np.log1p(customer_df['total_spent'])
    customer_df['total_spent'] = scaler.fit_transform(customer_df[['total_spent']])

    scaler_robust = RobustScaler()
    customer_df['avg_delay'] = scaler_robust.fit_transform(customer_df[['avg_delay']])

    scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
    customer_df['num_orders'] = scaler_minmax.fit_transform(customer_df[['num_orders']])
    
    # Convert booleans to int
    bool_cols = customer_df.select_dtypes(include='bool').columns
    customer_df[bool_cols] = customer_df[bool_cols].astype(int)

    print("Feature engineering completed.")
    return customer_df, customer_df_original

# ==========================================
# 5. Cluster Modelling
# ==========================================

def find_optimal_k(X_scaled, max_k=6):
    """Run K-Means for multiple k values and calculate metrics."""
    print("\n--- Finding Optimal K ---")
    results = {'k': [], 'silhouette': [], 'dbi': [], 'ch': []}
    
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, clusters)
        dbi = davies_bouldin_score(X_scaled, clusters)
        ch = calinski_harabasz_score(X_scaled, clusters)
        
        results['k'].append(k)
        results['silhouette'].append(sil)
        results['dbi'].append(dbi)
        results['ch'].append(ch)
        
        print(f"k={k}: Silhouette={sil:.4f}, DBI={dbi:.4f}, CH={ch:.4f}")
    
    return pd.DataFrame(results)

def perform_clustering(X_scaled, customer_df_original, k=2):
    """Execute final clustering with chosen k."""
    print(f"\n--- Performing Final Clustering (k={k}) ---")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters back to original data
    customer_df_original['cluster'] = clusters
    
    # Label Clusters
    label_map = {
        0: "On-Time High-Value Customers",
        1: "Delay-Affected Low-Value Customers"
    }
    customer_df_original['cluster_label'] = customer_df_original['cluster'].map(label_map)
    
    print("Clustering completed.")
    print("Cluster Counts:")
    print(customer_df_original['cluster_label'].value_counts())
    
    return customer_df_original

# ==========================================
# Main Execution
# ==========================================

def main():
    input_path = "./blinkit_orders.csv"
    
    # 1. Load Data
    df = load_data(input_path)
    if df is None:
        return

    # 2. Validation
    basic_exploration(df)
    summarize_df(df)

    # 3. Clean Data
    df_cleaned = clean_data(df)
    
    # 4. EDA
    perform_eda(df_cleaned)

    # 5. Feature Engineering
    customer_df_processed, customer_df_original = process_customer_data(df_cleaned)
    
    # Prepare X for clustering (drop ID)
    X = customer_df_processed.drop(columns=['customer_id'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Find Optimal K
    metrics_df = find_optimal_k(X_scaled)
    # metrics_df.to_csv("clustering_metrics.csv", index=False)

    # 7. Final Clustering (k=2 based on analysis)
    final_df = perform_clustering(X_scaled, customer_df_original, k=2)
    
    # Save Results
    final_df.to_csv("customer_segmentation_results.csv", index=False)
    print("\nResult saved to 'customer_segmentation_results.csv'")

if __name__ == "__main__":
    main()
