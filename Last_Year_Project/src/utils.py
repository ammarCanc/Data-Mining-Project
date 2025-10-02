# Data Processing and Analysis
import pandas as pd  
import numpy as np        
from scipy import stats   
from sklearn.metrics import f1_score

import json
import os

# Visualization
import geopandas as gpd
import matplotlib.pyplot as plt 
import seaborn as sns    
from matplotlib.colors import LinearSegmentedColormap       

# Others
import holidays
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import linkage, dendrogram
import joblib


# Color of plots
plot_color = '#568789'
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#568789', '#efb440'])



def plot_missing_values_dendrogram(df, figsize=(10, 6), leaf_rotation=90, leaf_font_size=10, title='Dendrogram of Missing Values'):
    missing_matrix = df.isnull().astype(int)
    
    linkage_matrix = linkage(missing_matrix.T, method='ward', metric='euclidean')
    
    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=missing_matrix.columns,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size
    )
    
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Euclidean Distance')
    
    plt.tight_layout()
    plt.show()
    
    
def plot_numerical_features(df, numerical_features, figsize=(15, 5), plot_color=plot_color):
    fig, axes = plt.subplots(len(numerical_features), 2, figsize=(figsize[0], figsize[1] * len(numerical_features)))
    fig.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)

    for idx, feature in enumerate(numerical_features):
        # Histogram with KDE
        sns.histplot(data=df, x=feature, kde=True, ax=axes[idx, 0], color=plot_color)
        axes[idx, 0].set_title(f'Distribution of {feature}')
        axes[idx, 0].set_xlabel(feature)

        # Box plot
        sns.boxplot(data=df, y=feature, ax=axes[idx, 1], color=plot_color)
        axes[idx, 1].set_title(f'Box Plot of {feature}')

    plt.tight_layout()
    plt.show()
    
    
def plot_correlation_heatmap(df, 
                              figsize=(10, 8), 
                              title='Correlation Heatmap of Numerical Features', 
                              custom_cmap=None, 
                              center=0, 
                              fmt='.2f', 
                              cbar_shrink=0.8):
    
    plt.figure(figsize=figsize)
    
    correlation_matrix = df.corr()
    
    if custom_cmap is None:
        custom_cmap = plt.cm.coolwarm
    
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap=custom_cmap,
                center=center,
                fmt=fmt,
                cbar_kws={"shrink": cbar_shrink})
    
    plt.title(title)
    
    plt.tight_layout()
    plt.show()
    


def plot_categorical_features(categorical_df, max_features_per_plot=6, figsize=(20, 15)):
    categorical_features = [
        'customer_region', 
        'last_promo',
        'payment_method'
    ]

    num_features = len(categorical_features)
    num_plots = (num_features + max_features_per_plot - 1) // max_features_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_features_per_plot
        end_idx = min((plot_idx + 1) * max_features_per_plot, num_features)
        current_features = categorical_features[start_idx:end_idx]

        fig, axes = plt.subplots(
            nrows=(len(current_features) + 2) // 3,
            ncols=3,
            figsize=(20, 5 * ((len(current_features) + 2) // 3))
        )
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            value_counts = categorical_df[feature].value_counts()
            title = feature
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)  
                title = f"{feature} (Top 10 Values)"
            sns.barplot(x=value_counts.index, y=value_counts.values, color="blue", ax=axes[i])  # Adicione uma cor válida
            axes[i].set_title(title)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].set_xlabel(None)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()



def float_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype('Int64')
        
def identify_outliers(dataframe, metric_features, lower_lim, upper_lim):
    outliers = {}
    obvious_outliers = []

    for metric in metric_features:
        if metric not in dataframe.columns:
            continue
        
        if metric not in lower_lim or metric not in upper_lim:
            continue
        
        outliers[metric] = []
        llim = lower_lim[metric]
        ulim = upper_lim[metric]
        
        for i, value in enumerate(dataframe[metric]):
            if pd.isna(value):
                continue
            
            if value < llim or value > ulim:
                outliers[metric].append(value)
        
        print(f"Total outliers in {metric}: {len(outliers[metric])}")

    # Check for observations that are outliers in all features (Obvious Outliers)
    for index, row in dataframe.iterrows():
        is_global_outlier = True
        for metric in metric_features:
            if metric not in dataframe.columns or metric not in lower_lim or metric not in upper_lim:
                is_global_outlier = False
                break
            
            value = row[metric]
            if pd.isna(value):
                is_global_outlier = False
                break
            
            llim = lower_lim[metric]
            ulim = upper_lim[metric]
            
            if llim <= value <= ulim:
                is_global_outlier = False
                break
        
        if is_global_outlier:
            obvious_outliers.append(index)
    print("-----------------------------")
    print(f"Total global outliers: {len(obvious_outliers)}")
    return outliers, obvious_outliers