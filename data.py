import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv('your_dataset.csv')

numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])


encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('label', LabelEncoder(), ['education_level']) 
    ],
    remainder='passthrough' 
)

data = pd.DataFrame(encoder.fit_transform(data))

scaler = StandardScaler()  
data[numeric_features] = scaler.fit_transform(data[numeric_features])


pca = PCA(n_components=0.95)  
data_pca = pca.fit_transform(data[numeric_features])

data_pca = pd.DataFrame(data_pca, columns=[f'PCA_{i+1}' for i in range(data_pca.shape[1])])
data = pd.concat([data, data_pca], axis=1)

X = data.drop(columns=['target']) 
y = data['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)