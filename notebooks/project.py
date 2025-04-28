#!/usr/bin/env python
# coding: utf-8

# Calling Libraries

# In[89]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_regression

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings('ignore')


# Data Upload

# In[8]:


df1 = pd.read_csv('datasets project\FOOD-DATA-GROUP5.csv')
df2 = pd.read_csv('datasets project\macros-RDAs.csv')
df3 = pd.read_csv('datasets project\macros.csv')
df4 = pd.read_csv(r'datasets project\nutrition_dataset.csv')
df5 = pd.read_csv('datasets project\workout_fitness_tracker_data.csv')


# Data Exploration

# In[9]:


df1.info()


# In[10]:


df2.info()


# In[11]:


df3.info()


# In[12]:


df4.info()


# In[13]:


df5.info()


# In[16]:


df4.head(2)


# In[18]:


df5.head(3)


# In[20]:


df4.info()


# In[21]:


df4.head()


# Data cleaning - converting values into lowercase and standardizing column names for efficiency

# In[22]:


for col in ['Age', 'Height', 'Weight', 'Daily Calorie Target', 'Protein', 'Carbohydrates', 'Fat']:
    df4[col] = pd.to_numeric(df4[col], errors='coerce')

df4['Gender'] = df4['Gender'].str.lower().str.strip()
df4['Activity Level'] = df4['Activity Level'].str.lower().str.strip()
df4['Fitness Goal'] = df4['Fitness Goal'].str.lower().str.strip()
df4['Dietary Preference'] = df4['Dietary Preference'].str.lower().str.strip()


df4.dropna(subset=['Age', 'Height', 'Weight'], inplace=True)


# In[23]:


df4.info()


# In[24]:


df4.head()


# In[25]:


df5['Gender'] = df5['Gender'].str.lower().str.strip()
df5['Workout Intensity'] = df5['Workout Intensity'].str.lower().str.strip()
df5['Mood Before Workout'] = df5['Mood Before Workout'].str.lower().str.strip()
df5['Mood After Workout'] = df5['Mood After Workout'].str.lower().str.strip()


# In[32]:


df5.rename(columns={'Height (cm)': 'Height', 'Weight (kg)': 'Weight'}, inplace=True)


# In[36]:


common_cols = ['Age', 'Gender', 'Weight', 'Height']


# In[37]:


merged_df = pd.concat([df4[common_cols], df5[common_cols]], ignore_index=True)


# In[39]:


merged_df.info()


# In[ ]:


columns_df4 = [
    'Age', 'Gender', 'Height', 'Weight',
    'Activity Level', 'Fitness Goal',
    'Dietary Preference', 'Daily Calorie Target',
    'Protein', 'Carbohydrates', 'Fat'
]
df4_selected = df4[columns_df4]


# In[43]:


df4_selected.columns.to_list()


# In[44]:


columns_df5 = [
    'Age', 'Gender', 'Height', 'Weight',
    'Workout Type', 'Workout Duration (mins)', 'Calories Burned',
    'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)',
    'Workout Intensity', 'Sleep Hours', 'Water Intake (liters)',
    'Daily Calories Intake', 'Resting Heart Rate (bpm)', 'VO2 Max',
    'Body Fat (%)', 'Mood Before Workout', 'Mood After Workout'
]
df5_selected = df5[columns_df5]


# In[46]:


for col in set(df4_selected.columns).symmetric_difference(set(df5_selected.columns)):
  if col not in df4_selected.columns:
    df4_selected[col] = pd.NA
  if col not in df5_selected.columns:
    df5_selected[col] = pd.NA

df4_selected = df4_selected[df5_selected.columns]
merged_df = pd.concat([df4_selected, df5_selected], ignore_index=True)


# In[48]:


merged_df.info()


# In[50]:


print(merged_df.shape)


# In[52]:


merged_df.head()


# In[53]:


cols_to_convert = [
    'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)',
    'Steps Taken', 'Daily Calories Intake', 'Resting Heart Rate (bpm)'
]

for col in cols_to_convert:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')


# In[59]:


merged_df['Caloric Balance'] = merged_df['Daily Calories Intake'] - merged_df['Calories Burned']


# In[62]:


columns_to_keep = [
    'Age', 'Gender', 'Height', 'Weight', 'Workout Type', 
    'Calories Burned', 'Workout Intensity', 'Sleep Hours', 
    'Water Intake (liters)', 'VO2 Max', 'Body Fat (%)', 
    'Mood Before Workout', 'Mood After Workout', 'Caloric Balance'
]


# In[65]:


df = merged_df[columns_to_keep]


# Data Preprocessing

# In[66]:


df.info()


# In[69]:


for col in df.columns:
  if df[col].dtype == 'number':
    df[col].fillna(df[col].mean(), inplace=True)
  else:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[70]:


df.info()


# In[71]:


df.to_csv('ready_df.csv', index=False)


# In[ ]:


x = df.drop(columns=['Water Intake (liters)'])
y = df['Water Intake (liters)']
num_col = df.select_dtypes(include='number').columns
mutual_info = mutual_info_regression(df[num_col], y)
mutual_info_df = pd.DataFrame(mutual_info, index=df[num_col].columns, columns=['Mutual Info'])
mutual_info_df = mutual_info_df.sort_values(by='Mutual Info', ascending=False)
mutual_info_df


# In[79]:


for col in df.select_dtypes(include='number').columns:
  plt.figure(figsize=(6,4))
  sns.histplot(df[col], kde=True)
  plt.title(f"Distribution of {col}")
  plt.xlabel(col)
  plt.ylabel('Frequency')
  plt.show()


# In[80]:


num_col.to_list()


# In[81]:


scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])


# In[90]:


cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
  cardinality = df[col].nunique()
  if cardinality < 5:
    df = pd.get_dummies(df, columns=[col], dtype=int, drop_first=True)
  elif 5 <= cardinality < 50:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
  else:
    freq = df[col].value_counts()
    df[col] = df[col].map(freq)


# In[85]:


df.head()


# In[88]:


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_maps = {col: X[col].value_counts().to_dict() for col in X.columns}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X_copy[col].map(self.freq_maps[col]).fillna(0)
        return X_copy


# In[91]:


# First, split categorical columns based on cardinality
low_cardinality_cols = [col for col in cat_cols if df[col].nunique() < 50]
high_cardinality_cols = [col for col in cat_cols if df[col].nunique() >= 50]

# Pipelines
numerical_features = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

low_card_cat_features = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

high_card_cat_features = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('freq_encoder', FrequencyEncoder())
])

# Final preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_features, num_col),
    ('low_card_cat', low_card_cat_features, low_cardinality_cols),
    ('high_card_cat', high_card_cat_features, high_cardinality_cols)
])

