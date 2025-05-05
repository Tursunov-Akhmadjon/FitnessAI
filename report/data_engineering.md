# Model Report

To prepare the data for modeling, we applied the following preprocessing steps:

# 1. Missing Value Handling

Numerical columns: Missing values were filled with the mean of each column.

Categorical columns: Missing values were filled with the most frequent (mode) value.

# 2. Feature Scaling

All numerical features were standardized using **StandardScaler** for better performance.

# 3. Encoding

All categorical features were converted into numerical format using **OneHotEncoder**.

**handle_unknown='ignore'** ensures that any category not seen during training doesn't crash the model during prediction.

# 4. ColumnTransformer

A ColumnTransformer was used to apply the correct pipeline:

Numerical pipeline to numeric columns.

Categorical pipeline to object (text) columns.
