import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv('temperature.csv')

# 2. Data Preprocessing & Cleaning
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Drop rows where the target 'average_temperature' is missing (we cannot train without target)
df = df.dropna(subset=['average_temperature'])

# Feature Engineering: Extract Year and Month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Define Features (X) and Target (y)
# We use district_name and parameter (min/max temp) as key features
X = df[['year', 'month', 'district_name', 'parameter']]
y = df['average_temperature']

# Outlier Detection (Optional: Remove extreme outliers using IQR)
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
# Filter out outliers
mask = ~((y < (Q1 - 1.5 * IQR)) | (y > (Q3 + 1.5 * IQR)))
X = X[mask]
y = y[mask]

# 3. Pipeline Construction
numerical_features = ['year', 'month']
categorical_features = ['district_name', 'parameter']

# Create preprocessor with Imputation and Scaling/Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 4. Model Training & Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -np.inf
best_name = ""

print("Model Evaluation:")
for name, model in models.items():
    # Create a full pipeline for each model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f" - {name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_name = name

print(f"\nBest Model Selected: {best_name} with R2: {best_score:.4f}")

# 5. Serialization
# Save the best pipeline (includes preprocessor and model)
joblib.dump(best_model, 'temperature_model.pkl')
print("Model saved to 'temperature_model.pkl'")