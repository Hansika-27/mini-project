"""
Food Wastage Predictor — Model Training
Style: follows Model_Selection_in_ML.ipynb exactly.

page.tsx (predict) sends to POST /predict:
    studentsEnrolled, averageAttendance, menusServed,
    leftoverFromPreviousDay, specialEvent, mealType,
    dayOfWeek, menuItems (list of strings)

nonveg_items = count of 'nonveg' entries in menuItems (0, 1, 2 …)

app.py derives the three missing model features at inference:
    People_served  = Students_enrolled × people_per_student  (median ratio)
    Main_nonveg_kg = People_served     × nonveg_kg_per_person (median ratio)
    Main_veg_kg    = People_served     × veg_kg_per_person    (median ratio)

Ratios are saved in model_metadata.json so app.py loads them at startup.

Run:
    python train_model_new.py    — trains and saves model + metadata
"""

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Load Data ──────────────────────────────────────────────────────────────────

df = pd.read_csv('Final_data_1.csv')

df.head()
df.shape
df.isnull().sum()

# ── Feature Engineering ────────────────────────────────────────────────────────

DAY_ORDER = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

df['Day_enc']           = df['Day'].map(DAY_ORDER)
df['Meal_type_enc']     = (df['Meal_type'] == 'Dinner').astype(int)
df['Meal_category_enc'] = (df['Meal_category'] == 'Mixed').astype(int)
df['Special_event_enc'] = (df['Special_event'] == 'Yes').astype(int)

# Nonveg_items in the CSV is already a count (0, 1, 2 …)
# Matches page.tsx which counts 'nonveg' selections from menuItems — no change needed.

# ── Compute per-person ratios ──────────────────────────────────────────────────
# Saved to model_metadata.json so app.py can derive People_served,
# Main_nonveg_kg, Main_veg_kg at inference time (page.tsx does not send them).

df['people_per_student']   = df['People_served']  / df['Students_enrolled'].replace(0, np.nan)
df['nonveg_kg_per_person'] = df['Main_nonveg_kg'] / df['People_served'].replace(0, np.nan)
df['veg_kg_per_person']    = df['Main_veg_kg']    / df['People_served'].replace(0, np.nan)

PEOPLE_RATIO    = df['people_per_student'].median()
NONVEG_KG_RATIO = df['nonveg_kg_per_person'].median()
VEG_KG_RATIO    = df['veg_kg_per_person'].median()

# ── Features & Target ─────────────────────────────────────────────────────────

features = [
    'Students_enrolled',        # from studentsEnrolled
    'Attendance_percent',       # from averageAttendance
    'Special_event_enc',        # specialEvent yes/no → 1/0
    'Menu_count',               # from menusServed
    'Previous_day_leftover_kg', # from leftoverFromPreviousDay
    'Nonveg_items',             # count of 'nonveg' in menuItems
    'Meal_type_enc',            # dinner/snacks → 1, else 0
    'Day_enc',                  # Monday→0 … Sunday→6
    'Main_nonveg_kg',           # derived: people × nonveg_kg_per_person
    'Main_veg_kg',              # derived: people × veg_kg_per_person
    'People_served',            # derived: students × people_per_student
]

x = df[features]
y = df['Total_Waste_kg']

print(x)
print(y)

x = np.asarray(x)
y = np.asarray(y)

# ── Train / Test Split ─────────────────────────────────────────────────────────

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ── Model Selection ────────────────────────────────────────────────────────────
# Step 1 — compare models with default hyperparameters using cross validation

# list of models
models = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

def compareusingcrossvalidation():
    for model in models:

        cv_score      = cross_val_score(model, x, y, cv=5, scoring='r2')
        mean_accuracy = sum(cv_score) / len(cv_score)
        mean_accuracy = mean_accuracy * 100
        mean_accuracy = round(mean_accuracy, 2)

        print('Cross validation accuracies for the', model, '=', cv_score)
        print('Accuracy score of the ', model, '=', mean_accuracy, '%')
        print('-----------------------------------------')

compareusingcrossvalidation()

# ── Step 2 — compare models with different hyperparameters using GridSearchCV ──

models = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

# creating a dictionary that contains hyperparameter values for above models
model_hyperparameters = {
    'ridge_hyperparameters': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'KNN_hyperparameters': {
        'n_neighbors': [3, 5, 10]
    },
    'random_forest_hyperparameters': {
        'n_estimators': [10, 50, 100, 200]
    },
    'gradientboosting_hyperparameters': {
        'n_estimators':  [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth':     [3, 4],
    }
}

print(model_hyperparameters.keys())

modelkeys = list(model_hyperparameters.keys())
print(modelkeys)

def modelselection(list_of_models, hyperparameters_dictionary):
    result = []
    i = 0
    for model in list_of_models:
        key    = modelkeys[i]
        params = hyperparameters_dictionary[key]
        i += 1
        print(model)
        print(params)
        print('---------------------------------------------------')

        classifier = GridSearchCV(model, params, cv=5, scoring='r2')
        classifier.fit(x_train, y_train)
        result.append({
            'model used':           model,
            'highest score':        classifier.best_score_,
            'best hyperparameters': classifier.best_params_,
            'best estimator':       classifier.best_estimator_,
        })

    result_dataframe = pd.DataFrame(
        result,
        columns=['model used', 'highest score', 'best hyperparameters', 'best estimator']
    )

    return result_dataframe

result = modelselection(models, model_hyperparameters)
print(result)

# ── Step 3 — pick best model ───────────────────────────────────────────────────

best_idx   = result['highest score'].idxmax()
best_row   = result.iloc[best_idx]
best_model = best_row['best estimator']

print('\nBest model:', best_row['model used'])
print('Best R² score:', round(best_row['highest score'] * 100, 2), '%')
print('Best hyperparameters:', best_row['best hyperparameters'])

# ── Step 4 — final evaluation on test set ─────────────────────────────────────

best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print('\nFinal Test Metrics:')
print(f'  MAE:  {mae:.4f} kg')
print(f'  RMSE: {rmse:.4f} kg')
print(f'  R²:   {r2:.4f}')

if hasattr(best_model, 'feature_importances_'):
    print('\nFeature Importances:')
    for f, imp in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
        print(f'  {f}: {imp:.4f}')

# ── Step 5 — save model + metadata ────────────────────────────────────────────

joblib.dump(best_model, 'waste_model.pkl')
print('\nModel saved → waste_model.pkl')

metadata = {
    'features': features,
    # Ratios used by app.py to derive People_served, Main_nonveg_kg, Main_veg_kg
    # at inference time from the fields page.tsx actually sends
    'ratios': {
        'people_per_student':   round(PEOPLE_RATIO, 6),
        'nonveg_kg_per_person': round(NONVEG_KG_RATIO, 6),
        'veg_kg_per_person':    round(VEG_KG_RATIO, 6),
    },
    'metrics': {
        'mae':  round(mae, 4),
        'rmse': round(rmse, 4),
        'r2':   round(r2, 4),
    },
    'feature_importances': (
        dict(zip(features, best_model.feature_importances_.tolist()))
        if hasattr(best_model, 'feature_importances_') else {}
    ),
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Metadata saved → model_metadata.json')
print('\nDone. Start the API with:  python app.py')
