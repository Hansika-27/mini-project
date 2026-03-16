"""
Food Wastage Predictor — Model Training
Matches the page_new.tsx → app.py contract.

page_new.tsx sends to POST /predict:
    studentsEnrolled, averageAttendance, menusServed,
    leftoverFromPreviousDay, specialEvent, mealType,
    dayOfWeek, menuItems (list of strings)

nonveg_items = count of 'nonveg' entries in menuItems  (0, 1, 2 …)

app.py derives the three missing model features at inference time:
    People_served  = Students_enrolled × people_per_student  (median ratio)
    Main_nonveg_kg = People_served     × nonveg_kg_per_person (median ratio, if nonveg > 0)
    Main_veg_kg    = People_served     × veg_kg_per_person    (median ratio)

These ratios are saved in model_metadata.json so app.py loads them at startup.

Run:
    python train_model_new.py          — trains and saves model + metadata
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Load Data ──────────────────────────────────────────────────────────────────

df = pd.read_csv('Final_data_1.csv')

print('Shape:', df.shape)
print(df.isnull().sum())

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
# This matches page_new.tsx which counts 'nonveg' selections from menuItems.
# No transformation needed — use the column directly.

# ── Compute per-person ratios from training data ───────────────────────────────
# These are saved to model_metadata.json so app.py can derive
# People_served, Main_nonveg_kg, Main_veg_kg at inference time
# (page_new.tsx does not send these fields directly).

df['people_per_student']   = df['People_served']   / df['Students_enrolled'].replace(0, np.nan)
df['nonveg_kg_per_person'] = df['Main_nonveg_kg']  / df['People_served'].replace(0, np.nan)
df['veg_kg_per_person']    = df['Main_veg_kg']     / df['People_served'].replace(0, np.nan)

PEOPLE_RATIO    = df['people_per_student'].median()
NONVEG_KG_RATIO = df['nonveg_kg_per_person'].median()
VEG_KG_RATIO    = df['veg_kg_per_person'].median()

print(f'\nMedian ratios:')
print(f'  people / student:    {PEOPLE_RATIO:.4f}')
print(f'  nonveg_kg / person:  {NONVEG_KG_RATIO:.4f}')
print(f'  veg_kg / person:     {VEG_KG_RATIO:.4f}')

# ── Features — must match app.py _parse() vector order exactly ────────────────

features = [
    'Students_enrolled',       # from studentsEnrolled
    'Attendance_percent',      # from averageAttendance
    'Special_event_enc',       # from specialEvent (yes/no → 1/0)
    'Menu_count',              # from menusServed
    'Previous_day_leftover_kg',# from leftoverFromPreviousDay
    'Nonveg_items',            # count of 'nonveg' in menuItems
    'Meal_type_enc',           # dinner/snacks → 1, else 0
    'Day_enc',                 # Monday→0 … Sunday→6
    'Main_nonveg_kg',          # derived: people × nonveg_kg_per_person
    'Main_veg_kg',             # derived: people × veg_kg_per_person
    'People_served',           # derived: students × people_per_student
]

x = df[features].values
y = df['Total_Waste_kg'].values

# ── Train/Test Split ───────────────────────────────────────────────────────────

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ── Step 1: Model Comparison via Cross-Validation ─────────────────────────────

models_cv = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

def compare_using_cross_validation():
    print('\n── Cross-Validation Comparison ──────────────────────')
    for model in models_cv:
        cv_scores = cross_val_score(model, x, y, cv=5, scoring='r2')
        mean_r2   = round(cv_scores.mean() * 100, 2)
        print(f'CV R² scores for {model}: {cv_scores}')
        print(f'Mean R² score: {mean_r2}%')
        print('------------------------------------------')

compare_using_cross_validation()

# ── Step 2: Hyperparameter Tuning with GridSearchCV ────────────────────────────

models_grid = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

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

model_keys = list(model_hyperparameters.keys())

def model_selection(list_of_models, hyperparameters_dictionary):
    print('\n── GridSearchCV Tuning ───────────────────────────────')
    result = []
    for i, model in enumerate(list_of_models):
        key    = model_keys[i]
        params = hyperparameters_dictionary[key]
        print(f'\nTuning: {model}')
        print(f'Params: {params}')
        print('--------------------------------------------------')

        gs = GridSearchCV(model, params, cv=5, scoring='r2')
        gs.fit(x_train, y_train)
        result.append({
            'model used':           model,
            'highest score':        gs.best_score_,
            'best hyperparameters': gs.best_params_,
            'best estimator':       gs.best_estimator_,
        })

    return pd.DataFrame(
        result, columns=['model used', 'highest score', 'best hyperparameters', 'best estimator']
    )

result = model_selection(models_grid, model_hyperparameters)
print('\n', result)

# ── Step 3: Pick Best Model ────────────────────────────────────────────────────

best_idx   = result['highest score'].idxmax()
best_row   = result.iloc[best_idx]
best_model = best_row['best estimator']

print(f'\nBest model:           {best_row["model used"]}')
print(f'Best CV R²:           {round(best_row["highest score"] * 100, 2)}%')
print(f'Best hyperparameters: {best_row["best hyperparameters"]}')

# ── Step 4: Final Evaluation ───────────────────────────────────────────────────

best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print(f'\nFinal Test Metrics:')
print(f'  MAE:  {mae:.4f} kg')
print(f'  RMSE: {rmse:.4f} kg')
print(f'  R²:   {r2:.4f}')

if hasattr(best_model, 'feature_importances_'):
    print('\nFeature Importances:')
    for f, imp in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
        print(f'  {f}: {imp:.4f}')

# ── Step 5: Save Model + Metadata ─────────────────────────────────────────────

joblib.dump(best_model, 'waste_model.pkl')
print('\nModel saved → waste_model.pkl')

metadata = {
    'features': features,
    # Ratios used by app.py to derive People_served, Main_nonveg_kg, Main_veg_kg
    # at inference time from the fields page_new.tsx actually sends
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
