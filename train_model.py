"""
Food Wastage Predictor — Model Training
Dataset: Final_data_2years.csv  (1460 rows, no missing values)
Style: follows Model_Selection_in_ML.ipynb exactly.
 
page.tsx (predict) sends to POST /predict:
    studentsEnrolled, averageAttendance, menusServed,
    leftoverFromPreviousDay, specialEvent, mealType,
    dayOfWeek, menuItems (list of strings)
 
nonveg_items = count of 'nonveg' entries in menuItems (0, 1, 2)
 
app.py derives recommended food quantities from per-person ratios
saved in model_metadata.json.
 
Run:
    python train_model.py    — trains and saves model + metadata
"""

#importing libraries
import numpy as np
import pandas as pd
import joblib
import json 

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


#Load Data
df = pd.read_csv('Final_data_2years.csv')

df.head()
df.shape
df.isnull().sum()
df['Total_Waste_kg'].describe()

print('Dataset shape: ', df.shape)
print('\nNull values:\n', df.isnull().sum())
print('nTarget stats:\n', df['Total_Waste_kg'].describe())


#Feature Engineering
DAY_ORDER = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

df['Day_enc']           = df['Day'].map(DAY_ORDER)
df['Meal_type_enc']     = (df['Meal_type'] == 'Dinner').astype(int)
df['Special_event_enc'] = (df['Special_event'] == 'Yes').astype(int)

#people_served derived from Students_enrolled x Attendance_percent
df['People_served'] = (df['Students_enrolled'] * df['Attendance_percent'] / 100).round()

#per-person food quantity ratios - derived from waste and people served 
#used by app.py to generate recommend cooking quantities
#Baseline serving sizes (Kg per person , from hostel norms)
RICE_KG_PP   =    0.15   #150g rice per person
DAL_KG_PP    =    0.10   #100g dal from person
ROTI_PP      =    3      #3 rotis per person
VEG_KG_PP    =    0.12   #120g veg per person
NONVEG_KG_PP =    0.10   #100g non-veg per person (when served)


#Features and Targets

features = [
    'Students_enrolled',        #studentsEnrolled
    'Attendance_percent',       #averageAttendance
    'Special_event_enc',        #specialEvent yes/no -> 1/0
    'Menu_count',               #menusServed
    'Previous_day_leftover_kg', #leftoverFromPreviousDay
    'Nonveg_items',             #count of 'nonveg' entries in menuItems (0, 1, 2)
    'Meal_type_enc',            #dinner -> 1, lunch -> 0
    'Day_enc',                  #Monday -> 0.... Sunday-> 6
    'People_served',            #derived: students x attendance/100
]


x = df[features]
y = df['Total_Waste_kg']

print('\nFeatures shape: ', x.shape)
print(x.head())
print('\nTarget head:\n', y.head())

x = np.asarray(x)
y = np.asarray(y)


#Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
 
print(f'\nTrain size: {x_train.shape[0]}  |  Test size: {x_test.shape[0]}')


# ── Step 1 — Model Comparison via Cross Validation ────────────────────────────
# comparing the models with default hyperparameters using cross validation
 
# list of models
models = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

def compareusingcrossvalidation():
    for model in models:

        cv_score          = cross_val_score(model, x, y, cv=5, scoring='r2')
        mean_accuracy     = sum(cv_score) / len(cv_score)
        mean_accuracy     = mean_accuracy * 100
        mean_accuracy     = round(mean_accuracy, 2)

        print('Cross validation accuracies for the' , model, '=', cv_score)
        print('Accuracy score of the ' , model, '=', mean_accuracy, '%')
        print('-----------------------------------------')

compareusingcrossvalidation()


# ── Step 2 — GridSearchCV Hyperparameter Tuning ───────────────────────────────
# comparing models with different hyperparameters using GridSearchCV
models = [
    Ridge(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
]

#creating a dictionary that contains hyperparameter values for above models
model_hyperparameters = {
    'ridge_hyperparameters': {
        'alpha' : [0.1, 1.0, 10.0, 100.0]
    },
    'KNN_hyperparameters' : {
        'n_neighbors': [3, 5, 7, 10]
    },
    'random-forest-hyperparameters' : {
        'n_estimators': [50, 100, 200],
        'max_depth' : [None, 10, 20]
    },
    'gradientboosting_hyperparamerts' : {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth':     [3, 4, 5],
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

        classifier = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs = -1)
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


# ── Step 3 — Pick Best Model ──────────────────────────────────────────────────

best_idx     = result['highest score'].idxmax()
best_row     = result.iloc[best_idx]
best_model   = best_row['best estimator']

print('\nBest model:', best_row['model used'])
print('Best R² score:', round(best_row['highest score'] * 100, 2), '%')
print('Best hyperparameters:', best_row['best hyperparameters'])


# ── Step 4 — Final Evaluation on Test Set ────────────────────────────────────

best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print('\nFinal Test Metrics: ')
print(f' MAE: {mae:.4f} kg')
print(f' RMSE: {rmse:.4f} kg')
print(f'  R²:   {r2:.4f}  ({round(r2*100,2)}%)')

if hasattr(best_model, 'feature_importances_'):
    print('\nFeature Importances: ')
    for f, imp in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
        print(f' {f}: {imp:.4f}')



# ── Step 5 — Save Model + Metadata ────────────────────────────────────────────

joblib.dump(best_model, 'waste_model.pkl')
print('\nModel saved -> waste_model.pkl')

metadata = {
    'features': features,

    # Per-Person food quantity ratios used by app.py
    # to generate recommend cooking quantities
    'serving_ratios': {
        'rice_kg_per_person':     RICE_KG_PP,
        'dal_kg_per_person':      DAL_KG_PP,
        'roti_per_person':        ROTI_PP,
        'veg_kg_per_person':      VEG_KG_PP,
        'nonveg_kg_per_person':   NONVEG_KG_PP,
    },
    'metrics': {
        'mae':     round(mae, 4),
        'rmse':    round(rmse, 4),
        'r2':      round(r2, 4),
    },
    'feature_importances': (
        dict(zip(features, best_model.feature_importances_.tolist()))
        if hasattr(best_model, 'feature_importances_') else {}
    ),
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Metadata saved -> model_metadata.json')
print('\nDone. Start the API with: python app.py')

    