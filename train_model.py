import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier

# --- LOAD FROM STEP 1 ---
location = pd.read_csv("data/location.csv")
conditions = pd.read_csv("data/conditions.csv")
y = pd.read_csv("data/y.csv")['severe']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, random_state=42)

pred = cross_val_predict(model, conditions, y, cv=cv, method='predict_proba')[:, 1]

location['pred'] = pred
location['diff'] = location['severe'] - location['pred']

# --- CONNECTION TO NEXT FILE ---
location.to_csv("data/location_with_preds.csv", index=False)
#print("Step 2 Done: Saved location_with_preds.csv")