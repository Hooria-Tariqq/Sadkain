import pandas as pd

# --- LOAD FROM STEP 2 ---
location = pd.read_csv("data/location_with_preds.csv")
conditions = pd.read_csv("data/conditions.csv")

def build_reason(row_loc, row_cond):
    if row_loc['diff'] > 0.5:
        return "Not explainable by weather/time conditions alone (inspect road geometry)."

    factors = []
    if row_cond.get('Weather_Group_Rain', 0) == 1:
        factors.append("Rain")
    elif row_cond.get('Weather_Group_Snow', 0) == 1:
        factors.append("Snow")
    elif row_cond.get('Weather_Group_Fog', 0) == 1:
        factors.append("Fog")
    elif row_cond.get('Weather_Group_Cloudy', 0) == 1:
        factors.append("Overcast")

    if row_cond['Visibility(mi)'] <= 3.0:
        factors.append(f"Low visibility ({row_cond['Visibility(mi)']} mi)")
    if row_cond['Temperature(F)'] <= 32.0:
        factors.append(f"Freezing ({row_cond['Temperature(F)']}°F)")
    if row_cond['Is_Night'] == 1:
        factors.append("Nighttime")

    if factors:
        return "Condition-driven: " + ", ".join(factors) + "."
    return "Clear daytime conditions (severity matched baseline expectation)."

location['Condition_Note'] = [
    build_reason(location.iloc[i], conditions.iloc[i]) for i in range(len(location))
]
    
location['Street'] = location['Street'].str.strip()

location['Lat_R'] = location['Start_Lat'].round(2)
location['Lng_R'] = location['Start_Lng'].round(2)

location['Stretch'] = (location['Street'] + ' | '
                       + location['Lat_R'].astype(str) + ', '
                       + location['Lng_R'].astype(str))
location['Is_Highway'] = location['Street'].str.startswith(('I-', 'US-', 'VA-')).astype(int)

g = location.groupby('Stretch')

stretch = pd.DataFrame({
    'Accidents': g['severe'].count(),
    'Severe': g['severe'].sum(),
    'Avg_Actual': g['severe'].mean().round(3),
    'Avg_pred': g['pred'].mean().round(3),
    'Avg_Diff': g['diff'].mean().round(3),
    'Total_Diff': g['diff'].sum().round(1),
    'Is_Highway': g['Is_Highway'].first(),
    'Lat': g['Lat_R'].first(),
    'Lng': g['Lng_R'].first(),
})

for flag in ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Station', 'Amenity']:
    stretch[flag + '_%'] = (g[flag].mean() * 100).round(1)

sev_counts = pd.crosstab(location['Stretch'], location['Severity'])
sev_counts.columns = ['Sev_1', 'Sev_2', 'Sev_3', 'Sev_4']
stretch = stretch.join(sev_counts)

stretch = stretch[stretch['Accidents'] >= 30]

highways = stretch[stretch['Is_Highway'] == 1].sort_values('Total_Diff', ascending=False)
local = stretch[stretch['Is_Highway'] == 0].sort_values('Total_Diff', ascending=False)

# --- CONNECTION TO STREAMLIT (STEP 4) ---
# reset_index() keeps the 'Stretch' column inside the saved CSV!
highways.reset_index().to_csv("data/highways.csv", index=False)
local.reset_index().to_csv("data/local.csv", index=False)
location.to_csv("data/location_final.csv", index=False)
#print("Step 3 Done: Saved highways.csv, local.csv, location_final.csv")