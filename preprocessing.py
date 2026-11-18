import pandas as pd

df = pd.read_csv("data/richmond.csv")

df = df.dropna()
df = df.copy()

#Feature Extraction 
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month

#Reducing redundant entries
def group_weather(label):
    label = str(label).lower()

    # Fog conditions
    if any(word in label for word in ['fog', 'haze', 'mist']):
        return 'Fog'

    # Rain and thunderstorms
    if any(word in label for word in [
        'rain', 'drizzle', 'storm', 'thunder', 't-storm'
    ]):
        return 'Rain'

    # Snow and wintry conditions
    if any(word in label for word in [
        'snow', 'sleet', 'wintry'
    ]):
        return 'Snow'

    # Clear conditions
    if any(word in label for word in ['fair', 'clear']):
        return 'Clear'

    # Cloudy conditions
    if any(word in label for word in ['cloud', 'overcast']):
        return 'Cloudy'

    return 'Other'

df['Weather_Group'] = df['Weather_Condition'].apply(group_weather)


df['severe'] = (df['Severity'] >= 3).astype(int)

#Dividing main df into multiple
#Location is used for the final reasoning 
#Conditions is used for training
location = df[['Severity', 'severe', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Street', 'County', 'Zipcode', 'Crossing', 'Junction', 'Station', 'Stop', 'Traffic_Signal', 'Amenity']]
location = location.copy()
conditions = df[['Temperature(F)', 'Visibility(mi)', 'Weather_Group', 'Sunrise_Sunset', 'Hour', 'DayOfWeek', 'Month']]
conditions = conditions.copy()

conditions['Is_Night'] = (conditions['Sunrise_Sunset'] == 'Night').astype(int)
conditions = conditions.drop(columns=['Sunrise_Sunset'])

conditions = pd.get_dummies(conditions, columns=['Weather_Group'], drop_first=True, dtype=int)

#Actual 
y = df['severe']
y = y.copy()

# --- CONNECTION TO NEXT FILE ---
location.to_csv("data/location.csv", index=False)
conditions.to_csv("data/conditions.csv", index=False)
y.to_csv("data/y.csv", index=False)
#print("Step 1 Done: Saved location.csv, conditions.csv, y.csv")