import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Richmond Road Priority", layout="wide")

# --- LOAD OUTPUTS FROM STEP 3 ---
@st.cache_data
def load_data():
    highways = pd.read_csv("data/highways.csv")
    local = pd.read_csv("data/local.csv")
    location = pd.read_csv("data/location_final.csv")
    return highways, local, location

highways, local, location = load_data()

# Only keep individual points belonging to valid stretches (Accidents >= 30)
valid_stretches = set(highways['Stretch']).union(set(local['Stretch']))
map_points = location[location['Stretch'].isin(valid_stretches)].copy()

st.header("Saڑkain")

left_col, right_col = st.columns(2, gap="medium")

with left_col:
    st.subheader("Top 10 Highway Stretches")
    st.dataframe(highways[['Stretch', 'Accidents', 'Total_Diff', 'Avg_Diff']].head(10), use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Top 10 City Street Stretches")
    st.dataframe(local[['Stretch', 'Accidents', 'Total_Diff', 'Avg_Diff']].head(10), use_container_width=True, hide_index=True)

st.markdown("---")

st.subheader("Search Street")
streets = sorted(map_points['Street'].unique())
selected_street = st.selectbox("Select a street to zoom in:", ["All Richmond (Overview)"] + streets)

st.subheader("Inspection Priority Map")
if selected_street != "All Richmond (Overview)":
    plot_df = map_points[map_points['Street'] == selected_street]
    center = {"lat": plot_df['Start_Lat'].mean(), "lon": plot_df['Start_Lng'].mean()}
    zoom = 14
else:
    plot_df = map_points
    center = {"lat": plot_df['Start_Lat'].mean(), "lon": plot_df['Start_Lng'].mean()}
    zoom = 10.5

fig = px.scatter_map(
    plot_df,
    lat="Start_Lat",
    lon="Start_Lng",
    color="diff",
    color_continuous_scale="RdYlBu_r",
    hover_name="Stretch",
    hover_data={
        "Severity": True,
        "Condition_Note": True,
        "pred": ":.2f",
        "diff": ":.2f",
        "Start_Lat": ":.4f",
        "Start_Lng": ":.4f"
    },
    center=center,
    zoom=zoom,
    map_style="open-street-map",
    height=650
)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)