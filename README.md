# Saڑkain (سڑکیں) — Road Anomaly & Residual Risk Triage

**Saڑkain** *(Urdu: سڑکیں — "Roads")* is an end-to-end machine learning and spatial analytics dashboard that isolates unexplained road crash severity in Richmond, Virginia. Instead of ranking corridors solely by raw accident counts, it identifies **1 km road stretches where crashes are significantly more severe than weather, visibility, and time-of-day conditions alone predict**—pinpointing underlying road geometry and infrastructure hazards.

---

## Core Methodology (Residual Risk Analysis)

1. **Condition-Only Baseline Model (`2_train_model.py`):**
   A 5-fold cross-validated `RandomForestClassifier` predicts the baseline probability of a severe accident (`Severity >= 3`) using *strictly* environmental and temporal features (`Temperature(F)`, `Visibility(mi)`, `Weather_Group`, `Hour`, `DayOfWeek`, `Month`, `Is_Night`).
2. **Residual Scoring (`diff`):**
   Each crash receives a residual score:
   `diff = actual_severe - predicted_probability`
   Positive residuals highlight accidents worse than weather and time explain.
3. **1 km Spatial Chunking (`3_postprocess_chunks.py`):**
   GPS coordinates are binned (`round(2)`, ~1.1 km) and grouped by street name (`Accidents >= 30`), separating major **Highways** (`I-`, `US-`, `VA-`) from **Local City Streets**.
4. **Impact-Weighted Priority (`Total_Diff`):**
   Road segments are ranked by the sum of residuals (`Total_Diff`) to prioritize stretches with the highest total volume of excess severe crashes rather than small-sample rate spikes.

---

## Repository Structure

```text
Saڑkain/
├── data/
│   ├── richmond.csv              # Raw Richmond crash dataset
│   ├── location.csv              # Spatial coordinates & road infrastructure flags
│   ├── conditions.csv            # Encoded weather & temporal features
│   ├── y.csv                     # Binary severity target (1 = Severity >= 3)
│   ├── location_with_preds.csv   # Records with out-of-fold probabilities & residuals
│   ├── location_final.csv        # Enriched dataset with 1 km Stretch IDs & Condition Notes
│   ├── highways.csv              # Ranked inspection table for highway corridors
│   └── local.csv                 # Ranked inspection table for local city streets
├── 1_preprocess.py               # Data cleaning, weather grouping, and feature split
├── 2_train_model.py              # 5-fold Random Forest cross-validation & residual scoring
├── 3_postprocess_chunks.py       # 1 km stretch grouping, Total_Diff ranking, & condition notes
├── 4_app.py                      # Interactive Streamlit & Plotly inspection dashboard
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusion rules
└── README.md                     # Project documentation
```
## Dashboard Features (4_app.py)
Dual Inspection Priority Tables: Side-by-side Top 10 rankings for Highway corridors and Local City Streets ordered by excess severe crashes (Total_Diff).

Street Search & Auto-Zoom: Select or type any flagged street in Richmond to zoom directly onto its crash cluster.

Severity Filtering: Filter individual map pins by 1 - Nonfatal, 2 - Minor, 3 - Major, or 4 - Fatal.

Automated Condition Notes: Hovering over any accident pin displays a single-line diagnostic note stating whether the crash severity matched adverse weather/night conditions or is unexplainable by conditions alone.

Quickstart
Install dependencies:

```Bash
pip install -r requirements.txt
Run the ML & post-processing pipeline (optional—precomputed CSVs are included in data/):
```
```Bash
python preprocessing.py
python train_model.py
python postprocessing.py
Launch the interactive dashboard:
```
```Bash
streamlit run 4_app.py
```