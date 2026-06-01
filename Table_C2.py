import pandas as pd
from statsmodels.tsa.api import VAR

# =====================================================================
# STEP 1: LOAD AND PROCESS RAW AGMARKNET CROPS DATA
# =====================================================================
print("Processing raw Agmarknet crop balances...")
df_crops = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\agmarknet_data.csv", skiprows=1)
df_crops.columns = df_crops.columns.str.strip()

df_crops['State'] = df_crops['State'].astype(str).str.strip()
df_crops['Commodity'] = df_crops['Commodity'].astype(str).str.strip()
df_crops['Modal Price'] = df_crops['Modal Price'].astype(str).str.replace(',', '', regex=True)
df_crops['Modal Price'] = pd.to_numeric(df_crops['Modal Price'], errors='coerce')

df_crops['Date'] = pd.to_datetime(df_crops['Date'], format='%d-%m-%Y', errors='coerce')
df_crops.dropna(subset=['Date', 'Modal Price'], inplace=True)

mh_onion = df_crops[(df_crops['State'] == 'Maharashtra') & (df_crops['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_onion = df_crops[(df_crops['State'] == 'West Bengal') & (df_crops['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_potato = df_crops[(df_crops['State'] == 'West Bengal') & (df_crops['Commodity'] == 'Potato')].groupby('Date')['Modal Price'].mean()

crop_matrix = pd.concat([mh_onion, wb_potato, wb_onion], axis=1, keys=['MH_Onion', 'WB_Potato', 'WB_Onion'])

# =====================================================================
# STEP 2: LOAD AND PROCESS FUEL CONTROL VECTOR
# =====================================================================
print("Processing Kolkata retail diesel inputs...")
df_fuel = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\diesel_prices.csv") 
df_fuel.columns = df_fuel.columns.str.strip()

df_fuel['Date'] = pd.to_datetime(df_fuel['Date'], format='mixed', errors='coerce')
df_fuel['Kolkata'] = pd.to_numeric(df_fuel['Kolkata'], errors='coerce')
df_fuel.dropna(subset=['Date', 'Kolkata'], inplace=True)

df_fuel_clean = df_fuel.groupby('Date')['Kolkata'].mean()
diesel_vector = df_fuel_clean.rename('Kolkata_Diesel')

# =====================================================================
# STEP 3: INTERPOLATE, MERGE, AND FILTER BY THESIS TIME WINDOW
# =====================================================================
print("Merging vectors into structural database...")
crop_matrix.index = pd.to_datetime(crop_matrix.index)
diesel_vector.index = pd.to_datetime(diesel_vector.index)

final_matrix = pd.concat([crop_matrix, diesel_vector], axis=1)
final_matrix.sort_index(inplace=True)
final_matrix = final_matrix.ffill().bfill()

appendix_data = final_matrix.loc['2024-04-01':'2026-03-31']

# =====================================================================
# STEP 4: INITIALIZE AND FIT THE MULTIVARIATE VAR(7) MODEL
# =====================================================================
print("Initializing and fitting structural VAR(7) model...")
model = VAR(appendix_data)
fitted_var_model = model.fit(maxlags=7, ic=None)

# =====================================================================
# STEP 5: PRINT TABLE C.1 SAMPLE PREVIEW
# =====================================================================
print("\n" + "="*80)
print("            APPENDIX C.1: RAW DATA MATRIX CHRONOLOGICAL PREVIEW          ")
print("="*80)
display_df = appendix_data.copy()
display_df.index = display_df.index.strftime('%d-%m-%Y')
display_df.index.name = 'Date'

print("\n--- STARTING OBSERVATIONS MATRIX SAMPLE (April 2024) ---")
print(display_df.head(5).to_markdown(floatfmt=".2f"))
print("\n                        [...]                        \n")
print("--- TERMINATING OBSERVATIONS MATRIX SAMPLE (March 2026) ---")
print(display_df.tail(5).to_markdown(floatfmt=".2f"))

# =====================================================================
# STEP 6: EXTRACT AND PRINT VALUES FOR APPENDIX TABLE C.2
# =====================================================================
print("\n" + "="*80)
print("          EXTRACTING VALUES FOR APPENDIX TABLE C.2               ")
print("="*80)

target_equation = "WB_Onion"
c2_matrix = pd.DataFrame({
    'Coefficient': fitted_var_model.params[target_equation],
    'Standard Error': fitted_var_model.stderr[target_equation],
    't-Statistic': fitted_var_model.tvalues[target_equation],
    'Prob. (p-value)': fitted_var_model.pvalues[target_equation]
})

new_index_labels = []
for label in c2_matrix.index:
    if label == 'const':
        new_index_labels.append('Constant (c)')
    elif label.startswith('L'):
        parts = label.split('.')
        lag_num = parts[0][1:]
        var_name = parts[1]
        new_index_labels.append(f"{var_name} (-{lag_num})")
    else:
        new_index_labels.append(label)

c2_matrix.index = new_index_labels
c2_matrix.index.name = 'Regressor (Lag)'

print(f"\n--- Complete VAR(7) Regression Estimates for Target Node: {target_equation} ---")
print(c2_matrix.to_markdown(floatfmt=".4f"))
print("="*80)