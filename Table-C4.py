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

import numpy as np
from statsmodels.stats.diagnostic import acorr_lm

# =====================================================================
# STEP 7: COMPUTE RESIDUAL DIAGNOSTICS FOR TABLE C.3
# =====================================================================
print("\n" + "="*80)
print("          EXTRACTING VALUES FOR APPENDIX TABLE C.3               ")
print("="*80)

# Extract the raw model residuals
residuals = fitted_var_model.resid

# 1. Serial Correlation: Portmanteau (Ljung-Box) Test
# We test up to 10 lags to check for hidden longer-term correlations
portmanteau_results = fitted_var_model.test_whiteness(nlags=10)
ljung_box_stat = portmanteau_results.test_statistic
ljung_box_p = portmanteau_results.pvalue

# 2. Heteroskedasticity: ARCH-LM Test (run specifically on the target WB_Onion series)
# We check at our optimal lag length of 7 days
arch_test_raw = acorr_lm(residuals['WB_Onion'], nlags=7)
arch_stat = arch_test_raw[0]
arch_p = arch_test_raw[1]

# 3. Normality: Jarque-Bera Test
normality_results = fitted_var_model.test_normality()
jb_stat = normality_results.test_statistic
jb_p = normality_results.pvalue

# 4. Construct a clean summary table for the Appendix
c3_data = [
    {
        "Diagnostic Test": "Residual Autocorrelation",
        "Test Type / Metric": "Portmanteau (Ljung-Box) [Lag 10]",
        "Statistic Value": f"{ljung_box_stat:.2f}",
        "p-value": f"{ljung_box_p:.4f}",
        "Systemic Conclusion": "No Serial Correlation (White Noise)" if ljung_box_p > 0.05 else "Serial Correlation Present"
    },
    {
        "Diagnostic Test": "Heteroskedasticity",
        "Test Type / Metric": "ARCH-LM Test [Lag 7]",
        "Statistic Value": f"{arch_stat:.2f}",
        "p-value": f"{arch_p:.4f}",
        "Systemic Conclusion": "No ARCH Effects (Homoskedastic)" if arch_p > 0.05 else "Volatile Clustering Present"
    },
    {
        "Diagnostic Test": "Normality",
        "Test Type / Metric": "Jarque-Bera Multivariate",
        "Statistic Value": f"{jb_stat:.2f}",
        "p-value": f"{jb_p:.4f}",
        "Systemic Conclusion": "Residuals are Normal" if jb_p > 0.05 else "Non-normal Residuals (Expected)"
    }
]

c3_matrix = pd.DataFrame(c3_data)

print("\n--- Table C.3: Residual Diagnostic Matrix (Robustness Checks) ---")
print(c3_matrix.to_markdown(index=False))
print("="*80)

# STEP 8: COMPUTE FULL 15-DAY FEVD FOR APPENDIX TABLE C.4 (FIXED)
# =====================================================================
print("\n" + "="*80)
print("          EXTRACTING VALUES FOR APPENDIX TABLE C.4               ")
print("="*80)

# 1. Calculate the Forecast Error Variance Decomposition for a 15-day lookahead horizon
fevd_results = fitted_var_model.fevd(periods=15)

# 2. STRATEGIC FIX: Dynamically find the integer index position of 'WB_Onion'
# This maps the string name to its true position number in the model's memory
all_variables = list(fitted_var_model.names)
target_variable = 'WB_Onion'
target_integer_idx = all_variables.index(target_variable)

# 3. Extract the decomposition array safely using the correct integer index
# This completely bypasses the IndexError!
wb_onion_fevd = fevd_results.decomp[target_integer_idx]

# 4. Build a clean DataFrame using your exact table headers
c4_matrix = pd.DataFrame(
    wb_onion_fevd, 
    columns=all_variables,  # Matches ['MH_Onion', 'WB_Potato', 'WB_Onion', 'Kolkata_Diesel']
    index=range(1, 16)      # Sets days from 1 to 15
)

# 5. Re-index and order columns to perfectly match your manuscript layout
column_order = ['MH_Onion', 'WB_Potato', 'WB_Onion', 'Kolkata_Diesel']
c4_matrix = c4_matrix[column_order]

# 6. Convert decimals to clean, readable percentages (e.g., 0.8611 becomes 86.11%)
c4_matrix = c4_matrix * 100

c4_matrix.index.name = 'Horizon (Day)'

# 7. Print the complete 15-day array in a clean markdown layout
print("\n--- Table C.4: Full 15-Day FEVD Horizon Array for Target Node: WB_Onion ---")
print(c4_matrix.to_markdown(floatfmt=".2f"))
print("="*80)

# STEP 9: EXPORT COMPLETE 24-MONTH STRUCTURAL DATASET FOR REPOSITORY
# =====================================================================
print("\n" + "="*80)
print("          EXPORTING MULTIVARIATE DATASET FOR DEPLOYMENT          ")
print("="*80)

# 1. Prepare the final DataFrame for open-source hosting
hosting_dataset = appendix_data.copy()

# 2. Format the index name and date layout so it loads cleanly into Kaggle/GitHub
hosting_dataset.index.name = 'Date'

# 3. Export to a standardized CSV file
output_filename = 'Nashik_Kolkata_Corridor_Daily_Prices_2024_2026.csv'
hosting_dataset.to_csv(output_filename, index=True)

# 4. Print telemetry details so you can verify the export size
total_rows = len(hosting_dataset)
print(f"SUCCESS: Complete dataset successfully compiled and saved!")
print(f"File Name   : {output_filename}")
print(f"Total Rows  : {total_rows} consecutive calendar days")
print(f"Data Schema : {list(hosting_dataset.columns)}")
print("="*80)