import pandas as pd
import numpy as np

# Loading data
df_agri = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\agmarknet_data.csv", skiprows=1)
df_agri.columns = df_agri.columns.str.strip()
df_agri['State'] = df_agri['State'].astype(str).str.strip().str.title() # Normalize state names
df_agri['Commodity'] = df_agri['Commodity'].astype(str).str.strip().str.title() # Normalize commodity names

states = ['Maharashtra', 'West Bengal']
commodities = ['Onion', 'Potato']
df_filtered = df_agri[df_agri['State'].isin(states) & df_agri['Commodity'].isin(commodities)].copy()
df_filtered.columns = df_filtered.columns.str.strip()  # Strip whitespace from column names
df_filtered['Modal Price'] = (
    df_filtered['Modal Price'].replace({',': ""}, regex=True).pipe(pd.to_numeric, errors='coerce'))
df_filtered = df_filtered.dropna(subset = ['Modal Price'])  # Drop rows where Modal Price is NaN


print(df_filtered['State'].unique())
print(df_filtered['Commodity'].unique())
print(f"Rows after filtering: {len(df_filtered)}")

# loading the diesel prices data
df_diesel_raw = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\diesel_prices.csv")
df_diesel_raw.columns = df_diesel_raw.columns.str.strip()
df_diesel_raw['Date'] = pd.to_datetime(df_diesel_raw['Date'], dayfirst = True, errors = 'coerce')
df_diesel = df_diesel_raw[['Date', 'Kolkata']].rename(columns = {'Kolkata': 'Kolkata_Diesel'})
df_diesel['Kolkata_Diesel'] = df_diesel['Kolkata_Diesel'].replace({',': ""}, regex=True).pipe(pd.to_numeric, errors='coerce')

df_diesel = df_diesel.set_index("Date").sort_index()


# Create pivot table from filtered data
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], dayfirst = True, errors = 'coerce')
# df_pivot = df_filtered.set_index('Date').sort_index()

df_pivot = df_filtered.pivot_table(index = 'Date',
                                   columns = ['State', 'Commodity'],
                                   values = 'Modal Price',
                                   aggfunc = 'mean')
# Renaming columns
df_pivot.columns = [f"{'MH' if s == 'Maharashtra' else 'WB'}_{c}" for s, c in df_pivot.columns]

# Ensuring index is unique by averaging any accidental duplicates
df_pivot = df_pivot.groupby(df_pivot.index).mean()

# 5 variable merge
# Joining the consolidated pivot table with the diesel data

df_master = df_pivot.join(df_diesel, how = 'inner')
df_master = df_master.groupby(df_master.index).mean()  # Average any duplicates that may arise from the join

all_days = pd.date_range(start='2024-04-01', end = '2026-03-31')
df_master = df_master.reindex(all_days).ffill().dropna() # Reindexing and forward filling for missing values & dropping NaNs

print(f"Dataset Ready! Total Variables: {len(df_master.columns)}")

df_master.dtypes
print(df_pivot.index[df_pivot.index.duplicated()])
print(df_master.corr())

# Running ADF test on the price series to check for stationarity

from statsmodels.tsa.stattools import adfuller

def run_adf_test(df):
    for col in df.columns:
        res = adfuller(df[col].dropna())
        print(f"Variable: {col}")
        print(f"p-value: {res[1]:.4f}")
        if res[1] < 0.05:
            print("Result: Stationary ✅")
        else:
            print("Result: Non-Stationary ❌")
        print("-"*30)

run_adf_test(df_master)

# Differencing the non-stationary series to make it stationary to run VAR test

df_diff = df_master.diff().dropna() # First order differencing to achieve stationarity

# Re-running ADF test on the differenced series to confirm stationarity
print("Checking statinarity of Daily Changes:")

for col in df_diff.columns:
    res = adfuller(df_diff[col])
    print(f"{col:15} | p-value: {res[1]:.4f} | {'Stationary ✅' if res[1] < 0.05 else 'Still Non-Stationary ❌'}")

# Creating the VAR model on the differenced data to analyze the relationships between the variables and to check for Granger causality.

from statsmodels.tsa.api import VAR
model = VAR(df_diff)

# Finding optimal lag length ( The memory of the market)
order = model.select_order(maxlags = 15)
print(order.summary())

# Fitting the model using AIC recommended lag length
results = model.fit(maxlags = 15, ic = 'aic')
print(f"Model fitted with lag order: {results.k_ar}")

# Calculating FEVD for 10 days
fevd = results.fevd(10)
print(fevd.summary())

# Plotting to see the 'flow of influence' over time (Impulse Response Function)
import matplotlib.pyplot as plt
fevd.plot()
plt.show()

# Granger Causality Test to prove whether lead-lag relationship is "real" or just a coincidence.
# Does MH_Onion 'Granger Cause' WB_Onion?
causality = results.test_causality('WB_Onion', ['MH_Onion'], kind = 'f')
print(f"Causality P-Value: {causality.pvalue:.4f}")

# Combined plot of IRF(IMPULSE RESPONSE FUNCTION) to show the ripple moving through the system
# Final plot for thesis appendix
irf = results.irf(10)
fig = irf.plot(impulse = 'MH_Onion', response = 'WB_Onion', orth = True)
plt.title("Fig 1: Dynamic Response of WB Onion Prices to a Shock in MH Supply", fontsize = 12)
plt.xlabel("Days after Shock (Lags)")
plt.ylabel("Price Change (Rs./Quintal)")
plt.grid(True, alpha = 0.3)
plt.show()

# Tables for Results chapter

# Creating a list to store results
table1_data = []
for col in df_diff.columns:
    stats = df_diff[col].describe()
    adf_result = adfuller(df_diff[col])

    table1_data.append({
        'Variable': col,
        'Mean': stats['std'],
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Result': 'Stationary(l(1))'
    })

table1 = pd.DataFrame(table1_data)
print("--- TABLE 1: DESCRIPTIVE STATISTICS & ADF ---")
print(table1.to_string(index = False))

# Code for Table-2: Granger Causality
# Testing which variables "cause" WB Onion
print("--- TABLE 2: GRANGER CAUSALITY (Target: WB_Onion) ---")

predictors = [c for c in df_diff.columns if c != 'WB_Onion']

for p in predictors:
    test_result = results.test_causality('WB_Onion', [p], kind = 'f') # Testing if 'p' causes 'WB_Onion'
    print(f"H0:{p} does not Granger Cause WB_Onion")
    print(f"F-Statistic:{test_result.test_statistic:.4f}")
    print(f"p-value: {test_result.pvalue:.4f}")
    print(f"Decision: {'Reject H0' if test_result.pvalue < 0.05 else 'Fail to Reject'}")
    print("-"*40)

# Code for Table-3: FEVD (Influence Breakdown)
# Extracting the exact percentages for 1st, 3rd, 7th and 10th days to show how the influence grows over time
# Calculate FEVD
fevd = results.fevd(10)
# finding the index of WB_Onion automatically
target_idx = df_diff.columns.get_loc('WB_Onion')

# Extracting decomposition values
# Getting the values where the 'response' is WB_Onion
decomp_wb = fevd.decomp[target_idx] 


print("--- TABLE 3: FEVD FOR WB_ONION (Percentages) ---")
header = f"{'Day':>4} | "+" |".join([f"{col:>15}" for col in df_diff.columns])
print(header)
print("-" * len(header))


# Looping through specific days (Day 1 is index 0)
for i in [0, 2, 6, 9]:
    day_label = i + 1
    row_values = decomp_wb[i] # row values contains the contribution of each variable to WB_Onion's variance

    row_str = f"{day_label:>4} |"
    row_str += "|".join([f"{val*100:>14.2f}%" for val in row_values])
    print(row_str)

# STEP 5: EXTRACT RAW EQUATION COEFFICIENTS FOR TABLE C.2
# =====================================================================
print("\n" + "="*80)
print("          EXTRACTING VALUES FOR APPENDIX TABLE C.2               ")
print("="*80)

# 1. Isolate the regression results specifically for the 'WB_Onion' equation
target_equation = "WB_Onion"

# 2. Extract the raw coefficients, standard errors, t-stats, and p-values
coefficients = fitted_var_model.params[target_equation]
standard_errors = fitted_var_model.stderr[target_equation]
t_values = fitted_var_model.tvalues[target_equation]
p_values = fitted_var_model.pvalues[target_equation]

# 3. Combine these vectors into a clean summary DataFrame
c2_matrix = pd.DataFrame({
    'Coefficient': coefficients,
    'Standard Error': standard_errors,
    't-Statistic': t_values,
    'Prob. (p-value)': p_values
})

# 4. Clean up the row names (index) so they look professional in your thesis
# This replaces names like 'L1.MH_Onion' with 'MH_Onion (-1)'
new_index_labels = []
for label in c2_matrix.index:
    if label == 'const':
        new_index_labels.append('Constant (c)')
    elif label.startswith('L'):
        # Split 'L1.Variable' into lag number and variable name
        parts = label.split('.')
        lag_num = parts[0][1:] # extracts the '1' from 'L1'
        var_name = parts[1]    # extracts the variable name
        new_index_labels.append(f"{var_name} (-{lag_num})")
    else:
        new_index_labels.append(label)

c2_matrix.index = new_index_labels
c2_matrix.index.name = 'Regressor (Lag)'

# 5. Print the complete matrix in a clean markdown table
print(f"\n--- Complete VAR(7) Regression Estimates for Target Node: {target_equation} ---")
print(c2_matrix.to_markdown(floatfmt=".4f"))
print("="*80)

# Optional: Save it directly to a CSV file for easy copying into Microsoft Word
# c2_matrix.to_csv('Appendix_C2_Coefficient_Matrix.csv')




                                                

                                                
                                                









                                                     




