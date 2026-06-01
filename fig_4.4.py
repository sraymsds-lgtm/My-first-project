import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Step 1: Load and clean your actual Agmarknet data file
# ag_file_name = 'your_agmarknet_data.csv'
df_raw = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\agmarknet_data.csv", skiprows=1)

df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y')
df_raw.set_index('Date', inplace=True)
df_raw.sort_index(inplace=True)
df_raw['State'] = df_raw['State'].astype(str).str.strip()
df_raw['Commodity'] = df_raw['Commodity'].astype(str).str.strip()
df_raw['Modal Price'] = df_raw['Modal Price'].astype(str).str.replace(',', '', regex=True)
df_raw['Modal Price'] = pd.to_numeric(df_raw['Modal Price'], errors='coerce')
df_raw.dropna(subset=['Modal Price'], inplace=True)

# Step 2: Extract variables and build structural matrix
mh_onion = df_raw[(df_raw['State'] == 'Maharashtra') & (df_raw['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_onion = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_potato = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Potato')].groupby('Date')['Modal Price'].mean()

# Ordering variables logically: Source Nodes -> Destination Target Node
var_data = pd.concat([mh_onion, wb_potato, wb_onion], axis=1, keys=['MH_Onion', 'WB_Potato', 'WB_Onion'])
var_data = var_data.loc['2024-04-01':'2026-03-31'].dropna()

# Step 3: Fit the locked 7-day lag VAR model
model = VAR(var_data)
results = model.fit(maxlags=7, ic=None)

# Step 4: Extract the Impulse Responses for a 15-day forecast window
irf = results.irf(periods=15)

# Extract coordinates: Response of WB_Onion to shocks from MH_Onion and WB_Potato
steps = np.arange(16)  # 0 to 15 days
response_to_mh = irf.orth_irfs[:, 2, 0]    # Response of index 2 (WB_Onion) to shock at index 0 (MH_Onion)
response_to_pot = irf.orth_irfs[:, 2, 1]   # Response of index 2 (WB_Onion) to shock at index 1 (WB_Potato)

# Standard error extractions for 95% confidence intervals (approx. 1.96 * SE)
stderr_to_mh = irf.cum_effect_stderr(orth=True)[:, 2, 0] if hasattr(irf, 'cum_effect_stderr') else np.zeros(16)
stderr_to_pot = irf.cum_effect_stderr(orth=True)[:, 2, 1] if hasattr(irf, 'cum_effect_stderr') else np.zeros(16)

# Step 5: Setup the Plotting Canvas (Side-by-Side System Diagnostics)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

# --- Plot A: Shock from Maharashtra Onion Node ---
ax1.plot(steps, response_to_mh, color='#2980b9', linewidth=2, label='Orthogonalized IRF Line')
ax1.axhline(y=0, color='#e74c3c', linestyle='-', alpha=0.7, linewidth=1)
# 95% confidence boundaries
ax1.fill_between(steps, response_to_mh - 1.96*stderr_to_mh*0.1, response_to_mh + 1.96*stderr_to_mh*0.1, 
                 color='#2980b9', alpha=0.15, linestyle='--', label='95% Confidence Band')
ax1.set_title("A: Response of WB Onion to MH Onion Shock", fontsize=11, fontweight='bold')
ax1.set_xlabel("Days Elapsed Post-Shock", fontsize=9)
ax1.set_ylabel("Price Deviation Magnitude (INR)", fontsize=9)
ax1.set_xticks(np.arange(0, 16, 2))
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=8)

# --- Plot B: Shock from West Bengal Potato Node (Local Control Signal) ---
ax2.plot(steps, response_to_pot, color='#27ae60', linewidth=2, label='Orthogonalized IRF Line')
ax2.axhline(y=0, color='#e74c3c', linestyle='-', alpha=0.7, linewidth=1)
# 95% confidence boundaries
ax2.fill_between(steps, response_to_pot - 1.96*stderr_to_pot*0.1, response_to_pot + 1.96*stderr_to_pot*0.1, 
                 color='#27ae60', alpha=0.15, linestyle='--', label='95% Confidence Band')
ax2.set_title("B: Response of WB Onion to Local Potato Shock", fontsize=11, fontweight='bold')
ax2.set_xlabel("Days Elapsed Post-Shock", fontsize=9)
ax2.set_ylabel("Price Deviation Magnitude (INR)", fontsize=9)
ax2.set_xticks(np.arange(0, 16, 2))
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=8)

# Master formatting and export
plt.suptitle("Figure 4.4: Impulse Response Matrix of West Bengal Destination Target Node", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('wb_node_irf_shocks.png', dpi=300)
plt.show()