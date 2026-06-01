import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Step 1: Load and clean your actual Agmarknet data file
# ag_file_name = 'your_agmarknet_data.csv'
df_raw = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\agmarknet_data.csv", skiprows=1)

# Clean, format, and align your structural time-series data
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y')
df_raw.set_index('Date', inplace=True)
df_raw.sort_index(inplace=True)
df_raw['State'] = df_raw['State'].astype(str).str.strip()
df_raw['Commodity'] = df_raw['Commodity'].astype(str).str.strip()
df_raw['Modal Price'] = df_raw['Modal Price'].astype(str).str.replace(',', '', regex=True)
df_raw['Modal Price'] = pd.to_numeric(df_raw['Modal Price'], errors='coerce')
df_raw.dropna(subset=['Modal Price'], inplace=True)

# Step 2: Build the parallel data matrix for your VAR system nodes
mh_onion = df_raw[(df_raw['State'] == 'Maharashtra') & (df_raw['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_onion = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Onion')].groupby('Date')['Modal Price'].mean()
wb_potato = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Potato')].groupby('Date')['Modal Price'].mean()

# Merge into a single continuous DataFrame across your 2-year thesis window
var_data = pd.concat([mh_onion, wb_onion, wb_potato], axis=1, keys=['MH_Onion', 'WB_Onion', 'WB_Potato'])
var_data = var_data.loc['2024-04-01':'2026-03-31'].dropna()

# Step 3: Fit your actual VAR model using your locked 7-day optimal lag length
model = VAR(var_data)
results = model.fit(maxlags=7, ic=None)  # FIXED: Updated to match your true 7-day lag specification

# Extract your model's actual inverse characteristic roots
actual_roots = results.roots
inverse_roots = 1.0 / actual_roots  

real_parts = np.real(inverse_roots)
imag_parts = np.imag(inverse_roots)

# Step 4: Setup the Plotting Canvas (The Unit Circle)
theta = np.linspace(0, 2 * np.pi, 150)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

fig, ax = plt.subplots(figsize=(6.5, 6.5))

# Draw the geometric layout boundaries
ax.plot(x_circle, y_circle, color='#2c3e50', linewidth=1.5, label='Unit Circle Boundary ($|z| = 1$)')
ax.axhline(y=0, color='#7f8c8d', linestyle=':', linewidth=1)
ax.axvline(x=0, color='#7f8c8d', linestyle=':', linewidth=1)

# Plot YOUR model's actual roots calculated from the 7-day lag system
ax.scatter(real_parts, imag_parts, color='#e74c3c', marker='o', s=45, edgecolors='#c0392b', zorder=5, label='Actual Inverse Roots ($p=7$)')

# Step 5: Formatting Typography for Thesis Submission
ax.set_title("Figure 4.3: Inverse Roots of AR Characteristic Polynomial", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Real Structural Axis", fontsize=9, labelpad=8)
ax.set_ylabel("Imaginary Structural Axis", fontsize=9, labelpad=8)

# Format viewing bounds and aspect ratio
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower left', frameon=True, fontsize=9)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('var_stability_roots.png', dpi=300)
plt.show()