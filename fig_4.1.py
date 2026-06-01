import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load your actual Agmarknet data file

df_raw = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\agmarknet_data.csv", skiprows=1)

# Step 2: Clean column strings and enforce numeric types
df_raw['Modal Price'] = df_raw['Modal Price'].astype(str).str.replace(',', '', regex=True)
df_raw['Modal Price'] = pd.to_numeric(df_raw['Modal Price'], errors='coerce')
df_raw.dropna(subset=['Modal Price'], inplace=True)

# Step 3: Format, Clean, and Chronologically Sort the Data Structure
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y')
df_raw.set_index('Date', inplace=True)
df_raw.sort_index(inplace=True)

# Clean up any accidental leading/trailing blank spaces from your text columns
df_raw['State'] = df_raw['State'].astype(str).str.strip()
df_raw['Commodity'] = df_raw['Commodity'].astype(str).str.strip()

# Step 4: Extract and Filter using your exact CSV spelling strings
# This will now perfectly match 'Maharashtra', 'West Bengal', 'Onion', and 'Potato'
mh_onion_data = df_raw[(df_raw['State'] == 'Maharashtra') & (df_raw['Commodity'] == 'Onion')]
wb_onion_data = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Onion')]
wb_potato_data = df_raw[(df_raw['State'] == 'West Bengal') & (df_raw['Commodity'] == 'Potato')]

# Group by Date and calculate the mean daily price across the state
mh_onion_daily = mh_onion_data.groupby('Date')['Modal Price'].mean()
wb_onion_daily = wb_onion_data.groupby('Date')['Modal Price'].mean()
wb_potato_daily = wb_potato_data.groupby('Date')['Modal Price'].mean()

# Slice each state series down to your 2-year thesis window
mh_onion_series = mh_onion_daily.loc['2024-04-01':'2026-03-31']
wb_onion_series = wb_onion_daily.loc['2024-04-01':'2026-03-31']
wb_potato_series = wb_potato_daily.loc['2024-04-01':'2026-03-31']

# Step 5: Setup the Plotting Canvas
fig, ax = plt.subplots(figsize=(12, 6))

# Plot lines dynamically if data exists
if not mh_onion_series.empty:
    ax.plot(mh_onion_series.index, mh_onion_series, color='#2980b9', linestyle='-', linewidth=1.5, label='MH Onion (State Mean Modal)')
if not wb_onion_series.empty:
    ax.plot(wb_onion_series.index, wb_onion_series, color='#e74c3c', linestyle='-', linewidth=1.8, label='WB Onion (State Mean Modal)')
if not wb_potato_series.empty:
    ax.plot(wb_potato_series.index, wb_potato_series, color='#7f8c8d', linestyle=':', linewidth=1.2, label='WB Potato (Baseline Control Mean)')

# Formatting axes and text metadata for publication
ax.set_title("Figure 4.1: Time-Series Vector of Daily State Prices (April 2024 - March 2026)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Timeline (Daily Data Ingestion)", fontsize=10, labelpad=10)
ax.set_ylabel("Price Index (INR / Quintal)", fontsize=10, labelpad=10)

# Apply a clean grid background
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#b5b5b5')

# Clean layout configuration
plt.xticks(rotation=15)
plt.tight_layout()

# Save the finalized visual layout directly to your thesis directory
plt.savefig('time_series_prices.png', dpi=300)

# Render Chart
plt.show()