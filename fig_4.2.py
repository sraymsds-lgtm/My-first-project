import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load your actual PPAC data file

df_raw = pd.read_csv(r"C:\Users\sr030\OneDrive\Desktop\SANJAY\MSDS\SEM-IV\Project\diesel_prices.csv")

# Step 2: Format, Sort, and Clean the Data Matrix
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw.set_index('Date', inplace=True)

# FIX: Sort the dates chronologically so Pandas can slice the timeline without crashing
df_raw.sort_index(inplace=True)

# Filter the dataset to lock down your strict 2-year thesis window
df_diesel = df_raw.loc['2024-04-01':'2026-03-31']

# Step 3: Setup the Plotting Canvas
fig, ax = plt.subplots(figsize=(12, 4.5))

# Plot your actual 'Kolkata' column data
ax.plot(df_diesel.index, df_diesel['Kolkata'], color='#2c3e50', linewidth=2, label='Kolkata Retail Diesel RSP')

# Step 4: Formatting Labels and Title framework for academic presentation
ax.set_title("Figure 4.2: Daily Retail Price Signals of Kolkata Diesel (2024 - 2026)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Timeline (Daily Ingested PPAC Records)", fontsize=10, labelpad=10)
ax.set_ylabel("Diesel Price (INR / Litre)", fontsize=10, labelpad=10)

# Set grid lines and rotate timeline tags
ax.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=15)

# Layout optimization
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save the finalized plot
plt.savefig('kolkata_diesel_signals.png', dpi=300)

# Display chart
plt.show()