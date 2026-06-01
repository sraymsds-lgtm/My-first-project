import matplotlib.pyplot as plt

# Setup the canvas size
fig, ax = plt.subplots(figsize=(10, 11))

# Define box styling properties (Academic Blue/Grey palette)
box_style = dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#2c3e50", lw=1.5)
start_style = dict(boxstyle="round,pad=0.6", fc="#e8f4f8", ec="#2980b9", lw=2)
output_style = dict(boxstyle="round,pad=0.6", fc="#fcf3cf", ec="#f39c12", lw=2)

# Define step text and positions (X, Y centering)
steps = [
    ("Stage 1: Secondary Data Ingestion\n• Sourced daily Onion/Potato prices (Agmarknet)\n• Sourced daily Retail Diesel prices (PPAC)", 0.5, 0.92, start_style),
    ("Stage 2: Data Preprocessing Matrix\n• Chronological alignment (Daily frequency)\n• Linear Imputation for market holiday gaps", 0.5, 0.78, box_style),
    ("Stage 3: Statistical Calibration (ADF Test)\n• Checked variables for Unit Roots\n• Converted I(0) raw data to stationary I(1) fields", 0.5, 0.64, box_style),
    ("Stage 4: System Latency Calculation\n• Evaluated Akaike Information Criterion (AIC)\n• Calibrated look-back window to optimal 7-day lag", 0.5, 0.50, box_style),
    ("Stage 5: Multi-Variable VAR Modeling\n• Solved structural linear equations concurrently\n• Vector Matrix: [MH Onion, WB Onion, Potato, Diesel]", 0.5, 0.36, box_style),
    ("Output Node A:\nGranger Causality\n(Signal Direction & Unidirectional Flow)", 0.25, 0.18, output_style),
    ("Output Node B:\nVariance Decomposition (FEVD)\n(Quantified 1.6% Fuel Impact Factor)", 0.75, 0.18, output_style)
]

# Draw the pipeline boxes and text strings
for text, x, y, style in steps:
    ax.text(x, y, text, ha="center", va="center", fontsize=10, bbox=style, color="#2c3e50")

# Draw the directional vector arrows linking the nodes
arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=1.5, mutation_scale=15)

# Vertical main pipeline arrows
ax.annotate("", xy=(0.5, 0.83), xytext=(0.5, 0.87), arrowprops=arrow_props)
ax.annotate("", xy=(0.5, 0.69), xytext=(0.5, 0.73), arrowprops=arrow_props)
ax.annotate("", xy=(0.5, 0.55), xytext=(0.5, 0.59), arrowprops=arrow_props)
ax.annotate("", xy=(0.5, 0.41), xytext=(0.5, 0.45), arrowprops=arrow_props)

# Branched split to the final analytical outputs
ax.annotate("", xy=(0.25, 0.23), xytext=(0.5, 0.31), arrowprops=arrow_props)
ax.annotate("", xy=(0.75, 0.23), xytext=(0.5, 0.31), arrowprops=arrow_props)

# Final clean up configurations
ax.set_title("Figure 3.1: Technical Processing Pipeline and VAR Architecture Schematic", fontsize=13, fontweight='bold', color="#2c3e50", pad=20)
ax.axis('off') # Hide axis values for a graphical presentation layout
plt.tight_layout()

# Uncomment to save image directly:
plt.savefig('system_pipeline_architecture.png', dpi=300)
plt.show()