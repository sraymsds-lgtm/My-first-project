import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString

# Step 1: Simplified coordinates of India's main land boundary for a clean background canvas
# This bypasses all projection/geopandas dependency issues entirely
india_polygon_coords = [
    (68.1, 23.8), (73.5, 25.8), (74.2, 31.0), (74.8, 34.5), (78.0, 35.1), 
    (79.0, 31.0), (81.0, 30.2), (88.0, 27.8), (91.5, 28.0), (97.4, 28.3), 
    (94.5, 23.5), (92.2, 21.0), (88.3, 21.5), (85.0, 19.5), (80.0, 16.0), 
    (79.8, 10.0), (77.5, 8.1), (76.5, 10.0), (74.5, 15.0), (72.8, 21.0), 
    (68.5, 22.0), (68.1, 23.8) # Closing the polygon
]

# Step 2: Define Market Node Coordinates (Longitude, Latitude)
nodes = {
    'Nashik (Source Node)': (73.7898, 19.9975),
    'Kolkata (Destination Node)': (88.3629, 22.5744)
}

# Step 3: Setup Plotting Canvas
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the bounding canvas of India
poly_x, poly_y = zip(*india_polygon_coords)
ax.fill(poly_x, poly_y, color='#f2f2f2', edgecolor='#b5b5b5', linewidth=1.5, label='India Boundary Frame')

# Step 4: Manually plot the network flow line (The 2,000 km Supply Vector)
corridor_x = [nodes['Nashik (Source Node)'][0], nodes['Kolkata (Destination Node)'][0]]
corridor_y = [nodes['Nashik (Source Node)'][1], nodes['Kolkata (Destination Node)'][1]]
ax.plot(corridor_x, corridor_y, color='#e74c3c', linestyle='--', linewidth=2.5, label='Supply Corridor (2,000 km)')

# Step 5: Plot and Annotate the Nodes
for name, coord in nodes.items():
    # Draw points
    ax.scatter(coord[0], coord[1], color='#2c3e50', s=120, zorder=5)
    
    # Text alignments
    ax.annotate(
        text=name, 
        xy=coord,
        xytext=(8, 5), 
        textcoords="offset points", 
        fontsize=10, 
        fontweight='bold',
        color='#2c3e50'
    )

# Formatting for academic presentation layout
ax.set_title("Figure 1.1: Spatial Scope of the Nashik-Kolkata Supply Corridor", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Longitude", fontsize=10, labelpad=10)
ax.set_ylabel("Latitude", fontsize=10, labelpad=10)
ax.grid(True, linestyle=':', alpha=0.5)

# Crop canvas coordinates specifically to focus on the Indian subcontinent matrix
ax.set_xlim(65, 98)
ax.set_ylim(6, 38)

# Show layout
plt.legend(loc='upper left')
plt.tight_layout()

# Uncomment below to save directly for your thesis document structure
plt.savefig('nashik_kolkata_corridor.png', dpi=300)
plt.show()