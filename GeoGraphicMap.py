import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Load data into GeoDataFrame
# Change the geodata file for each clustering algorithm
data = pd.read_csv('/content/DBSCAN_us_geodata.csv')
data = data.dropna(subset=['geometry'])
us_states_gdf = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries.from_wkt(data['geometry']))
#c2c2f0
# Define the color map for clusters
# If there is more than one cluster add one more colour
cmap = ListedColormap(['#99ff99', '#ff9999'])
default_color = 'lightgrey'  # Grey for undefined clusters

# Create the main plot
fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off')

# Plot contiguous US states
contiguous_states = us_states_gdf[~us_states_gdf['STUSPS'].isin(['AK', 'HI'])]
contiguous_states.plot(
    column='Cluster',
    cmap=cmap,
    missing_kwds={"color": default_color},
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8'
)

# Alaska inset
ak_ax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
ak_ax.axis('off')
alaska_polygon = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
alaska_gdf = us_states_gdf[us_states_gdf['STUSPS'] == 'AK']
alaska_gdf.clip(alaska_polygon).plot(
    column='Cluster',
    cmap=cmap,
    missing_kwds={"color": default_color},
    linewidth=0.8,
    ax=ak_ax,
    edgecolor='0.8'
)

# Hawaii inset
hi_ax = fig.add_axes([0.28, 0.20, 0.1, 0.1])
hi_ax.axis('off')
hawaii_polygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
hawaii_gdf = us_states_gdf[us_states_gdf['STUSPS'] == 'HI']
hawaii_gdf.clip(hawaii_polygon).plot(
    column='Cluster',
    cmap=cmap,
    missing_kwds={"color": default_color},
    linewidth=0.8,
    ax=hi_ax,
    edgecolor='0.8'
)

# Add a legend on the right
# If there is more than one cluster or less we either add/delete one more colour, change the legend corresponding to the cluster
legend_labels = ['-1', '0', 'Undefined']
legend_colors = ['#99ff99', '#ff9999',  default_color]

for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
    ax.text(
        1.05, 0.95 - i * 0.05, label,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')
    )

# Change the title corresponding to the algorithm used
fig.suptitle('US States DBSCAN Cluster Visualization', fontsize=20, y=0.95)

# Show the plot
plt.show()
