import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# 1. Read the image and convert it to grayscale
image_path = 'YOUR_PATH'
image = Image.open(image_path).convert('L')
image_array = np.array(image)
print("Image shape:", image_array.shape)

# 2. Extract data points (non-background pixels)
threshold = 70
data_points = np.column_stack(np.where(image_array > threshold))
print("Number of data points:", data_points.shape[0])

# First round of DBSCAN clustering
db1 = DBSCAN(eps=15, min_samples=25).fit(data_points)
labels1 = db1.labels_
print("Unique labels (first DBSCAN):", set(labels1))

# Store the first round DBSCAN clustering results
clusters = {}
for k in set(labels1):   
    clusters[k] = data_points[labels1 == k]

# Visualize the first round DBSCAN clustering results
plt.figure(figsize=(8, 6))
unique_labels1 = set(labels1)
colors1 = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels1)))

for k, col in zip(unique_labels1, colors1):
    if k == -1:
        col = [0, 0, 0, 1]  # Mark noise points in black

    class_member_mask = (labels1 == k)
    xy = data_points[class_member_mask]
    
    plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)

    # Draw the convex hull
    if len(xy) >= 3 and np.unique(xy[:, 0]).shape[0] > 1 and np.unique(xy[:, 1]).shape[0] > 1:
        hull = ConvexHull(xy)
        for simplex in hull.simplices:
            plt.plot(xy[simplex, 1], xy[simplex, 0], 'r-', linewidth=1.5)

plt.gca().invert_yaxis()
plt.title('First DBSCAN Clustering Results (NPC)')

# Ensure output directory exists
output_dir = 'OUTPUT_PATH'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the first round clustering result image
plt.savefig(os.path.join(output_dir, 'first_dbscan_results.png'))
plt.show()

# Second round DBSCAN analysis for each cluster
final_clusters = {}
nup_centroids_per_npc = {}
colors2 = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))

plt.figure(figsize=(8, 6))
for k, col in zip(clusters.keys(), colors2):
    if k == -1:
        continue  # Skip noise points

    points = clusters[k]

    # Second round of DBSCAN
    db2 = DBSCAN(eps=1, min_samples=1).fit(points)
    labels2 = db2.labels_

    nup_centroids_per_npc[k] = []
    for k2 in set(labels2):
        if k2 == -1:
            continue  # Skip noise points

        class_member_mask = (labels2 == k2)
        xy = points[class_member_mask]

        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
        
        # Draw the convex hull
        if len(xy) >= 3 and np.unique(xy[:, 0]).shape[0] > 1 and np.unique(xy[:, 1]).shape[0] > 1:
            hull = ConvexHull(xy)
            for simplex in hull.simplices:
                plt.plot(xy[simplex, 1], xy[simplex, 0], 'r-', linewidth=1.5)

        # Store the result of second DBSCAN
        final_clusters[f"{k}_{k2}"] = xy

        # Calculate NUP centroid
        nup_centroid = np.mean(xy, axis=0)
        nup_centroids_per_npc[k].append(nup_centroid)

plt.gca().invert_yaxis()
plt.title('Second DBSCAN Clustering Results (NUP)')

# Save second round clustering result image
plt.savefig(os.path.join(output_dir, 'second_dbscan_results.png'))
plt.show()

# Filter NPCs with fewer than 2 NUP clusters
filtered_nup_centroids_per_npc = {k: v for k, v in nup_centroids_per_npc.items() if len(v) >= 2}

# Calculate centroid of each NPC (based on NUP centroids)
npc_centroids = {}
npc_original_centroids = {}  # Store NPC centroid from first DBSCAN

for k, nup_centroids in filtered_nup_centroids_per_npc.items():
    npc_centroids[k] = np.mean(nup_centroids, axis=0)

    # Calculate NPC centroid from first DBSCAN
    original_points = clusters[k]
    npc_original_centroids[k] = np.mean(original_points, axis=0)

# Create output directory for centroid connection images
line_output_dir = os.path.join(output_dir, 'lines')
if not os.path.exists(line_output_dir):
    os.makedirs(line_output_dir)

# Create output directory for cropped NPC images
patch_output_dir = os.path.join(output_dir, 'patches')
if not os.path.exists(patch_output_dir):
    os.makedirs(patch_output_dir)

# Crop and save image patches centered on each NPC centroid
patch_size = 50  # Patch size: patch_size x patch_size

for k, centroid in npc_centroids.items():
    y, x = int(centroid[0]), int(centroid[1])
    y_min = max(y - patch_size // 2, 0)
    y_max = min(y + patch_size // 2, image_array.shape[0])
    x_min = max(x - patch_size // 2, 0)
    x_max = min(x + patch_size // 2, image_array.shape[1])
    
    patch = image_array[y_min:y_max, x_min:x_max]
    patch_image = Image.fromarray(patch)
    patch_image.save(os.path.join(patch_output_dir, f"npc_patch_{k}.png"))

    # Draw centroid connection lines
    plt.figure(figsize=(8, 6))
    plt.imshow(patch, cmap='gray')
    
    # Plot NUP centroids
    for nup_centroid in filtered_nup_centroids_per_npc[k]:
        plt.plot(nup_centroid[1] - x_min, nup_centroid[0] - y_min, 'k+', markersize=10, label='NUP Centroid')

    # Plot NPC centroid
    plt.plot(centroid[1] - x_min, centroid[0] - y_min, 'ro', label='NPC Centroid')

    # Draw connection lines
    for nup_centroid in filtered_nup_centroids_per_npc[k]:
        plt.plot([centroid[1] - x_min, nup_centroid[1] - x_min], [centroid[0] - y_min, nup_centroid[0] - y_min], 'b--')

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.gca().invert_yaxis()
    plt.title(f'Centroid Connections for NPC {k}')
    plt.savefig(os.path.join(line_output_dir, f"npc_lines_{k}.png"))
    plt.close()

# Compute centroid distances and statistics, then save as CSV
results = []
for k, nup_centroids in filtered_nup_centroids_per_npc.items():
    npc_centroid = npc_centroids[k]
    npc_original_centroid = npc_original_centroids[k]

    distances = np.linalg.norm(nup_centroids - npc_centroid, axis=1)
    std_dev = np.std(distances)
    variance = np.var(distances)
    coeff_variation = std_dev / np.mean(distances)
    
    for distance in distances:
        results.append([
            k, distance, std_dev, variance, coeff_variation,
            npc_original_centroid[1], npc_original_centroid[0]  # Append NPC centroid (x, y)
        ])

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=[
    'NPC_ID', 'Distance', 'Standard Deviation', 'Variance', 'Coefficient of Variation',
    'NPC_Centroid_X', 'NPC_Centroid_Y'  # Ensure 7 columns
])

csv_output_path = os.path.join(output_dir, 'npc_nup_distances.csv')
results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")


