import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend

# Set input and output directories
input_dir = "YOUR_PATH"
output_dir = "OUTPUT_PATH"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define parameters
threshold = 80  # Pixel intensity threshold
eps1, min_samples1 = 25, 15  # Parameters for the first DBSCAN
eps2, min_samples2 = 1, 1    # Parameters for the second DBSCAN

# Store analysis results for all images
results = []
skip_log = []

# Iterate through all images in the input directory
for file_name in os.listdir(input_dir):
    if not file_name.lower().endswith((".jpg", ".png", ".tif")):
        continue

    image_path = os.path.join(input_dir, file_name)
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    print(f"Processing {file_name}, Image shape: {image_array.shape}")

    # Extract non-background pixels
    data_points = np.column_stack(np.where(image_array > threshold))
    print("Number of data points:", data_points.shape[0])

    if data_points.shape[0] == 0:
        print(f"Skipped {file_name} - no valid data points above threshold.")
        skip_log.append(file_name)
        continue

    # First round of DBSCAN clustering
    db1 = DBSCAN(eps=eps1, min_samples=min_samples1).fit(data_points)
    labels1 = db1.labels_

    # Store the first round clustering results
    clusters = {}
    for k in set(labels1):
        clusters[k] = data_points[labels1 == k]

    # Second round of DBSCAN clustering
    final_clusters = {}
    npc_centroids = {}

    for k, points in clusters.items():
        if k == -1:
            continue

        db2 = DBSCAN(eps=eps2, min_samples=min_samples2).fit(points)
        labels2 = db2.labels_

        nup_centroids = []
        for k2 in set(labels2):
            if k2 == -1:
                continue
            xy = points[labels2 == k2]
            final_clusters[f"{file_name}_{k}_{k2}"] = xy
            nup_centroids.append(np.mean(xy, axis=0))

        if len(nup_centroids) >= 2:
            npc_centroids[k] = np.mean(nup_centroids, axis=0)

    # Generate NPC centroid connection image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_array, cmap='gray')

    for k, centroid in npc_centroids.items():
        plt.plot(centroid[1], centroid[0], 'ro', label='NPC Centroid')
        for nup_centroid in nup_centroids:
            plt.plot(nup_centroid[1], nup_centroid[0], 'k+', markersize=10, label='NUP Centroid')
            plt.plot([centroid[1], nup_centroid[1]], [centroid[0], nup_centroid[0]], 'b--')

    plt.gca().invert_yaxis()
    plt.title(f'Centroid Connections for {file_name}')
    plt.savefig(os.path.join(output_dir, f"npc_lines_{file_name}.png"), bbox_inches='tight')
    plt.close()

    # Calculate distances and statistics between NPC and NUPs
    nup_centroids = np.atleast_2d(nup_centroids)
    for k, npc_centroid in npc_centroids.items():
        distances = np.linalg.norm(nup_centroids - np.atleast_2d(npc_centroid), axis=1)
        std_dev = np.std(distances)
        variance = np.var(distances)
        coeff_variation = std_dev / np.mean(distances)

        for distance in distances:
            results.append([file_name, k, distance, std_dev, variance, coeff_variation])

# Save statistical results to CSV
results_df = pd.DataFrame(results, columns=['File', 'NPC_ID', 'Distance', 'Standard Deviation', 'Variance', 'Coefficient of Variation'])
csv_output_path = os.path.join(output_dir, 'npc_nup_distances.csv')
results_df.to_csv(csv_output_path, index=False)
print(f"Results saved to {csv_output_path}")

# Save skipped image log
if skip_log:
    skip_log_path = os.path.join(output_dir, 'skipped_images.txt')
    with open(skip_log_path, 'w') as f:
        for fname in skip_log:
            f.write(fname + '\n')
    print(f"Skipped files logged in {skip_log_path}")

