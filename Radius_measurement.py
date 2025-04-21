import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import imageio
import os
import csv
from scipy.optimize import least_squares

def convert_im_to_point_cloud(im, thesh):
    """Convert image to point cloud."""
    coordinates = np.where(im >= thesh)
    coordinates = np.array(coordinates).T
    return coordinates

def calculate_centroids(data_points, labels):
    """Calculate centroids of each cluster."""
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = data_points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def fit_circle(centroids):
    """Fit a circle to the centroids using least squares method."""
    x_m, y_m = np.mean(centroids, axis=0)
    initial_guess = [x_m, y_m, np.mean(np.linalg.norm(centroids - [x_m, y_m], axis=1))]

    def residuals(params, points):
        xc, yc, r = params
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r

    result = least_squares(residuals, initial_guess, args=(centroids,))
    xc, yc, r = result.x
    return xc, yc, r

def visualize_and_save(data_points, labels, image, image_name, save_path, circle_center, circle_radius):
    """Visualize clusters and fitted circle, save to file."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot original clusters
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue  # Skip noise points
        cluster_points = data_points[labels == k]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], color=col, s=2)

    # Draw the fitted circle
    circle = plt.Circle((circle_center[1], circle_center[0]), circle_radius, color='blue', fill=False, linewidth=1.5, linestyle='--')
    plt.gca().add_artist(circle)

    plt.gca().invert_yaxis()
    plt.title(f'Fitted Circle - {image_name}')
    
    save_file = os.path.join(save_path, f'{image_name}_visualized.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Visualization saved to {save_file}")

def process_images_in_folder(folder_path, save_path, threshold=70):
    """Batch process all images in the folder and save results to CSV."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    csv_file = os.path.join(save_path, 'fitted_circle_data.csv')
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Only include image name, fitted circle radius, and diameter
        csv_writer.writerow(['Image Name', 'Fitted Circle Radius', 'Fitted Circle Diameter'])

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                image = imageio.imread(image_path)
                print(f"Processing image: {image_file}")
                data_points = convert_im_to_point_cloud(image, threshold)

                # Perform clustering
                db = DBSCAN(eps=1, min_samples=1).fit(data_points)
                labels = db.labels_

                # Calculate original centroids
                centroids = calculate_centroids(data_points, labels)
                
                # Fit a circle to the centroids
                circle_center_x, circle_center_y, circle_radius = fit_circle(centroids)
                circle_center = (circle_center_x, circle_center_y)
                circle_diameter = 2 * circle_radius  # Calculate diameter

                # Write image name, circle radius, and diameter to CSV
                csv_writer.writerow([image_file, circle_radius, circle_diameter])

                # Visualize and save results
                visualize_and_save(data_points, labels, image, image_file, save_path, circle_center, circle_radius)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")


# Paths to your images and save location
folder_path = 'YOUR_PATH'    
save_path = 'OUTPUT_PATH'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
process_images_in_folder(folder_path, save_path)
