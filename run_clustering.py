import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


class ClusterNode:
    def __init__(self, pixels_idx, depth):
        self.pixels_idx = pixels_idx
        self.depth = depth
        self.left = None
        self.right = None
        self.centroid = None


def split_cluster(node, pixels, max_depth):
    if node.depth >= max_depth or len(node.pixels_idx) < 2:
        return

    cluster_pixels = pixels[node.pixels_idx]
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(cluster_pixels)

    labels = kmeans.labels_
    node.centroid = cluster_pixels.mean(axis=0)

    idx_left = node.pixels_idx[labels == 0]
    idx_right = node.pixels_idx[labels == 1]

    node.left = ClusterNode(idx_left, node.depth + 1)
    node.right = ClusterNode(idx_right, node.depth + 1)

    split_cluster(node.left, pixels, max_depth)
    split_cluster(node.right, pixels, max_depth)


def get_clusters_by_level(root):
    from collections import defaultdict
    levels = defaultdict(list)
    queue = [(root, 0)]
    while queue:
        node, level = queue.pop(0)
        levels[level].append(node)
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))
    return levels


def generate_images(pixels, root, h, w):
    levels = get_clusters_by_level(root)
    images = []
    for level in sorted(levels.keys()):
        img = np.zeros((pixels.shape[0], 3), dtype=np.uint8)
        for node in levels[level]:
            if node.centroid is None:
                node.centroid = pixels[node.pixels_idx].mean(axis=0)
            img[node.pixels_idx] = node.centroid.astype(np.uint8)
        images.append(img.reshape(h, w, 3))
    return images


def count_leaves(node):
    if node.left is None and node.right is None:
        return 1
    leaves = 0
    if node.left:
        leaves += count_leaves(node.left)
    if node.right:
        leaves += count_leaves(node.right)
    return leaves


def run_clustering(image_path, max_depth=5, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)

    # Load and reshape image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w, c = image_np.shape
    pixels = image_np.reshape(-1, 3)
    pixels_idx = np.arange(pixels.shape[0])

    # Create and split root node
    root = ClusterNode(pixels_idx, depth=0)
    split_cluster(root, pixels, max_depth)

    # Generate images
    images = generate_images(pixels, root, h, w)

    # Save images per level
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"{output_folder}/output_depth_{i}.png")

    # Save GIF animation
    pil_images = [Image.fromarray(img) for img in images]
    gif_path = os.path.join(output_folder, "cluster_evolution.gif")
    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)

    print(f"✔️ Saved {len(images)} images to '{output_folder}'")
    print(f"✔️ GIF saved at '{gif_path}'")
    print(f"✔️ Actual clusters generated: {count_leaves(root)}")
    return {
        "images": images,
        "gif_path": gif_path,
        "leaf_count": count_leaves(root)
    }