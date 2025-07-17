from run_clustering import run_clustering

def main():
    # Parámetros de entrada
    image_path = "IC/monet1.png"
    max_depth = 5

    print("🔍 Starting hierarchical K-means clustering...")
    result = run_clustering(image_path=image_path, max_depth=max_depth)

    print("\n Summary:")
    print(f"  ↳ Image path: {image_path}")
    print(f"  ↳ Max depth: {max_depth}")
    print(f"  ↳ Total clusters generated: {result['leaf_count']}")
    print(f"  ↳ GIF saved at: {result['gif_path']}")
    print("Clustering complete.")

if __name__ == "__main__":
    main()