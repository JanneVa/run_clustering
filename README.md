# Hierarchical K-means Image Clustering

This project implements a recursive K-means clustering algorithm for image compression, using a binary tree structure with maximum depth control.

## Features

- K-means with \(k=2\) applied recursively
- Adaptive clustering by region
- Visual output per depth level
- GIF animation of clustering evolution

## Usage

```python
from run_clustering import run_clustering
run_clustering("sample/monet1.png", max_depth=6)
