import os
import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import argparse


"""
Script to visualize FAISS indices for each attribute directory. For each attribute directory in the base directory,
loads the corresponding FAISS index, reduces its dimensionality using the specified method (UMAP, PCA, or t-SNE),
and creates a separate plot file for each attribute.
"""

def load_faiss_index(index_path, pkl_path=None):
    """
    Loads vectors from a FAISS index and retrieves labels from a pickle file if available.

    Parameters:
        index_path (str): Path to the FAISS index file.
        pkl_path (str, optional): Path to the pickle file containing metadata.

    Returns:
        vectors (np.ndarray): Numpy array of vectors.
        labels (list, optional): List of labels corresponding to vectors.
    """
    try:
        index = faiss.read_index(index_path)
        n_vectors = index.ntotal
        dim = index.d
        print(f"Loaded FAISS index from {index_path} with {n_vectors} vectors of dimension {dim}.")

        # Initialize an array to hold all vectors
        vectors = np.zeros((n_vectors, dim), dtype='float32')

        # Reconstruct each vector from the index
        for i in tqdm(range(n_vectors), desc="Reconstructing vectors"):
            index.reconstruct(i, vectors[i])

        labels = None
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            # Assume labels are stored under the key 'labels'
            if isinstance(data, dict) and 'labels' in data:
                labels = data['labels']
                print(f"Retrieved {len(labels)} labels from {pkl_path}.")
            else:
                print(f"No 'labels' key found in {pkl_path}. Labels will not be used.")

        return vectors, labels

    except Exception as e:
        print(f"Error loading FAISS index from {index_path}: {e}")
        return None, None

def reduce_dimensions(vectors, method='umap'):
    """
    Reduces high-dimensional vectors to 2D using the specified method.

    Parameters:
        vectors (np.ndarray): Numpy array of vectors.
        method (str): Dimensionality reduction method ('umap', 'pca', 'tsne').

    Returns:
        reduced_vectors (np.ndarray): 2D numpy array of reduced vectors.
    """
    print(f"Reducing dimensions using {method.upper()}...")
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose from 'umap', 'pca', 'tsne'.")

    reduced_vectors = reducer.fit_transform(vectors)
    print("Dimensionality reduction complete.")
    return reduced_vectors

def plot_vectors(reduced_vectors, labels=None, attribute_name='attribute', output_dir='faiss_visualizations'):
    """
    Plots 2D vectors and saves the plot.

    Parameters:
        reduced_vectors (np.ndarray): 2D numpy array of vectors.
        labels (list, optional): List of labels for coloring.
        attribute_name (str): Name of the attribute for plot title.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    if labels is not None:
        unique_labels = list(set(labels))
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(
            x=reduced_vectors[:,0],
            y=reduced_vectors[:,1],
            hue=labels,
            palette=palette,
            legend='full',
            s=10,
            alpha=0.6
        )
        plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], s=10, alpha=0.6)
    
    plt.title(f"FAISS Index Visualization: {attribute_name}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{attribute_name}_2D_visualization.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}.")

def main():
    """
    Main function to visualize FAISS indices.
    """
    parser = argparse.ArgumentParser(description="Visualize FAISS indices in 2D.")
    parser.add_argument('--base_dir', type=str, default='faiss_index_trained',
                        help="Base directory containing FAISS index directories.")
    parser.add_argument('--output_dir', type=str, default='faiss_visualizations',
                        help="Directory to save the visualization plots.")
    parser.add_argument('--method', type=str, default='umap',
                        choices=['umap', 'pca', 'tsne'],
                        help="Dimensionality reduction method to use.")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    reduction_method = args.method

    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist. Exiting.")
        return

    # Iterate through each attribute directory
    for attr_dir in os.listdir(base_dir):
        attr_path = os.path.join(base_dir, attr_dir)
        if os.path.isdir(attr_path):
            print(f"\nProcessing attribute directory: {attr_dir}")

            index_faiss_path = os.path.join(attr_path, 'index.faiss')
            index_pkl_path = os.path.join(attr_path, 'index.pkl')

            if not os.path.exists(index_faiss_path):
                print(f"FAISS index file {index_faiss_path} not found. Skipping.")
                continue

            # Load vectors and labels
            vectors, labels = load_faiss_index(index_faiss_path, index_pkl_path)

            if vectors is None:
                print(f"Failed to load vectors for {attr_dir}. Skipping.")
                continue

            # Dimensionality reduction
            reduced_vectors = reduce_dimensions(vectors, method=reduction_method)

            # Plotting
            plot_vectors(reduced_vectors, labels, attribute_name=attr_dir, output_dir=output_dir)

    print("\nAll FAISS indices have been processed and visualized.")

if __name__ == "__main__":
    main()