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
import math

"""
Script to visualize all FAISS indices in one PNG file. For each attribute directory in the base directory,
loads the corresponding FAISS index, reduces its dimensionality using the specified method (UMAP, PCA, or t-SNE),
and creates a combined visualization showing all indices in a single plot file.
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

def plot_combined_vectors(all_reduced_vectors, all_labels, all_attribute_names, output_path):
    """
    Plots all 2D vectors in a single combined figure with subplots arranged in two columns.

    Parameters:
        all_reduced_vectors (list of np.ndarray): List of 2D numpy arrays of vectors.
        all_labels (list of list, optional): List containing label lists for each attribute.
        all_attribute_names (list of str): List of attribute names.
        output_path (str): Path to save the combined plot.
    """
    n_attrs = len(all_attribute_names)
    if n_attrs == 0:
        print("No attributes to plot. Exiting plotting function.")
        return

    # Determine number of rows needed for two columns
    n_cols = 2
    n_rows = math.ceil(n_attrs / n_cols)

    # Define subplot dimensions to maintain 5:3 aspect ratio
    subplot_width = 5  # inches
    subplot_height = 3  # inches

    # Calculate total figure size
    total_width = n_cols * subplot_width + 1  # additional space for spacing/margins
    total_height = n_rows * subplot_height + 1  # additional space for spacing/margins

    plt.figure(figsize=(total_width, total_height))

    for idx, (reduced, labels, attr) in enumerate(zip(all_reduced_vectors, all_labels, all_attribute_names)):
        row = idx // n_cols
        col = idx % n_cols
        subplot_idx = idx + 1
        ax = plt.subplot(n_rows, n_cols, subplot_idx)
        
        if labels is not None:
            unique_labels = sorted(list(set(labels)))
            palette = sns.color_palette("hsv", len(unique_labels))
            sns.scatterplot(
                x=reduced[:,0],
                y=reduced[:,1],
                hue=labels,
                palette=palette,
                legend=False,  # Legends handled separately
                s=10,
                alpha=0.8,  # Increased alpha for darker dots
                ax=ax
            )
            # Add legend only once, preferably for the first subplot
            if idx == 0:
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, unique_labels, title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            sns.scatterplot(
                x=reduced[:,0],
                y=reduced[:,1],
                color='steelblue',
                s=10,
                alpha=0.8,  # Increased alpha for darker dots
                ax=ax
            )
        
        # Extract just the attribute name without any path components
        attr_name = os.path.basename(attr)
        ax.set_title(attr_name, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Remove axes labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Only show bottom and left spines
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure with adequate DPI for clarity
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to {output_path}.")

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
    parser.add_argument('--output_file', type=str, default='combined_visualization.png',
                        help="Filename for the combined visualization image.")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    reduction_method = args.method
    output_file = args.output_file

    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_reduced_vectors = []
    all_labels = []
    all_attribute_names = []

    # Iterate through each attribute directory
    for attr_dir in sorted(os.listdir(base_dir)):
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

            # Collect data for combined plotting
            all_reduced_vectors.append(reduced_vectors)
            all_labels.append(labels)
            all_attribute_names.append(attr_dir)

    if not all_attribute_names:
        print("No valid FAISS indices found to visualize. Exiting.")
        return

    # Plot all collected data in a combined figure
    combined_plot_path = os.path.join(output_dir, output_file)
    plot_combined_vectors(all_reduced_vectors, all_labels, all_attribute_names, combined_plot_path)

    print("\nAll FAISS indices have been processed and visualized.")

if __name__ == "__main__":
    main()