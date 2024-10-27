import pickle
import os
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from bidi.algorithm import get_display 

def plot_vectors(pickle_file, show_plot=True, image_path=None):
    # Load vectors from a pickle file
    with open(pickle_file, 'rb') as f:
        vectors_dict = pickle.load(f)

    vectors = list(vectors_dict.values())
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    labels = []
    groups = []
    names = []
    for key in vectors_dict.keys():
        parts = key.split('_')
        name = parts[0]
        names.append(name)
        label = parts[1]
        labels.append(label)
        if len(parts) > 2:
            group = parts[2]
            groups.append(group)
        else:
            groups.append(None)

    # Plot the 2D vectors using seaborn
    if len(set(labels)) > 1:
        rtl_labels = [get_display(label) for label in labels]
        if any(groups) and len(set(groups)) > 1:
            rtl_groups = [get_display(group) if group else None for group in groups]
            plot = sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hue=rtl_labels, style=rtl_groups, palette='hls')
        else:
            plot = sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hue=rtl_labels, palette='hls')
    else:
        plot = sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1])

    filename_without_extension, _ = os.path.splitext(os.path.basename(pickle_file))
    rtl_title = get_display(filename_without_extension)
    plt.title(rtl_title)

    # Add text labels for each point
    for i, name in enumerate(names):
        rtl_name = get_display(name)
        plot.text(vectors_2d[i, 0], vectors_2d[i, 1], rtl_name)

    if show_plot:
        plt.show()
        plt.close()
    if image_path:
        plt.savefig(image_path)
        plt.close()
    else:
        fig = plot.get_figure()
        plt.close(fig)
        return fig


def plot_vectors_to_pdf(vectors, labels, classifier_name, metrics, pdf_pages):
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Plot the 2D vectors using seaborn, experiment with color palettes
    fig, ax = plt.subplots()
    sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hue=labels, palette='hls', ax=ax)  # Example palette change
    ax.set_title(f'2D Vectors for {classifier_name}')

    # Display the metrics
    metrics_text = f"Precision: {metrics['Precision']:.2f}\nRecall: {metrics['Recall']:.2f}\nF1 Score: {metrics['F1 Score']:.2f}\nAccuracy: {metrics['Accuracy']:.2f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Adjust marker size and transparency if needed
    # ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=5, alpha=0.7)  # Example adjustments

    pdf_pages.savefig(fig)
    plt.close(fig)


# import pickle
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.manifold import TSNE

# def plot_vectors(pickle_file):
#     # Load vectors from pickle file
#     with open(pickle_file, 'rb') as f:
#         vectors = pickle.load(f)

#     # Separate names and vectors
#     names = list(vectors.keys())
#     vectors = list(vectors.values())

#     # Reduce dimensionality to 3D using t-SNE
#     tsne = TSNE(n_components=3)

#     # Convert list of lists to 2D NumPy array
#     vectors = np.array(vectors)

#     # Assuming that vectors is a list of vectors
#     n_samples = len(vectors)

#     # Set perplexity to a value less than n_samples
#     perplexity = n_samples - 1 if n_samples > 1 else 1

#     tsne = TSNE(n_components=3, perplexity=perplexity)
#     vectors_3d = tsne.fit_transform(vectors)    # Create a DataFrame for the 3D vectors
#     df = pd.DataFrame(vectors_3d, columns=['x', 'y', 'z'])
#     df['name'] = names

#     # Plot the 3D vectors using Plotly
#     fig = px.scatter_3d(df, x='x', y='y', z='z', hover_data=['name'])
#     fig.show()