import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use Seaborn's default theme with minimal grid lines.
sns.set_theme(style="white")  # use "white" to have a cleaner background

def sum_matrices(query, parglmo):
    """
    Sum matrices given a query list.
    
    Each element of the query list is either:
      - an int (index into parglmo['factors']),
      - or a list of ints (to select an interaction in parglmo['interactions']).
    
    Parameters:
      query (list): e.g. [0, 1, [1, 2]]
      parglmo (dict): Dictionary with keys 'factors' and 'interactions'.
    
    Returns:
      Tensor: The sum of the matrices.
    """
    total = None
    for item in query:
        if isinstance(item, int):
            mat = parglmo['factors'][item]['matrix']
        elif isinstance(item, list):
            mat = None
            for inter in parglmo['interactions']:
                if inter['factors'] == item:
                    mat = inter['matrix']
                    break
            if mat is None:
                raise ValueError(f"Interaction with factors {item} not found")
        else:
            raise TypeError("Each item in the query must be an int or a list of ints.")
        if total is None:
            total = mat.clone()
        else:
            total = total + mat
    return total

def asca(parglmo, query, residuals=None):
    """
    Computes the ASCA model reconstructed matrix by summing specified matrices and adding the residual matrix.
    Also computes the percentage of total variation explained via a truncated SVD.
    
    If residuals is not provided, it defaults to parglmo['residuals'].
    
    Parameters:
      parglmo (dict): Contains keys 'factors', 'interactions', and 'residuals'.
      query (list): e.g. [0, 1, [1, 2]]
      residuals (Tensor, optional): Defaults to parglmo['residuals'].
    
    Returns:
      dict: Contains:
            - 'scores': The scores matrix.
            - 'loadings': The loadings matrix.
            - 'query': The query used.
            - 'percent_variation': Tensor with percent variation for each singular value.
    """
    if residuals is None:
        residuals = parglmo.get('residuals')
        if residuals is None:
            raise ValueError("Residual matrix not provided and not found in parglmo.")
    model_matrix = sum_matrices(query, parglmo)
    rank = torch.linalg.matrix_rank(model_matrix).item()
    U, S, V = torch.svd_lowrank(model_matrix, q=rank)
    loadings = V
    scores = torch.matmul(model_matrix + residuals, loadings)
    total_variation = torch.sum(S ** 2)
    percent_variation = (S ** 2) / total_variation * 100
    return {
        'scores': scores, 
        'loadings': loadings, 
        'query': query,
        'percent_variation': percent_variation
    }

def get_numeric_categories(F, query):
    """
    For each sample (row) in F (n_samples x n_factors), generate a numeric category string based on the query.
    
    For each query element:
      - If an int, output its value as a string.
      - If a list/tuple (an interaction), output the values joined by "/".
    
    The parts are joined with ", " to form the overall category.
    
    Returns:
      List of strings (one per sample).
    """
    n_samples = F.shape[0]
    categories = []
    for i in range(n_samples):
        parts = []
        for q in query:
            if isinstance(q, int):
                parts.append(str(F[i, q].item()))
            elif isinstance(q, (list, tuple)):
                vals = [str(F[i, j].item()) for j in q]
                parts.append("/".join(vals))
            else:
                raise ValueError("Each query element must be an int or a list/tuple of ints.")
        categories.append(", ".join(parts))
    return categories

def asca_seaborn(ascao, F, legend_labels=None, plot_title=None, subplot_size=4,
                 cmap=None, show=True, save_filename=None):
    """
    Creates a publication-ready plot using Seaborn from the ASCA output.
    
    Behavior:
      - If scores have 1 component: produces a bar chart.
      - If scores have 2 components: produces a scatter plot.
      - If scores have >2 components: produces a full pairplot (all panels filled).
    
    The axis labels include the percent variance (e.g., "PC 1 (50.0%)").
    A figure-wide title is placed above the plot (via suptitle).
    
    Parameters:
      ascao (dict): Must contain keys 'scores', 'query', and 'percent_variation'.
      F (Tensor or array-like): Factor matrix (n_samples x n_factors).
      legend_labels (list, optional): (Not directly used here.)
      plot_title (str, optional): Overall title for the plot.
      subplot_size (float, optional): Size (in inches) for each subplot (default 4).
      cmap (str or list, optional): Colormap to use. If a string, passed to sns.color_palette; defaults to "viridis".
      show (bool, optional): Whether to display the plot.
      save_filename (str, optional): Filename to save the figure as PNG.
    
    Returns:
      - For 1 or 2 components: (fig, ax) for a single plot.
      - For >2 components: The Seaborn PairGrid object.
    """
    # Ensure F is a tensor.
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F)
    
    scores = ascao.get('scores')
    if scores is None:
        raise ValueError("ascao must contain the 'scores' key.")
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    n_samples, n_components = scores.shape
    
    percent_variation = ascao.get('percent_variation')
    if percent_variation is None:
        raise ValueError("ascao must contain the 'percent_variation' key.")
    if not isinstance(percent_variation, torch.Tensor):
        percent_variation = torch.tensor(percent_variation)
    
    query = ascao.get('query')
    if query is None:
        raise ValueError("ascao must contain the 'query' key.")
    categories = get_numeric_categories(F, query)
    
    # Extract the real part of scores in case data is complex.
    data = scores.cpu().numpy()
    if np.iscomplexobj(data):
        data = np.real(data)
    
    # Create a DataFrame from the real part of scores.
    pc_cols = [f"PC {i+1} ({percent_variation[i].item():.1f}%)" for i in range(n_components)]
    df = pd.DataFrame(data, columns=pc_cols)
    df["Category"] = categories
    
    # Determine the color palette.
    unique_categories = sorted(set(categories))
    n_categories = len(unique_categories)
    if cmap is None:
        palette = sns.color_palette("viridis", n_categories)
    else:
        palette = sns.color_palette(cmap, n_categories)
    palette_dict = dict(zip(unique_categories, palette))
    
    # Create the figure-wide title text.
    fig_title = plot_title if plot_title is not None else ""
    
    if n_components == 1:
        # Bar chart.
        df["Sample"] = df.index
        plt.figure(figsize=(subplot_size, subplot_size))
        ax = sns.barplot(data=df, x="Sample", y=pc_cols[0], hue="Category", palette=palette_dict)
        ax.set_xlabel("Sample")
        ax.set_ylabel("PC 1 Score")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.legend(title="", loc='upper right')
        fig = plt.gcf()
        fig.suptitle(fig_title, fontsize=16, y=0.98)
        fig.set_size_inches(subplot_size, subplot_size)  # Force square
        if save_filename is None and not show:
            save_filename = (plot_title.replace(" ", "_") + ".png") if plot_title else "plot.png"
            fig.savefig(save_filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return fig, ax
    elif n_components == 2:
        # Scatter plot.
        plt.figure(figsize=(subplot_size, subplot_size))
        ax = sns.scatterplot(data=df, x=pc_cols[0], y=pc_cols[1], hue="Category", palette=palette_dict)
        ax.set_xlabel(pc_cols[0])
        ax.set_ylabel(pc_cols[1])
        ax.legend(title="", loc='upper right')
        fig = plt.gcf()
        fig.suptitle(fig_title, fontsize=16, y=0.98)
        fig.set_size_inches(subplot_size, subplot_size)  # Force square
        if save_filename is None and not show:
            save_filename = (plot_title.replace(" ", "_") + ".png") if plot_title else "plot.png"
            fig.savefig(save_filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return fig, ax
    else:
        # For more than 2 components, use a full pairplot (i.e. fill the grid).
        pairgrid = sns.pairplot(df, vars=pc_cols, hue="Category",
                                palette=palette_dict, height=subplot_size)
        # Remove gridlines from each subplot for a cleaner look.
        for ax in pairgrid.axes.flatten():
            if ax is not None:
                ax.grid(False)
        # Adjust the subplots so that the title has room above.
        pairgrid.fig.subplots_adjust(top=0.92)
        if fig_title:
            pairgrid.fig.suptitle(fig_title, fontsize=16, y=0.98)
        if save_filename is None and not show:
            save_filename = (plot_title.replace(" ", "_") + ".png") if plot_title else "plot.png"
            pairgrid.fig.savefig(save_filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return pairgrid
