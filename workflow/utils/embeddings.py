"""
Utilities for computing, saving, and visualizing dimensionality reduction embeddings.

This module provides:
- ResourceGuard: Monitor and limit computational resources
- EmbeddingPipeline: Compute/cache UMAP, t-SNE, PCA embeddings
- Visualization utilities: scatter plots, layered plots, multi-panel figures
- Data loading and alignment helpers

Supports PCA preprocessing: UMAP/t-SNE can be computed on PCA-reduced features
for denoising and improved performance.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    import resource

    import psutil

    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    warnings.warn("Resource monitoring not available. Install with: pip install psutil")


# =============================================================================
# Resource Management
# =============================================================================


class ResourceGuard:
    """
    Monitor and limit resource usage to prevent runaway processes.

    Features:
    - Memory usage monitoring and warnings
    - CPU count limiting for parallel operations
    - Data size validation before expensive operations
    - Automatic memory cleanup suggestions

    Examples
    --------
    >>> guard = ResourceGuard(max_memory_gb=32, max_cpus=10)
    >>> guard.check_memory("UMAP embedding", required_gb=5.0)
    >>> safe_jobs = guard.get_safe_n_jobs(requested_jobs=-1)
    """

    def __init__(self, max_memory_gb=32, max_cpus=10):
        """
        Initialize resource guard.

        Parameters
        ----------
        max_memory_gb : float
            Maximum memory in GB to allow (warning threshold)
        max_cpus : int
            Maximum number of CPUs for parallel operations
        """
        if not RESOURCE_MONITORING_AVAILABLE:
            raise ImportError(
                "Resource monitoring requires psutil. Install with: pip install psutil"
            )

        import os

        self.max_memory_bytes = max_memory_gb * 1024**3
        self.max_cpus = max_cpus
        self.process = psutil.Process(os.getpid())

        # Set memory limit on Unix systems
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Set soft limit to max_memory_gb (allow some overhead)
            resource.setrlimit(
                resource.RLIMIT_AS, (int(self.max_memory_bytes * 1.2), hard)
            )
            print(f"✓ Memory limit set to {max_memory_gb:.1f} GB (soft limit)")
        except (ValueError, OSError) as e:
            print(f"⚠ Could not set memory limit: {e}")
            print(f"  Memory monitoring will still work, but no hard limit enforced")

    def get_memory_usage(self):
        """Get current memory usage in GB."""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024**3

    def get_available_memory(self):
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / 1024**3

    def check_memory(self, operation_name="operation", required_gb=None):
        """
        Check current memory usage and warn if approaching limit.

        Parameters
        ----------
        operation_name : str
            Name of the operation for warning messages
        required_gb : float, optional
            Estimated memory required for operation

        Returns
        -------
        safe : bool
            True if safe to proceed, False if memory is too high
        """
        current_gb = self.get_memory_usage()
        available_gb = self.get_available_memory()
        max_gb = self.max_memory_bytes / 1024**3

        # Check current usage
        if current_gb > max_gb * 0.9:
            print(f"⚠ WARNING: High memory usage!")
            print(
                f"  Current: {current_gb:.2f} GB / {max_gb:.1f} GB limit ({current_gb/max_gb*100:.1f}%)"
            )
            print(f"  Available system memory: {available_gb:.2f} GB")
            print(f"  Consider: reducing N_DEBUG_SAMPLES or restarting kernel")
            return False

        # Check if operation will fit
        if required_gb is not None:
            if current_gb + required_gb > max_gb:
                print(f"⚠ WARNING: {operation_name} may exceed memory limit!")
                print(f"  Current: {current_gb:.2f} GB")
                print(f"  Required: ~{required_gb:.2f} GB")
                print(
                    f"  Total: ~{current_gb + required_gb:.2f} GB > {max_gb:.1f} GB limit"
                )
                return False

        return True

    def get_safe_n_jobs(self, requested_jobs=-1):
        """
        Get safe number of parallel jobs based on CPU limits.

        Parameters
        ----------
        requested_jobs : int
            Requested number of jobs (-1 for max_cpus)

        Returns
        -------
        n_jobs : int
            Safe number of jobs to use
        """
        total_cpus = psutil.cpu_count(logical=True)

        # Use configured max_cpus limit
        if requested_jobs == -1:
            n_jobs = self.max_cpus
        else:
            n_jobs = min(requested_jobs, self.max_cpus)

        if n_jobs < total_cpus:
            print(
                f"ℹ Limiting parallel jobs to {n_jobs}/{total_cpus} CPUs (max_cpus={self.max_cpus})"
            )

        return n_jobs

    def estimate_embedding_memory(self, n_samples, n_features, method="umap"):
        """
        Estimate memory required for embedding computation.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        method : str
            'umap', 'tsne', or 'pca'

        Returns
        -------
        estimated_gb : float
            Estimated memory in GB
        """
        # Rough estimates based on typical usage patterns
        if method == "umap":
            # UMAP scales roughly O(n_samples * n_neighbors)
            estimated_bytes = n_samples * n_features * 8 * 10  # Conservative estimate
        elif method == "tsne":
            # Peak Memory Formula: M = 208 × N × P + 1,285 × N + 126,000,000
            estimated_bytes = (
                208 * n_samples * n_features + 1285 * n_samples + 126_000_000
            )
        elif method == "pca":
            # PCA is relatively memory-efficient
            estimated_bytes = n_samples * n_features * 8 * 3
        else:
            estimated_bytes = n_samples * n_features * 8 * 5  # Generic estimate

        return estimated_bytes / 1024**3

    def print_status(self):
        """Print current resource usage status."""
        mem_gb = self.get_memory_usage()
        mem_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.1)

        print(f"{'='*60}")
        print(f"Resource Usage Status:")
        print(f"  Memory: {mem_gb:.2f} GB (System: {mem_percent:.1f}% used)")
        print(f"  CPU: {cpu_percent:.1f}% (Current)")
        print(f"  CPUs available: {psutil.cpu_count(logical=True)}")
        print(
            f"  Max allowed: {self.max_memory_bytes/1024**3:.1f} GB memory, {self.max_cpus} CPUs"
        )
        print(f"{'='*60}")


# =============================================================================
# Embedding Pipeline
# =============================================================================


class EmbeddingPipeline:
    """
    Pipeline for computing, saving, and loading dimensionality reduction embeddings.
    Ensures alignment between embeddings, features, and labels.

    Supports PCA preprocessing: UMAP/t-SNE can be computed on PCA-reduced features
    for denoising and improved performance.

    Examples
    --------
    >>> pipeline = EmbeddingPipeline("embeddings", subset_id="all")
    >>> umap_emb, meta = pipeline.compute_or_load_embedding(
    ...     X, index, method="umap", n_neighbors=15, min_dist=0.1
    ... )
    >>> pipeline.save_features(features_df, name="scaled_features")
    >>> pipeline.save_labels(labels_df, name="ground_truth")
    """

    def __init__(
        self, embedding_dir="embeddings", subset_id="all", resource_guard=None
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        embedding_dir : str or Path
            Base directory for embeddings
        subset_id : str
            Identifier for the data subset (used in filenames)
        resource_guard : ResourceGuard, optional
            Resource monitoring and limiting instance
        """
        self.embedding_dir = Path(embedding_dir)
        self.subset_id = subset_id
        self.cache_dir = self.embedding_dir / "cache"
        self.features_dir = self.embedding_dir / "features"
        self.labels_dir = self.embedding_dir / "labels"
        self.resource_guard = resource_guard

        # Ensure directories exist
        for d in [self.cache_dir, self.features_dir, self.labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_embedding_path(self, method, preprocess=None, **kwargs):
        """Generate standardized filename for embeddings."""
        # Add preprocessing suffix to filename
        preprocess_str = f"_{preprocess}" if preprocess and preprocess != "raw" else ""

        if method == "umap":
            n_neighbors = kwargs.get("n_neighbors", 15)
            min_dist = kwargs.get("min_dist", 0.1)
            filename = (
                f"umap_n{n_neighbors}_d{min_dist}{preprocess_str}_{self.subset_id}.npz"
            )
            return self.cache_dir / filename
        elif method == "tsne":
            perplexity = kwargs.get("perplexity", 30)
            max_iter = kwargs.get("max_iter", kwargs.get("n_iter", 1000))
            filename = (
                f"tsne_p{perplexity}_i{max_iter}{preprocess_str}_{self.subset_id}.npz"
            )
            return self.cache_dir / filename
        elif method == "pca":
            variance_pct = kwargs.get("variance_pct", None)
            n_components = kwargs.get("n_components", 2)
            whiten = kwargs.get("whiten", False)
            whiten_str = "_whiten" if whiten else ""
            # Use variance percentage in filename if specified
            if variance_pct is not None:
                filename = (
                    f"pca_var{int(variance_pct*100)}{whiten_str}_{self.subset_id}.npz"
                )
            else:
                filename = f"pca_c{n_components}{whiten_str}_{self.subset_id}.npz"
            return self.cache_dir / filename
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_or_load_embedding(
        self, X, index, method="umap", preprocess=None, force_recompute=False, **kwargs
    ):
        """
        Compute embedding or load from cache if exists.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        index : pd.Index or list
            Sample identifiers (must match X rows)
        method : str
            'umap', 'tsne', or 'pca'
        preprocess : str, optional
            Preprocessing to apply before embedding:
            - None or "raw": Use raw features
            - "pca_N": First reduce to N PCA components (e.g., "pca_50")
        force_recompute : bool
            If True, recompute even if cache exists
        **kwargs : dict
            Hyperparameters for the embedding method

        Returns
        -------
        embedding : np.ndarray
            2D embedding coordinates
        metadata : dict
            Hyperparameters and provenance info

        Examples
        --------
        # Standard UMAP on raw features
        >>> umap_emb, _ = pipeline.compute_or_load_embedding(X, idx, method="umap")

        # UMAP on 50 PCA components (for denoising/speed)
        >>> umap_emb, _ = pipeline.compute_or_load_embedding(
        ...     X, idx, method="umap", preprocess="pca_50", n_neighbors=15
        ... )

        # t-SNE on 100 PCA components
        >>> tsne_emb, _ = pipeline.compute_or_load_embedding(
        ...     X, idx, method="tsne", preprocess="pca_100", perplexity=30
        ... )
        """
        cache_path = self._get_embedding_path(method, preprocess=preprocess, **kwargs)

        # Try to load from cache
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached embedding from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            embedding = data["embedding"]
            metadata = json.loads(data["metadata"].item())

            # Verify index alignment
            if list(data["index"]) != list(index):
                print("Warning: Cached index doesn't match. Recomputing...")
            else:
                return embedding, metadata

        # Handle preprocessing
        X_processed = X
        preprocess_metadata = {}

        if preprocess and preprocess.startswith("pca_"):
            # Extract number of components
            n_pca_components = int(preprocess.split("_")[1])
            print(f"Preprocessing with PCA ({n_pca_components} components)...")

            # Compute or load PCA preprocessing
            pca_emb, pca_meta = self.compute_or_load_embedding(
                X,
                index,
                method="pca",
                n_components=n_pca_components,
                random_state=kwargs.get("random_state", 42),
                force_recompute=False,
            )
            X_processed = pca_emb
            preprocess_metadata = {
                "preprocess": preprocess,
                "pca_explained_variance": pca_meta.get(
                    "explained_variance_total", None
                ),
            }
            print(
                f"  PCA explained variance: {preprocess_metadata['pca_explained_variance']:.2%}"
            )
        elif preprocess and preprocess != "raw":
            raise ValueError(f"Unknown preprocessing method: {preprocess}")

        # Resource check before expensive computation
        if self.resource_guard:
            estimated_mem = self.resource_guard.estimate_embedding_memory(
                X_processed.shape[0], X_processed.shape[1], method
            )
            if not self.resource_guard.check_memory(
                f"{method.upper()} embedding", estimated_mem
            ):
                raise MemoryError(
                    f"Insufficient memory for {method.upper()} embedding. "
                    f"Reduce N_DEBUG_SAMPLES or restart kernel."
                )

        # Compute embedding
        print(f"Computing {method.upper()} embedding with params: {kwargs}")
        if method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "UMAP not available. Install with: pip install umap-learn"
                )

            n_neighbors = kwargs.get("n_neighbors", 15)
            min_dist = kwargs.get("min_dist", 0.1)
            random_state = kwargs.get("random_state", 42)
            n_jobs = kwargs.get("n_jobs", -1)

            # Limit parallel jobs if resource guard is active
            if self.resource_guard:
                n_jobs = self.resource_guard.get_safe_n_jobs(n_jobs)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            embedding = reducer.fit_transform(X_processed)

        elif method == "tsne":
            perplexity = kwargs.get("perplexity", 30)
            max_iter = kwargs.get("max_iter", kwargs.get("n_iter", 1000))
            random_state = kwargs.get("random_state", 42)

            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=20,
            )
            embedding = reducer.fit_transform(X_processed)

        elif method == "pca":
            variance_pct = kwargs.get("variance_pct", None)
            n_components = kwargs.get("n_components", 2)
            whiten = kwargs.get("whiten", False)
            random_state = kwargs.get("random_state", 42)

            # Use variance percentage if specified, otherwise use n_components
            if variance_pct is not None:
                pca_param = variance_pct  # Float between 0 and 1
                print(f"  Computing PCA to retain {variance_pct*100:.1f}% of variance")
            else:
                pca_param = n_components

            reducer = PCA(
                n_components=pca_param, whiten=whiten, random_state=random_state
            )
            embedding = reducer.fit_transform(X_processed)

            # Add explained variance to metadata
            kwargs["explained_variance_ratio"] = (
                reducer.explained_variance_ratio_.tolist()
            )
            kwargs["explained_variance_total"] = float(
                reducer.explained_variance_ratio_.sum()
            )
            kwargs["n_components_actual"] = embedding.shape[
                1
            ]  # Actual number of components selected

        # Save to cache
        metadata = {
            "method": method,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "subset_id": self.subset_id,
            **preprocess_metadata,
            **kwargs,
        }

        np.savez_compressed(
            cache_path,
            embedding=embedding,
            index=np.array(index),
            metadata=json.dumps(metadata),
        )
        print(f"Saved embedding to {cache_path}")

        return embedding, metadata

    def save_features(self, features_df, name="scaled_features"):
        """
        Save feature matrix with index.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to save
        name : str
            Name identifier for the features
        """
        features_path = self.features_dir / f"{name}_{self.subset_id}.npz"
        np.savez_compressed(
            features_path,
            features=features_df.values,
            index=features_df.index.values,
            columns=features_df.columns.values,
        )
        print(f"Saved features to {features_path}")

    def load_features(self, name="scaled_features"):
        """
        Load feature matrix.

        Parameters
        ----------
        name : str
            Name identifier for the features

        Returns
        -------
        df : pd.DataFrame
            Loaded feature matrix
        """
        features_path = self.features_dir / f"{name}_{self.subset_id}.npz"
        data = np.load(features_path, allow_pickle=True)
        df = pd.DataFrame(
            data["features"], index=data["index"], columns=data["columns"]
        )
        return df

    def save_labels(self, labels_df, name):
        """
        Save label/metadata dataframe.

        Parameters
        ----------
        labels_df : pd.DataFrame
            Labels or metadata to save
        name : str
            Name identifier for the labels
        """
        labels_path = self.labels_dir / f"{name}_{self.subset_id}.csv"
        labels_df.to_csv(labels_path)
        print(f"Saved labels to {labels_path}")

    def load_labels(self, name):
        """
        Load label/metadata dataframe.

        Parameters
        ----------
        name : str
            Name identifier for the labels

        Returns
        -------
        df : pd.DataFrame
            Loaded labels or metadata
        """
        labels_path = self.labels_dir / f"{name}_{self.subset_id}.csv"
        return pd.read_csv(labels_path, index_col=0)

    def list_cached_embeddings(self):
        """
        List all cached embeddings.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with cached embedding information
        """
        cached = []
        for f in self.cache_dir.glob("*.npz"):
            data = np.load(f, allow_pickle=True)
            metadata = json.loads(data["metadata"].item())
            cached.append(
                {
                    "file": f.name,
                    "method": metadata["method"],
                    "n_samples": metadata["n_samples"],
                    "subset_id": metadata.get("subset_id", "unknown"),
                    "preprocess": metadata.get("preprocess", "raw"),
                    "params": {
                        k: v
                        for k, v in metadata.items()
                        if k
                        not in [
                            "method",
                            "n_samples",
                            "n_features",
                            "subset_id",
                            "preprocess",
                            "pca_explained_variance",
                        ]
                    },
                }
            )
        return pd.DataFrame(cached) if cached else pd.DataFrame()


# =============================================================================
# Visualization Utilities
# =============================================================================


def embed_scatter_plot(
    embedding,
    ax,
    colors,
    title="",
    mask=None,
    alpha=0.6,
    s=1,
    cmap="viridis",
    annotate=True,
    add_colorbar=True,
):
    """
    Scatter plot of 2D embeddings.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates (n_samples, 2)
    ax : matplotlib.axes.Axes
        Axes to plot on
    colors : array-like
        Color values (numeric array or RGB tuples)
    title : str
        Title for the plot
    mask : array-like, optional
        Boolean mask to filter points
    alpha : float
        Transparency level
    s : float
        Point size
    cmap : str
        Colormap name
    annotate : bool
        Whether to add labels and colorbar
    add_colorbar : bool
        Whether to add a colorbar (default True)

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        Scatter plot object
    """
    if mask is not None:
        scatter = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=(
                colors[mask]
                if hasattr(colors, "__len__") and len(colors) > 1
                else colors
            ),
            cmap=cmap,
            alpha=alpha,
            s=s,
        )
    else:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=alpha, s=s
        )

    if annotate:
        method = title.split()[0] if title else "Embedding"
        ax.set_xlabel(f"{method} 1", fontsize=12)
        ax.set_ylabel(f"{method} 2", fontsize=12)
        if title:
            ax.set_title(title, fontsize=13)

        # Add colorbar for numeric data
        if add_colorbar:
            if isinstance(colors, (np.ndarray, pd.Series, list)) and len(colors) > 0:
                # Check if it's numeric data (not RGB tuples or categorical strings)
                if isinstance(colors, pd.Series):
                    is_numeric = pd.api.types.is_numeric_dtype(colors)
                elif isinstance(colors, np.ndarray):
                    is_numeric = np.issubdtype(colors.dtype, np.number)
                elif isinstance(colors, list):
                    is_numeric = not isinstance(colors[0], (tuple, str))
                else:
                    is_numeric = False

                if is_numeric:
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Value", fontsize=11)

    return scatter


def layered_plot(
    embedding,
    ax,
    colors,
    title="",
    mask=None,
    bg_alpha=0.2,
    fg_alpha=0.8,
    bg_s=0.1,
    fg_s=0.5,
    cmap="viridis",
    annotate=True,
    add_colorbar=True,
):
    """
    Create a two-layer plot: gray background + colored foreground.
    Common pattern for highlighting a subset (e.g., test set) over all data.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates
    ax : matplotlib.axes.Axes
        Axes to plot on
    colors : array-like
        Color values for the foreground layer
    title : str
        Plot title
    mask : array-like
        Boolean mask for foreground points (if None, plots all)
    bg_alpha, fg_alpha : float
        Transparency for background and foreground
    bg_s, fg_s : float
        Point sizes for background and foreground
    cmap : str
        Colormap name
    annotate : bool
        Whether to add labels and colorbar
    add_colorbar : bool
        Whether to add a colorbar (default True)

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        Foreground scatter plot object
    """
    # Background layer (all points in gray)
    embed_scatter_plot(
        embedding,
        ax,
        colors="lightgray",
        title="",
        alpha=bg_alpha,
        s=bg_s,
        annotate=False,
        add_colorbar=False,
    )

    # Foreground layer (colored points, possibly masked)
    scatter = embed_scatter_plot(
        embedding,
        ax,
        colors=colors,
        mask=mask,
        title=title,
        alpha=fg_alpha,
        s=fg_s,
        cmap=cmap,
        annotate=annotate,
        add_colorbar=add_colorbar,
    )
    return scatter


def prepare_color_data(series, transformation=None, fillna_value=0):
    """
    Prepare a pandas Series for use as color data in plots.

    Parameters
    ----------
    series : pd.Series
        Data to transform
    transformation : str or callable
        'log10', 'sqrt', 'rank', or a custom function
    fillna_value : float
        Value to fill NaNs with

    Returns
    -------
    transformed : np.ndarray or pd.Series
        Transformed data ready for coloring
    """
    data = series.fillna(fillna_value)

    if transformation == "log10":
        return np.log10(data + 1)
    elif transformation == "sqrt":
        return np.sqrt(data)
    elif transformation == "rank":
        return data.rank(pct=True)
    elif callable(transformation):
        return transformation(data)
    else:
        return data


def create_multipanel_plot(
    embedding,
    features_dict,
    titles_dict,
    nrows=2,
    ncols=2,
    figsize=(16, 14),
    mask=None,
    cmap_dict=None,
    transformation_dict=None,
    **plot_kwargs,
):
    """
    Create a multi-panel figure with different features.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates (shared across all panels)
    features_dict : dict
        {key: feature_array} - features to plot
    titles_dict : dict
        {key: title_string} - titles for each subplot
    nrows, ncols : int
        Grid dimensions
    figsize : tuple
        Figure size
    mask : array-like, optional
        Boolean mask to apply to all panels
    cmap_dict : dict, optional
        {key: colormap_name} - custom colormaps per feature
    transformation_dict : dict, optional
        {key: transformation} - transformations per feature
    **plot_kwargs : dict
        Additional arguments passed to layered_plot

    Returns
    -------
    fig, axes : tuple
        Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    cmap_dict = cmap_dict or {}
    transformation_dict = transformation_dict or {}

    for idx, (key, colors) in enumerate(features_dict.items()):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        title = titles_dict.get(key, key)
        cmap = cmap_dict.get(key, "viridis")
        transform = transformation_dict.get(key, None)

        # Prepare color data
        colors_prepared = prepare_color_data(colors, transformation=transform)

        # Plot
        layered_plot(
            embedding,
            ax,
            colors_prepared,
            title=title,
            mask=mask,
            cmap=cmap,
            **plot_kwargs,
        )

    # Hide unused axes
    for idx in range(len(features_dict), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    return fig, axes


def create_biplot(
    scores,
    loadings,
    feature_names,
    pc1=0,
    pc2=1,
    colors=None,
    ax=None,
    n_top_features=10,
    alpha_samples=0.4,
):
    """
    Create a biplot showing both PC scores (samples) and loadings (features).

    Parameters
    ----------
    scores : np.ndarray
        PCA scores (samples x components)
    loadings : np.ndarray
        PCA loadings (features x components)
    feature_names : list
        Names of features
    pc1, pc2 : int
        Which PCs to plot (0-indexed)
    colors : array-like
        Colors for samples
    ax : matplotlib axes
        Axes to plot on
    n_top_features : int
        Number of top contributing features to label
    alpha_samples : float
        Transparency for sample points

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        Scatter plot object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot samples (scores)
    scatter = ax.scatter(
        scores[:, pc1],
        scores[:, pc2],
        c=colors,
        cmap="viridis",
        alpha=alpha_samples,
        s=1,
    )

    # Plot feature loadings (arrows)
    # Scale loadings to fit on the same plot as scores
    score_range = np.max(np.abs(scores[:, [pc1, pc2]]))
    loading_range = np.max(np.abs(loadings[:, [pc1, pc2]]))
    scale_factor = score_range / loading_range * 0.8

    loadings_scaled = loadings * scale_factor

    # Find top contributing features for each PC
    pc1_contrib = np.abs(loadings[:, pc1])
    pc2_contrib = np.abs(loadings[:, pc2])
    total_contrib = pc1_contrib + pc2_contrib
    top_indices = np.argsort(total_contrib)[-n_top_features:]

    # Plot arrows for top features
    for idx in top_indices:
        ax.arrow(
            0,
            0,
            loadings_scaled[idx, pc1],
            loadings_scaled[idx, pc2],
            color="red",
            alpha=0.7,
            head_width=0.1,
            head_length=0.1,
            linewidth=1.5,
        )

        # Add feature label
        ax.text(
            loadings_scaled[idx, pc1] * 1.1,
            loadings_scaled[idx, pc2] * 1.1,
            feature_names[idx],
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="red"
            ),
        )

    # Add origin lines
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    return scatter


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_and_align_data(file_path, index, index_col=0, **read_kwargs):
    """
    Load a dataframe and reindex to align with a reference index.

    Parameters
    ----------
    file_path : str or Path
        Path to data file
    index : pd.Index
        Reference index to align to
    index_col : int or str
        Column to use as index
    **read_kwargs : dict
        Additional arguments for pd.read_csv

    Returns
    -------
    df : pd.DataFrame
        Loaded and aligned dataframe
    """
    df = pd.read_csv(file_path, index_col=index_col, **read_kwargs)

    # Clean index if needed (e.g., remove metadata after pipe)
    if df.index.dtype == "object" and "|" in str(df.index[0]):
        df.index = df.index.str.split("|").str[0]

    return df.reindex(index)
