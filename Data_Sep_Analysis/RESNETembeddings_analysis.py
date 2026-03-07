import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50V2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigvalsh
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import UMAP gracefully.
try:
    from umap import UMAP
except ImportError:
    UMAP = None

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
CONFIG = {
    # Paths (Update these as needed)
    'BALANCED_DIR': r"E:\Study Materials\Projects\ML_PlantVillage\Curriculum Learning\TeacherModel\plantvillage_whole_balanced",
    'WEIGHTS_PATH': r"E:\Study Materials\Projects\ML_PlantVillage\Curriculum Learning\TeacherModel\plantvillage_whole_resnet_2stage_weights.h5",
    
    # Preprocessing
    'IMG_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    
    # Model Architecture Specifics (Matching training state)
    'UNFREEZE_LAST_N': 50,
    
    # Analysis Parameters
    'PCA_VARIANCE_RETAINED': 0.95,
    'REG_EPSILON': 1e-6, # Regularization for covariance stability
    
    # Visualization Settings
    'PLOT_FIGSIZE_HEATMAP': (12, 10),
    'PLOT_FIGSIZE_SCATTER': (24, 10),
    'TSNE_PERPLEXITY': 30.0,
    'UMAP_NEIGHBORS': 15,
    'UMAP_MIN_DIST': 0.1
}

# Add standard input shape to config
CONFIG['INPUT_SHAPE'] = CONFIG['IMG_SIZE'] + (3,)

# General printing separator
def print_sep(title=""):
    width = 60
    if not title:
        print("=" * width)
    else:
        print(f"\n{'='*5} {title.upper()} {'='* (width - 7 - len(title))}")


# ==============================================================================
# 2. MODEL & DATA LOADING FUNCTIONS
# ==============================================================================
def load_model_and_generator(cfg):
    """Rebuilds architecture, loads weights, and sets up data generator."""
    print_sep("Loading Resources")
    
    # --- Check Prerequisites ---
    if not os.path.exists(cfg['WEIGHTS_PATH']):
        raise FileNotFoundError(f"Cannot find weights at {cfg['WEIGHTS_PATH']}.")

    # --- Recreate Data Generator (Standard preprocessing) ---
    print("Initializing Validation Generator...")
    # NOTE: shuffle=False is strictly required to align features with true labels
    datagen = ImageDataGenerator(rescale=1./255)
    val_gen = datagen.flow_from_directory(
        cfg['BALANCED_DIR'], 
        target_size=cfg['IMG_SIZE'], 
        batch_size=cfg['BATCH_SIZE'],
        class_mode='categorical',
        subset=None, 
        shuffle=False 
    )

    num_classes = val_gen.num_classes
    print(f"Detected {num_classes} classes from generator.")

    # --- Rebuild Model Skeleton ---
    print("Rebuilding ResNet50V2 architecture...")
    inputs = layers.Input(shape=cfg['INPUT_SHAPE'])
    
    # Internal graph needs imagenet for consistency, weights loaded over top
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=cfg['INPUT_SHAPE'])

    # --- MATCH TRAINING FREEZE STATE EXACTLY ---
    # This aligns the weight lists for correct loading
    base_model.trainable = True
    freeze_until_layer = len(base_model.layers) - cfg['UNFREEZE_LAST_N']

    for layer in base_model.layers[:freeze_until_layer]:
        layer.trainable = False
        
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Build classification head
    x = base_model(inputs) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x) 
    model = Model(inputs, outputs)

    # --- Load Weights ---
    print("Loading trained weights (by_name=True)...")
    model.load_weights(cfg['WEIGHTS_PATH'], by_name=True)
    print("Weights loaded successfully!")

    return model, val_gen


# ==============================================================================
# 3. FEATURE EXTRACTION & PCA FUNCTIONS
# ==============================================================================
def extract_and_reduce_features(model, val_gen, cfg):
    """Extracts GAP embeddings, aligns labels, and applies PCA."""
    print_sep("Feature Extraction & PCA")

    # Target the GlobalAveragePooling2D layer
    # gap_layer_output = model.layers[-3].output # Dependent on exact stack
    # Safer approach: find layer by type or name
    gap_layer = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
    feature_extractor = Model(inputs=model.input, outputs=gap_layer.output)
    
    print(f"Feature extractor built targeting layer: {gap_layer.name}")

    # Extract embeddings
    print("Running inference on validation set (this may take a moment)...")
    raw_embeddings = feature_extractor.predict(val_gen)
    true_labels = val_gen.classes
    print(f"Raw features matrix: {raw_embeddings.shape}")

    # Reverse class indices mapping for friendly names
    class_indices = val_gen.class_indices
    class_names_map = {v: k for k, v in class_indices.items()}

    # Apply PCA for dimensional reduction and covariance stability
    print(f"Applying PCA (retaining {cfg['PCA_VARIANCE_RETAINED']*100}% variance)...")
    pca = PCA(n_components=cfg['PCA_VARIANCE_RETAINED'])
    reduced_embeddings = pca.fit_transform(raw_embeddings)
    
    print(f"Reduced features matrix: {reduced_embeddings.shape}")
    
    return reduced_embeddings, true_labels, class_names_map, pca.n_components_


# ===========================================
# 4. QUANTITATIVE ANALYSIS FUNCTIONS
# ===========================================
def compute_class_statistics(embeddings, labels, class_names_map, cfg):
    """
    Computes critical stats (mean, variance, covariance) for each class cluster.
    Regularizes covariance matrices for subsequent high-d determinant math.
    """
    print_sep("Class Statistics (Gaussian Modeling)")
    stats = {}
    num_classes = len(class_names_map)
    n_features = embeddings.shape[1]
    reg_eps = cfg['REG_EPSILON']

    for class_idx in range(num_classes):
        class_name = class_names_map[class_idx]
        # Isolate embeddings for this class
        mask = (labels == class_idx)
        class_feats = embeddings[mask]
        
        if len(class_feats) <= 1:
            print(f"Warning: Class '{class_name}' has insufficient samples for modeling.")
            continue

        # mu (centroid)
        mu = np.mean(class_feats, axis=0)
        
        # Sigma (Covariance), rowvar=False because samples are rows
        sigma = np.cov(class_feats, rowvar=False)
        
        # Intra-class variance (avg variance across all dimensions)
        variance = np.var(class_feats, axis=0).mean()

        # D. REGULARIZATION (CRITICAL): Ensure stability for determinants in 523 dimensions
        sigma += np.eye(n_features) * reg_eps
        
        stats[class_name] = {
            'mu': mu,
            'sigma': sigma,
            'variance': variance,
            'n_samples': len(class_feats)
        }

    print(f"Gaussian models established for {len(stats)} classes with {reg_eps:.1e} regularization.")
    return stats


def analyze_basic_clustering(class_stats):
    """Displays class consistency (Intra-Class Variance)."""
    print_sep("Intra-Class Variance Analysis")
    
    variances = []
    for name, data in class_stats.items():
        variances.append({'Class': name, 'Variance': data['variance']})
        
    variance_df = pd.DataFrame(variances).sort_values(by='Variance', ascending=True)

    print("Top 5 Classes with LOWEST Variance (Tighter clustering):")
    print(variance_df.head(5).to_string(index=False))
    print("\nTop 5 Classes with HIGHEST Variance (Looser clustering):")
    print(variance_df.tail(5).to_string(index=False))
    return variance_df


def compute_fdr(embeddings, labels, class_stats):
    """Calculates Global Fisher Discriminant Ratio (FDR) [S_B / S_W]."""
    print_sep("Fisher Discriminant Ratio (Global)")
    
    n_samples, n_features = embeddings.shape
    
    # Global mean of entire dataset
    global_mean = np.mean(embeddings, axis=0)

    # Initialize d x d scatter matrices
    S_W = np.zeros((n_features, n_features)) # Within-class
    S_B = np.zeros((n_features, n_features)) # Between-class

    print(f"Calculating FDR scatter matrices in {n_features} dimensions...")

    # Reuse pre-calculated class statistics
    for class_idx, class_name in enumerate(class_stats.keys()):
        stats = class_stats[class_name]
        
        mu_c = stats['mu']
        # Sigma_c is scaled sample covariance, S_W requires total scatter (sum of squared diffs)
        # S_W_c = (N_c - 1) * Sigma_c
        S_W_c = (stats['n_samples'] - 1) * stats['sigma']
        S_W += S_W_c
        
        # Between-class weighted spread of centroids around global mean
        mean_diff = (mu_c - global_mean).reshape(n_features, 1)
        S_B_c = stats['n_samples'] * (mean_diff.dot(mean_diff.T))
        S_B += S_B_c

    # FDR based on Trace Ratio (Simplified Multiclass extension)
    between_variance = np.trace(S_B)
    within_variance = np.trace(S_W)
    
    # 1e-9 safety factor against division by zero theory
    global_fdr = between_variance / (within_variance + 1e-9)

    print("\n--- FINAL FDR RESULTS ---")
    print(f"Total Between-Class Variance (Signal): {between_variance:.4f}")
    print(f"Total Within-Class Variance (Noise):  {within_variance:.4f}")
    print("-" * 35)
    print(f"GLOBAL FISHER DISCRIMINANT RATIO:     {global_fdr:.6f}")
    print("-" * 35)

    if global_fdr > 1.0:
        print("Interpretation: Signal > Noise. The dataset shows good global separability.")
    else:
        print("Interpretation: Noise > Signal. The dataset shows poor global separability (Raw Trace Bias).")
    
    return global_fdr


# ===========================================
# 5. PROBABILISTIC DISTANCES (BHAT/CHERNOFF)
# ===========================================
def _compute_chernoff_objective(s, mu1, sigma1, mu2, sigma2, diff_mu, n_features):
    """Internal function minimized to find optimal Chernoff parameter 's'."""
    
    # Combined weighted covariance matrix
    sigma_s = s * sigma2 + (1 - s) * sigma1
    
    # --- 1. The Matrix Quadratic Term (Mean Gaps) ---
    term1 = 0.5 * s * (1 - s) * (diff_mu.T.dot(np.linalg.inv(sigma_s)).dot(diff_mu))[0, 0]
    
    # --- 2. The Log Determinant Term (Weighted Shape Gaps) ---
    # Must use slogdet (sign and log of determinant) to prevent 
    # underflow/overflow in high dimensions.
    sign_s, logdet_s = np.linalg.slogdet(sigma_s)
    sign_1, logdet_1 = np.linalg.slogdet(sigma1)
    sign_2, logdet_2 = np.linalg.slogdet(sigma2)
    
    # Penalize negative determinants (not expected for valid covariance matrices)
    if sign_s <= 0 or sign_1 <= 0 or sign_2 <= 0:
        return 1e10 

    term2 = 0.5 * (logdet_s - ( (1-s)*logdet_1 + s*logdet_2 ))
    
    total_dist = term1 + term2
    
    # We want to MAXIMIZE distance, so return negative distance to optimizer.
    return -total_dist 


def analyze_probabilistic_distances(class_stats, class_names_list, cfg):
    """Calculates Centroid Gaps, Bhattacharyya (s=0.5), and Optimized Chernoff distances."""
    print_sep("Theoretical Distances (Chernoff/Bhattacharyya)")
    
    n_classes = len(class_names_list)
    n_features = next(iter(class_stats.values()))['mu'].shape[0]

    # Initialize d x d distance matrices
    centroid_distances = np.zeros((n_classes, n_classes))
    chernoff_matrix = np.zeros((n_classes, n_classes))
    bhattacharyya_matrix = np.zeros((n_classes, n_classes))
    optimal_s_matrix = np.zeros((n_classes, n_classes))

    print(f"Starting exhaustive pairwise calculations for {n_classes} classes...")
    total_pairs = (n_classes * (n_classes - 1)) // 2 
    pairs_processed = 0

    for i in range(n_classes):
        name_i = class_names_list[i]
        stats_i = class_stats[name_i]
        
        for j in range(i + 1, n_classes):
            name_j = class_names_list[j]
            stats_j = class_stats[name_j]
            
            diff_mu = (stats_i['mu'] - stats_j['mu']).reshape(-1, 1)

            # --- A. Simple Inter-Class Centroid Distance ---
            dist_euc = np.linalg.norm(stats_i['mu'] - stats_j['mu'])
            centroid_distances[i, j] = centroid_distances[j, i] = dist_euc

            # --- B. Optimized Chernoff Distance ---
            args = (stats_i['mu'], stats_i['sigma'], stats_j['mu'], stats_j['sigma'], diff_mu, n_features)
            # Find optimal s in [0, 1] to minimize theoretical error bound
            res = minimize_scalar(_compute_chernoff_objective, bounds=(0, 1), method='bounded', args=args)
            c_dist = -res.fun # Optimized Chernoff distance
            
            # --- C. Bhattacharyya Distance (Chernoff where s is fixed at 0.5) ---
            b_dist = -_compute_chernoff_objective(0.5, *args)
            
            # Fill symmetric matrices
            chernoff_matrix[i, j] = chernoff_matrix[j, i] = c_dist
            bhattacharyya_matrix[i, j] = bhattacharyya_matrix[j, i] = b_dist
            optimal_s_matrix[i, j] = optimal_s_matrix[j, i] = res.x
            
            pairs_processed += 1
            if pairs_processed % 20 == 0:
                print(f"Processed {pairs_processed}/{total_pairs} pairs...")

    print("Distance calculations completed successfully.")
    
    # Log optimization analysis
    avg_s = np.mean(optimal_s_matrix[optimal_s_matrix > 0])
    max_gain = np.max(chernoff_matrix - bhattacharyya_matrix)
    print(f"\nChernoff Optimization Analysis:\nAverage optimal 's': {avg_s:.3f} (Bhattacharyya assumes s=0.5)")
    print(f"Max distance gain from optimizing s: {max_gain:.4f}")

    # Build return dictionary
    results = {
        'centroid': centroid_distances,
        'chernoff': chernoff_matrix,
        'bhattacharyya': bhattacharyya_matrix
    }
    return results

def print_top_raw_distances(distance_matrix, class_names_list, title, top_k=5):
    """Prints the pairs with the HIGHEST raw distances (Greatest Separation)."""
    print(f"\n--- Top {top_k} Pairs: {title} (Raw Values) ---")
    print("Interpretation: HIGHER raw distance = BETTER fundamental separation.")
    
    # Create a copy and fill diagonal with NaNs so we don't pick class-to-self pairs
    plot_matrix = distance_matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    
    # Get indices of the highest distances (sorted ascending, then reversed)
    flat_indices = np.argsort(plot_matrix, axis=None)[::-1]
    
    count = 0
    seen_pairs = set()
    for idx in flat_indices:
        if count >= top_k:
            break
        i, j = np.unravel_index(idx, plot_matrix.shape)
        pair = tuple(sorted((i, j)))
        
        # Guard against NaNs (handled by argsort but safe practice)
        if np.isnan(plot_matrix[i, j]): continue

        if pair not in seen_pairs:
            seen_pairs.add(pair)
            print(f"Distance {plot_matrix[i, j]:.4f} : {class_names_list[i]} <--> {class_names_list[j]}")
            count += 1

# ==============================================================================
# 6. VISUALIZATION FUNCTIONS (HEATMAPS & SCATTERPLOTS)
# ==============================================================================
def plot_distance_heatmap(distance_matrix, class_names_list, cfg, title, cbar_label, mask_diagonal=True):
    """Plots clean heatmaps for various distance matrices."""
    # Create a copy so we don't modify the analytical matrix
    plot_matrix = distance_matrix.copy()

    # Mask the diagonal (distance to itself is usually non-informative 0 or inf)
    if mask_diagonal:
        plot_matrix[np.isinf(plot_matrix)] = np.nan
        np.fill_diagonal(plot_matrix, np.nan)

    plt.figure(figsize=cfg['PLOT_FIGSIZE_HEATMAP'])
    
    # Use standard reverse Viridis color map for distances (darker = closer)
    sns.heatmap(
        plot_matrix, 
        xticklabels=class_names_list, 
        yticklabels=class_names_list, 
        cmap='viridis_r',
        cbar_kws={'label': cbar_label},
        annot=(len(class_names_list) <= 15), # Add number annotations only for few classes
        fmt=".1f"   # Formatted to 1 decimal place
    )

    plt.title(f'{title}\n(Input: {distance_matrix.shape[0]} Classes)', fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_confusability_heatmap(chernoff_matrix, class_names_list, cfg):
    """
    Converts optimized Chernoff distances into theoretical upper bounds on 
    classification error (Confusability Matrix). Darker Reds indicate high theoretical ambiguity.
    """
    print_sep("Confusability Analysis (Error Bounds)")
    
    # Convert optimized distances into Bayes Error upper bounds (assuming balanced priors 0.5)
    # Probability Bound <= 0.5 * e^(-D_Chernoff)
    error_bound_matrix = 0.5 * np.exp(-chernoff_matrix)

    # Fill diag with 0 (not a potential confusion)
    np.fill_diagonal(error_bound_matrix, 0.0)

    # --- Print Top Confusable pairs ---
    flat_errors = error_bound_matrix.flatten()
    sorted_indices = np.argsort(flat_errors)[::-1] # descending

    print("Theoretical Bayes Error Bounds (Top 5 Potential Confusion Pairs):")
    print("Interpretation: Theoretical probability of confusion is LESS THAN these values.")

    seen_bounds = set()
    count = 0
    for idx in sorted_indices:
        if count >= 5: break
        i, j = np.unravel_index(idx, error_bound_matrix.shape)
        pair = tuple(sorted((i, j)))
        if pair not in seen_bounds and i != j:
            seen_bounds.add(pair)
            bound_percent = error_bound_matrix[i, j] * 100
            print(f"Bound {bound_percent:.4f}% : {class_names_list[i]} <--> {class_names_list[j]}")
            count += 1

    # --- Plot the Matrix ---
    plot_df = pd.DataFrame(error_bound_matrix, index=class_names_list, columns=class_names_list)
    # Mask diag for Seaborn
    mask = np.zeros_like(plot_df, dtype=bool)
    np.fill_diagonal(mask, True)

    plt.figure(figsize=cfg['PLOT_FIGSIZE_HEATMAP'])
    sns.heatmap(
        plot_df, mask=mask, cmap='Reds', annot=True, fmt=".3f",   
        cbar_kws={'label': 'Max Theoretical Confusion Probability (Bayes Bound)'}
    )
    plt.title('Theoretical Class Confusability Matrix\n(Optimized Chernoff Bound)', fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()


def run_qualitative_visualization(embeddings, labels, class_names_list, cfg):
    """Plots side-by-side t-SNE and UMAP visualizations."""
    print_sep("Qualitative Visualization (t-SNE & UMAP)")
    
    palette = sns.color_palette("husl", len(class_names_list))

    # Single plotting helper
    def _plot_embedding(embedding_2d, title, ax):
        df = pd.DataFrame(embedding_2d, columns=['C1', 'C2'])
        df['Class'] = [class_names_list[i] for i in labels]
        
        scatter = sns.scatterplot(
            data=df, x='C1', y='C2', hue='Class',
            palette=palette, s=15, alpha=0.6, ax=ax, edgecolor=None
        )
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
        if ax.get_legend(): ax.get_legend().remove()
        return scatter

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg['PLOT_FIGSIZE_SCATTER'])
    plt.suptitle(f'Visualizing Intrinsic Ambiguity: 15-Class PlantVillage Subset\n(Input: GAP-ResNet Features)', fontsize=20, y=0.98)

    # --- t-SNE (Local structure focus) ---
    print("\nRunning t-SNE optimization...")
    tsne_results = TSNE(
        n_components=2, perplexity=cfg['TSNE_PERPLEXITY'], 
        random_state=42, n_jobs=-1
    ).fit_transform(embeddings)
    _plot_embedding(tsne_results, 't-SNE (Focus: Local Neighbor Structure)', ax1)


    # --- UMAP (Balanced structure focus) ---
    if UMAP is not None:
        print("\nRunning UMAP optimization...")
        umap_results = UMAP(
            n_neighbors=cfg['UMAP_NEIGHBORS'], min_dist=cfg['UMAP_MIN_DIST'], 
            random_state=42, metric='euclidean'
        ).fit_transform(embeddings)
        _plot_embedding(umap_results, 'UMAP (Focus: Balanced Local/Global Structure)', ax2)
    else:
        ax2.text(0.5, 0.5, 'UMAP (Skipped)\npip install umap-learn', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('UMAP (Skipped)')

    # Single Shared Legend
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()


# ==============================================================================
# 7. MAIN EXECUTION PIPELINE (UPDATED FOR RAW DISTANCE DISPLAY)
# ==============================================================================
if __name__ == "__main__":
    print_sep("Separability Analysis Pipeline Initialized")
    
    # --- Step 1: Loading & Rebuilding ---
    resnet_model, val_generator = load_model_and_generator(CONFIG)
    
    # --- Step 2: Inference & Extraction ---
    reduced_embs, aligned_labels, class_map, pca_dims = extract_and_reduce_features(resnet_model, val_generator, CONFIG)
    
    # Build list of names in correct index order [0, 1, 2...]
    num_classes_found = len(class_map)
    ordered_names_list = [class_map[i] for i in range(num_classes_found)]

    # --- Step 3: Probability Statistics Modeling ---
    # This models each class as a 523-d Multivariate Gaussian
    stats_dictionary = compute_class_statistics(reduced_embs, aligned_labels, class_map, CONFIG)
    
    # --- Step 4: Consistency Analysis (Intra-Class Variance) ---
    analyze_basic_clustering(stats_dictionary)
    
    # --- Step 5: Global Separability (Fisher Ratio) ---
    compute_fdr(reduced_embs, aligned_labels, stats_dictionary)
    
    # --- Step 6: EXHAUSTIVE PAIRWISE DISTANCE ANALYSIS ---
    # This returns matrices for: Centroid (Euc), Bhattacharyya, and Chernoff
    distance_results = analyze_probabilistic_distances(stats_dictionary, ordered_names_list, CONFIG)
    
    # -------------------------------------------------------------------------
    # --- NEW STEP: DISPLAY RAW BHATTACHARYYA & CHERNOFF DISTANCES ---
    # This helps you get a feel for the magnitude of separation.
    # -------------------------------------------------------------------------
    
    # 1. Print Top 5 RAW Bhattacharyya Distances (s=0.5 Baseline)
    print_top_raw_distances(
        distance_results['bhattacharyya'], ordered_names_list, 
        "Bhattacharyya Distance (s=0.5 Baseline)"
    )
    
    # 2. Print Top 5 RAW Optimized Chernoff Distances (Definitive separation)
    # NOTE: You expect these numbers to be higher than Bhattacharyya.
    print_top_raw_distances(
        distance_results['chernoff'], ordered_names_list, 
        "Optimized Chernoff Distance"
    )

    # --- Step 7: VISUALIZATION PIPELINE (HEATMAPS) ---
    
    # Heatmap 1: Inter-Class Centroid gaps (Simplest Euclidean visual separation)
    plot_distance_heatmap(
        distance_results['centroid'], ordered_names_list, CONFIG,
        'Inter-Class Centroid Distance Heatmap', 'Euclidean Distance'
    )
    
    # NEW Heatmap 2: Raw Bhattacharyya distance heatmap
    plot_distance_heatmap(
        distance_results['bhattacharyya'], ordered_names_list, CONFIG,
        'Raw Bhattacharyya Distance Heatmap (s=0.5 Baseline)', 'Bhattacharyya Distance'
    )

    # NEW Heatmap 3: Raw Optimized Chernoff distance heatmap
    # NOTE: This matrix will look darker/better separated than Heatmap 2.
    plot_distance_heatmap(
        distance_results['chernoff'], ordered_names_list, CONFIG,
        'Raw Optimized Chernoff Distance Heatmap', 'Chernoff Distance'
    )
    
    # Heatmap 4: Theoretical Confusability matrix (The Probability bounds)
    plot_confusability_heatmap(distance_results['chernoff'], ordered_names_list, CONFIG)
    
    # --- Step 8: VISUALIZATION PIPELINE (QUALITATIVE SCATTERPLOTS) ---
    # Side-by-side t-SNE and UMAP
    run_qualitative_visualization(reduced_embs, aligned_labels, ordered_names_list, CONFIG)
    
    print_sep("Pipeline Complete")

# ==============================================================================
# 8. NEXT LOGICAL STEP: EXPORTING FINALIZED ANALYZED FEATURES (FIXED NAMES)
# ==============================================================================
print("\n" + "="*60)
print("OPTIONAL NEXT STEP: EXPORTING ANALYZED FEATURES FOR OPTIMIZATION")
print("="*60)

# 1. Define export path based on teacher weight directory
# Assumes the user wants features saved near the teacher model weights.
# We access os from the imports already established in main.
WEIGHTS_DIR = os.path.dirname(CONFIG['WEIGHTS_PATH'])
EXPORT_FILENAME = "plantvillage_whole_features_analyzed_523d.npz"
EXPORT_PATH = os.path.join(WEIGHTS_DIR, EXPORT_FILENAME)

# 2. Confirmation prompt (to avoid accidental overwrites)
# sys is imported from main for flush/read.choice.
sys.stdout.write(f"\nDo you wish to export the analyzed, PCA-reduced embeddings,\ntrue labels, and ordered class names to the following compressed NumPy file?\n({EXPORT_PATH})\n[y/N]: ")
sys.stdout.flush()
user_choice = sys.stdin.readline().strip().lower()

if user_choice in ['y', 'yes']:
    print(f"Compressing and exporting features...")
    
    # Use numpy's compressed save function for efficiency.
    # We corrected the variable names here to match the scope of main.
    np.savez_compressed(
        EXPORT_PATH,
        # DATA VARIABLES FROM THE REFACTORED MAIN SCOPE:
        reduced_embeddings=reduced_embs,  # matrix (N x 523)
        true_labels=aligned_labels,       # vector (N)
        # Convert list to numpy array for saving
        ordered_class_names=np.array(ordered_names_list) # list of 15 strings
    )
    
    print("\n[SUCCESS] Analyzed features exported successfully.")
    print("For future analysis, you can bypass this entire script and load")
    print("the features instantly using this command:")
    print(f"data = np.load(r'{EXPORT_PATH}')")
else:
    print("\nFeature export skipped.")

print("\n" + "="*60)
print("PIPELINE COMPLETE & OPTIMIZATION SAVED")
print("="*60)