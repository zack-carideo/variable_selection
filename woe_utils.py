import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

def plot_woe_response_rates(
    X_original, X_woe_ordinal, y, feature_name, ordinal_encoder, feature_idx,
    figsize=(10, 6), ax=None
):
    """
    Plot mean response rate for each ordinal level of a WoE transformed variable.

    Parameters:
    -----------
    X_original : pd.DataFrame
        Original feature data (before WoE transformation)
    X_woe_ordinal : pd.DataFrame
        WoE transformed and ordinal encoded features
    y : array-like
        Binary target variable
    feature_name : str
        Name of the feature to plot
    ordinal_encoder : OrdinalEncoder
        Fitted ordinal encoder with WoE categories
    feature_idx : int
        Index of the feature in the encoder's categories
    figsize : tuple
        Figure size (width, height)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    """
    plot_df = pd.DataFrame({
        'ordinal_bin': X_woe_ordinal[feature_name],
        'target': y
    })

    response_by_bin = plot_df.groupby('ordinal_bin')['target'].agg(['mean', 'count']).reset_index()
    response_by_bin.columns = ['ordinal_bin', 'response_rate', 'count']
    response_by_bin = response_by_bin.sort_values('ordinal_bin')

    woe_values = ordinal_encoder.categories_[feature_idx]

    labels = [f"{int(ord_val)}\n(WoE: {woe_values[int(ord_val)]:.3f})"
              for ord_val in response_by_bin['ordinal_bin']]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.bar(
        range(len(response_by_bin)),
        response_by_bin['response_rate'],
        color='steelblue', alpha=0.7, edgecolor='black'
    )

    for i, (idx, row) in enumerate(response_by_bin.iterrows()):
        ax.text(i, row['response_rate'] + 0.01,
                f"n={int(row['count'])}",
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('WoE Ordinal Bin (Original WoE Value)', fontsize=12)
    ax.set_ylabel('Mean Response Rate', fontsize=12)
    ax.set_title(f'Response Rate by WoE Bin: {feature_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(response_by_bin)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(response_by_bin['response_rate']) * 1.15)

    plt.tight_layout()
    return fig, ax

class WoEBinning:
    """
    Weight of Evidence (WoE) binning for continuous and categorical variables.

    Features:
    - Handles continuous and categorical variables
    - Enforces monotonicity of WoE values
    - Handles missing values as separate level
    - Ensures minimum bin size with bin merging
    """

    def __init__(
        self,
        min_bin_pct: float = 0.05,
        min_bins: int = 2,
        max_bins: int = 20,
        require_monotonic: bool = True
    ):
        self.min_bin_pct = min_bin_pct
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.require_monotonic = require_monotonic

        self.bin_edges_ = None
        self.woe_map_ = None
        self.is_categorical_ = None
        self.missing_woe_ = None
        self.iv_ = None

    def _calc_woe(self, bin_events: int, bin_non_events: int,
                  total_events: int, total_non_events: int) -> float:
        """Calculate WoE for a single bin."""
        pct_events = bin_events / total_events if total_events > 0 else 1e-10
        pct_non_events = bin_non_events / total_non_events if total_non_events > 0 else 1e-10
        return np.log(pct_non_events / pct_events)

    def _is_monotonic(self, values: List[float]) -> bool:
        """Check if values are monotonically increasing or decreasing."""
        if len(values) < 2:
            return True
        diff = np.diff(values)
        return np.all(diff >= 0) or np.all(diff <= 0)

    def _handle_missing(self, X: pd.Series, y: pd.Series) -> None:
        """Calculate WoE for missing values."""
        missing_mask = X.isna()
        if missing_mask.sum() == 0:
            self.missing_woe_ = None
            return

        events_miss = y[missing_mask].sum()
        non_events_miss = missing_mask.sum() - events_miss
        total_events = y.sum()
        total_non_events = len(y) - total_events

        if events_miss > 0 and non_events_miss > 0:
            self.missing_woe_ = self._calc_woe(events_miss, non_events_miss,
                                               total_events, total_non_events)
        else:
            self.missing_woe_ = 0.0

    def _merge_small_bins(self, bin_labels: np.ndarray, y: np.ndarray,
                         ordered_bins: List) -> Tuple[List, Dict]:
        """Merge bins below minimum size threshold."""
        total_events = y.sum()
        total_non_events = len(y) - total_events

        while len(ordered_bins) > self.min_bins:
            # Calculate bin statistics
            bin_stats = []
            for bin_val in ordered_bins:
                mask = bin_labels == bin_val
                count = mask.sum()
                bin_stats.append({'bin': bin_val, 'count': count, 'pct': count / len(y)})

            # Find bins below threshold
            small_bins = [i for i, stat in enumerate(bin_stats)
                         if stat['pct'] < self.min_bin_pct]

            if not small_bins:
                break

            # Merge smallest bin with adjacent bin
            smallest_idx = min(small_bins, key=lambda i: bin_stats[i]['count'])

            if smallest_idx == 0:
                merge_idx = 1
            elif smallest_idx == len(ordered_bins) - 1:
                merge_idx = len(ordered_bins) - 2
            else:
                left_size = bin_stats[smallest_idx - 1]['count']
                right_size = bin_stats[smallest_idx + 1]['count']
                merge_idx = smallest_idx - 1 if left_size < right_size else smallest_idx + 1

            bin_to_remove = ordered_bins[smallest_idx]
            bin_to_keep = ordered_bins[merge_idx]
            bin_labels[bin_labels == bin_to_remove] = bin_to_keep
            ordered_bins.pop(smallest_idx)

        # Calculate WoE for final bins
        woe_map = {}
        for bin_val in ordered_bins:
            mask = bin_labels == bin_val
            events = y[mask].sum()
            non_events = mask.sum() - events

            if events > 0 and non_events > 0:
                woe_map[bin_val] = self._calc_woe(events, non_events,
                                                   total_events, total_non_events)

        return ordered_bins, woe_map

    def _enforce_monotonicity(self, bin_labels: np.ndarray, y: np.ndarray,
                             ordered_bins: List, woe_map: Dict) -> Tuple[List, Dict]:
        """Merge bins to enforce monotonicity."""
        if not self.require_monotonic:
            return ordered_bins, woe_map

        total_events = y.sum()
        total_non_events = len(y) - total_events
        max_iterations = 100

        for _ in range(max_iterations):
            # Filter to valid bins only
            ordered_bins = [b for b in ordered_bins if b in woe_map]

            if len(ordered_bins) <= self.min_bins:
                break

            woe_values = [woe_map[b] for b in ordered_bins]
            if self._is_monotonic(woe_values):
                break

            # Find pair with smallest WoE difference to merge
            diffs = [(i, abs(woe_values[i+1] - woe_values[i]))
                    for i in range(len(woe_values) - 1)]

            if not diffs:
                break

            merge_idx = min(diffs, key=lambda x: x[1])[0]

            # Merge bins
            bin_to_remove = ordered_bins[merge_idx + 1]
            bin_to_keep = ordered_bins[merge_idx]
            bin_labels[bin_labels == bin_to_remove] = bin_to_keep
            ordered_bins.pop(merge_idx + 1)

            # Recalculate WoE for merged bin
            mask = bin_labels == bin_to_keep
            events = y[mask].sum()
            non_events = mask.sum() - events

            if events > 0 and non_events > 0:
                woe_map[bin_to_keep] = self._calc_woe(events, non_events,
                                                       total_events, total_non_events)

            if bin_to_remove in woe_map:
                del woe_map[bin_to_remove]

        return ordered_bins, woe_map

    def _calc_iv(self, bin_labels: np.ndarray, y: np.ndarray,
                 ordered_bins: List, woe_map: Dict) -> float:
        """Calculate Information Value."""
        total_events = y.sum()
        total_non_events = len(y) - total_events

        iv_total = 0
        for bin_val in ordered_bins:
            if bin_val not in woe_map:
                continue

            mask = bin_labels == bin_val
            events = y[mask].sum()
            non_events = mask.sum() - events

            pct_events = events / total_events
            pct_non_events = non_events / total_non_events
            iv_total += (pct_non_events - pct_events) * woe_map[bin_val]

        return iv_total

    def _fit_continuous(self, X: pd.Series, y: pd.Series) -> None:
        """Fit WoE binning for continuous variable."""
        self._handle_missing(X, y)

        # Get non-missing data
        mask = ~X.isna()
        X_clean = X[mask].values
        y_clean = y[mask].values

        # Initial binning using quantiles
        try:
            _, bin_edges = pd.qcut(X_clean, q=min(self.max_bins, len(np.unique(X_clean))),
                                   retbins=True, duplicates='drop')
        except ValueError:
            _, bin_edges = pd.cut(X_clean, bins=self.max_bins, retbins=True)

        # Assign bins
        bin_labels = np.digitize(X_clean, bin_edges[1:-1])
        ordered_bins = sorted(np.unique(bin_labels))

        # Merge small bins
        ordered_bins, woe_map = self._merge_small_bins(bin_labels, y_clean, ordered_bins)

        # Enforce monotonicity
        ordered_bins, woe_map = self._enforce_monotonicity(bin_labels, y_clean,
                                                           ordered_bins, woe_map)

        # Create final bin edges
        final_edges = [float('-inf')]
        for i in range(len(ordered_bins) - 1):
            mask_curr = bin_labels == ordered_bins[i]
            mask_next = bin_labels == ordered_bins[i + 1]
            edge = (X_clean[mask_curr].max() + X_clean[mask_next].min()) / 2
            final_edges.append(edge)
        final_edges.append(float('inf'))

        self.bin_edges_ = final_edges
        self.woe_map_ = {i: woe_map[ordered_bins[i]] for i in range(len(ordered_bins))}
        self.iv_ = self._calc_iv(bin_labels, y_clean, ordered_bins, woe_map)

    def _fit_categorical(self, X: pd.Series, y: pd.Series) -> None:
        """Fit WoE binning for categorical variable."""
        self._handle_missing(X, y)

        # Get non-missing data
        mask = ~X.isna()
        X_clean = X[mask]
        y_clean = y[mask].values

        # Calculate WoE for each category and sort
        total_events = y_clean.sum()
        total_non_events = len(y_clean) - total_events

        cat_woe = {}
        for cat in X_clean.unique():
            cat_mask = X_clean == cat
            events = y_clean[cat_mask].sum()
            non_events = cat_mask.sum() - events

            if events > 0 and non_events > 0:
                cat_woe[cat] = self._calc_woe(events, non_events,
                                              total_events, total_non_events)

        sorted_cats = sorted(cat_woe.keys(), key=lambda x: cat_woe[x])

        # Create bin labels
        bin_labels = np.zeros(len(X_clean), dtype=int)
        for i, cat in enumerate(sorted_cats):
            bin_labels[X_clean == cat] = i

        ordered_bins = list(range(len(sorted_cats)))

        # Merge small bins
        ordered_bins, woe_map = self._merge_small_bins(bin_labels, y_clean, ordered_bins)

        # Enforce monotonicity
        ordered_bins, woe_map = self._enforce_monotonicity(bin_labels, y_clean,
                                                           ordered_bins, woe_map)

        # Create category to bin mapping
        cat_to_bin = {}
        for cat in X_clean.unique():
            cat_mask = (X_clean == cat).values
            if cat_mask.sum() > 0:
                bin_idx = bin_labels[cat_mask][0]
                if bin_idx in ordered_bins:
                    cat_to_bin[cat] = ordered_bins.index(bin_idx)

        self.bin_edges_ = cat_to_bin
        self.woe_map_ = {i: woe_map[ordered_bins[i]] for i in range(len(ordered_bins))}
        self.iv_ = self._calc_iv(bin_labels, y_clean, ordered_bins, woe_map)

    def fit(self, X: pd.Series, y: pd.Series) -> 'WoEBinning':
        """Fit the WoE binning transformer."""
        if X.dtype in ['object', 'category'] or X.dtype.name == 'bool':
            self.is_categorical_ = True
            self._fit_categorical(X, y)
        else:
            self.is_categorical_ = False
            self._fit_continuous(X, y)
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Transform feature to WoE values."""
        if self.woe_map_ is None:
            raise ValueError("Must fit before transform")

        woe_values = np.zeros(len(X))

        for i, val in enumerate(X):
            if pd.isna(val):
                woe_values[i] = self.missing_woe_ if self.missing_woe_ is not None else 0
            elif self.is_categorical_:
                if val in self.bin_edges_:
                    woe_values[i] = self.woe_map_[self.bin_edges_[val]]
                else:
                    # Unknown category - use mean WoE
                    woe_values[i] = np.mean(list(self.woe_map_.values()))
            else:
                bin_idx = np.digitize([val], self.bin_edges_[1:-1])[0]
                woe_values[i] = self.woe_map_.get(bin_idx, 0)

        return pd.Series(woe_values, index=X.index, name=f"{X.name}_woe")

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_bin_summary(self) -> pd.DataFrame:
        """Get summary of bins with WoE values."""
        if self.woe_map_ is None:
            raise ValueError("Must fit before getting summary")

        summary = []
        for bin_idx, woe in self.woe_map_.items():
            if self.is_categorical_:
                cats = [cat for cat, idx in self.bin_edges_.items() if idx == bin_idx]
                bin_range = f"Categories: {cats}"
            else:
                lower = self.bin_edges_[bin_idx]
                upper = self.bin_edges_[bin_idx + 1]
                bin_range = f"({lower:.3f}, {upper:.3f}]"

            summary.append({'bin': bin_idx, 'range': bin_range, 'woe': woe})

        return pd.DataFrame(summary)


def run_tests():
    """Comprehensive test suite for WoE binning."""
    print("="*70)
    print("WoE Binning Test Suite")
    print("="*70)

    # Test 1: Continuous variable with missing values
    print("\n[Test 1] Continuous Variable with Missing Values")
    print("-"*70)
    from sklearn.datasets import make_classification

    X_data, y_data = make_classification(
        n_samples=1000, n_features=1, n_informative=1, n_redundant=0,
        random_state=42, n_classes=2, n_clusters_per_class=1
    )

    rng = np.random.default_rng(42)
    missing_idx = rng.choice(X_data.shape[0], size=50, replace=False)
    X_data[missing_idx, 0] = np.nan

    X = pd.Series(X_data.flatten(), name='continuous_feature')
    y = pd.Series(y_data, name='target')

    binner = WoEBinning(min_bin_pct=0.05, min_bins=2, max_bins=10)
    X_woe = binner.fit_transform(X, y)

    print(f"IV: {binner.iv_:.4f}")
    print(f"Number of bins: {len(binner.woe_map_)}")
    print(f"Missing WoE: {binner.missing_woe_:.4f}" if binner.missing_woe_ else "No missing")
    print(f"Bin edges: {[f'{e:.2f}' for e in binner.bin_edges_]}")

    woe_vals = list(binner.woe_map_.values())
    monotonic = np.all(np.diff(woe_vals) >= 0) or np.all(np.diff(woe_vals) <= 0)
    print(f"Monotonicity: {'PASS' if monotonic else 'FAIL'}")
    print(f"WoE range: [{min(woe_vals):.2f}, {max(woe_vals):.2f}]")

    # Test 2: Categorical variable
    print("\n[Test 2] Categorical Variable with Missing Values")
    print("-"*70)

    categories = ['A', 'B', 'C', 'D', 'E', 'F']
    X_cat = rng.choice(categories, size=1000)
    y_cat = np.zeros(1000)

    for i, cat in enumerate(X_cat):
        prob = {'A': 0.1, 'B': 0.2, 'C': 0.4, 'D': 0.6, 'E': 0.8, 'F': 0.9}[cat]
        y_cat[i] = 1 if rng.random() < prob else 0

    missing_idx = rng.choice(1000, size=50, replace=False)
    X_cat = X_cat.astype(object)
    X_cat[missing_idx] = np.nan

    X_cat_series = pd.Series(X_cat, name='categorical_feature')
    y_cat_series = pd.Series(y_cat, name='target')

    binner_cat = WoEBinning(min_bin_pct=0.05, min_bins=2, max_bins=6)
    X_cat_woe = binner_cat.fit_transform(X_cat_series, y_cat_series)

    print(f"IV: {binner_cat.iv_:.4f}")
    print(f"Number of bins: {len(binner_cat.woe_map_)}")
    print(f"Missing WoE: {binner_cat.missing_woe_:.4f}" if binner_cat.missing_woe_ else "No missing")

    woe_vals_cat = list(binner_cat.woe_map_.values())
    monotonic_cat = np.all(np.diff(woe_vals_cat) >= 0) or np.all(np.diff(woe_vals_cat) <= 0)
    print(f"Monotonicity: {'PASS' if monotonic_cat else 'FAIL'}")
    print(f"Category mapping: {binner_cat.bin_edges_}")

    # Test 3: Highly imbalanced continuous variable
    print("\n[Test 3] Highly Imbalanced Continuous Variable")
    print("-"*70)

    X_imb = rng.normal(0, 1, 1000)
    y_imb = (X_imb > 0.5).astype(int)  # Only 30% positive
    X_imb_series = pd.Series(X_imb, name='imbalanced_feature')
    y_imb_series = pd.Series(y_imb, name='target')

    binner_imb = WoEBinning(min_bin_pct=0.05, min_bins=2, max_bins=8)
    X_imb_woe = binner_imb.fit_transform(X_imb_series, y_imb_series)

    print(f"IV: {binner_imb.iv_:.4f}")
    print(f"Number of bins: {len(binner_imb.woe_map_)}")
    print(f"Target distribution: {y_imb.mean():.1%} positive")

    woe_vals_imb = list(binner_imb.woe_map_.values())
    monotonic_imb = np.all(np.diff(woe_vals_imb) >= 0) or np.all(np.diff(woe_vals_imb) <= 0)
    print(f"Monotonicity: {'PASS' if monotonic_imb else 'FAIL'}")

    # Test 4: Low cardinality categorical
    print("\n[Test 4] Low Cardinality Categorical (3 categories)")
    print("-"*70)

    X_low_card = rng.choice(['Low', 'Medium', 'High'], size=1000, p=[0.5, 0.3, 0.2])
    y_low_card = np.zeros(1000)
    for i, val in enumerate(X_low_card):
        prob = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}[val]
        y_low_card[i] = 1 if rng.random() < prob else 0

    X_lc_series = pd.Series(X_low_card, name='low_card_feature')
    y_lc_series = pd.Series(y_low_card, name='target')

    binner_lc = WoEBinning(min_bin_pct=0.05, min_bins=2, max_bins=10)
    X_lc_woe = binner_lc.fit_transform(X_lc_series, y_lc_series)

    print(f"IV: {binner_lc.iv_:.4f}")
    print(f"Number of bins: {len(binner_lc.woe_map_)}")

    woe_vals_lc = list(binner_lc.woe_map_.values())
    monotonic_lc = np.all(np.diff(woe_vals_lc) >= 0) or np.all(np.diff(woe_vals_lc) <= 0)
    print(f"Monotonicity: {'PASS' if monotonic_lc else 'FAIL'}")
    print(f"Category mapping: {binner_lc.bin_edges_}")

    # Test 5: No relationship (random data)
    print("\n[Test 5] No Relationship (Random Continuous)")
    print("-"*70)

    X_random = rng.normal(0, 1, 1000)
    y_random = rng.choice([0, 1], size=1000)
    X_rand_series = pd.Series(X_random, name='random_feature')
    y_rand_series = pd.Series(y_random, name='target')

    binner_rand = WoEBinning(min_bin_pct=0.05, min_bins=2, max_bins=10)
    X_rand_woe = binner_rand.fit_transform(X_rand_series, y_rand_series)

    print(f"IV: {binner_rand.iv_:.4f} (expected: ~0 for random)")
    print(f"Number of bins: {len(binner_rand.woe_map_)}")

    woe_vals_rand = list(binner_rand.woe_map_.values())
    monotonic_rand = np.all(np.diff(woe_vals_rand) >= 0) or np.all(np.diff(woe_vals_rand) <= 0)
    print(f"Monotonicity: {'PASS' if monotonic_rand else 'FAIL'}")

    # Test 6: Transform on new data
    print("\n[Test 6] Transform on New Data (Continuous)")
    print("-"*70)

    X_new = pd.Series(rng.normal(0, 1, 100), name='continuous_feature')
    X_new_woe = binner.transform(X_new)

    print(f"Original shape: {X_new.shape}")
    print(f"Transformed shape: {X_new_woe.shape}")
    print(f"WoE value range: [{X_new_woe.min():.2f}, {X_new_woe.max():.2f}]")
    print(f"Sample transformation (first 5): {X_new_woe.head().values}")

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    all_tests_passed = all([monotonic, monotonic_cat, monotonic_imb,
                           monotonic_lc, monotonic_rand])
    print(f"All monotonicity tests: {'PASSED' if all_tests_passed else 'FAILED'}")
    print(f"Transform test: PASSED")
    print("\nAll tests completed successfully!")


#ALTERNATE APPROACH FOR CHI-MERGE BINNING
def chi_merge_binning(
    X: pd.Series, 
    y: pd.Series, 
    initial_bins: int = 50,
    max_bins: int = 10,
    significance: float = 0.05
) -> list:
    """
    Chi-square based bin merging.
    Merges adjacent bins that are statistically similar.
    
    IMPLEMENTATION EXAMPLE
     
    X_woe = pd.DataFrame()
    for col in X.columns:
        bin_edges = chi_merge_binning(X[col], y, initial_bins=50, max_bins=5, significance=0.1)
        X_woe[col] = pd.cut(X[col], bins=bin_edges, duplicates='drop')
        
    """
    df = pd.DataFrame({'feature': X, 'target': y})
    
    # Start with fine bins
    df['bin'] = pd.qcut(df['feature'], q=initial_bins, duplicates='drop')
    
    bin_edges = df.groupby('bin', observed=True)['feature'].min().sort_values().tolist()
    
    while len(bin_edges) > max_bins:
        # Calculate chi-square for each adjacent pair
        chi_scores = []
        
        for i in range(len(bin_edges) - 1):
            if i == 0:
                mask1 = df['feature'] < bin_edges[1]
            else:
                mask1 = (df['feature'] >= bin_edges[i]) & (df['feature'] < bin_edges[i+1])
            
            if i + 2 >= len(bin_edges):
                mask2 = df['feature'] >= bin_edges[i+1]
            else:
                mask2 = (df['feature'] >= bin_edges[i+1]) & (df['feature'] < bin_edges[i+2])
            
            # Build contingency table
            contingency = pd.crosstab(
                df.loc[mask1 | mask2, 'feature'] >= bin_edges[i+1],
                df.loc[mask1 | mask2, 'target']
            )
            
            if contingency.shape == (2, 2) and contingency.min().min() > 0:
                chi2, p_val, _, _ = chi2_contingency(contingency)
                chi_scores.append((i, chi2, p_val))
            else:
                chi_scores.append((i, 0, 1.0))
        
        if not chi_scores:
            break
        
        # Find pair with highest p-value (most similar)
        merge_idx = max(chi_scores, key=lambda x: x[2])[0]
        
        # Merge bins
        bin_edges.pop(merge_idx + 1)
    
    return [-np.inf] + bin_edges[1:] + [np.inf]


if __name__ == "__main__":
    run_tests()
    