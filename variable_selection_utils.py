"""
Variable Selection Utilities for Logistic Regression
=====================================================

This module provides comprehensive utilities for ensemble-based variable selection
with nested cross-validation, specifically designed for fair lending models.

Modules:
- Data generation
- Assumption validation (VIF, linearity, EPV)
- Filter methods (IV, MI, correlation)
- Selection methods (LASSO, RFE, forward selection, tree importance)
- Nested CV framework
- Aggregation and stability metrics
- Visualization utilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Callable
import warnings

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import (
    mutual_info_classif, 
    SelectKBest, 
    RFECV,
    SelectPercentile
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

# Statsmodels imports
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_fair_lending_data(
    n_samples: int = 10000,
    random_state: int = 42,
    default_rate: float = 0.15,
    n_noise_features: int = 10
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic fair lending dataset with realistic feature distributions.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    default_rate : float
        Target default rate (approximate)
    n_noise_features : int
        Number of noise features to include
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target (1 = default, 0 = no default)
    """
    np.random.seed(random_state)
    
    # ===================
    # Core Credit Features
    # ===================
    
    # Credit Score (300-850, normally distributed around 680)
    credit_score = np.clip(np.random.normal(680, 80, n_samples), 300, 850)
    
    # Debt-to-Income Ratio (0-60%, right-skewed)
    dti_ratio = np.clip(np.random.exponential(15, n_samples) + 10, 0, 65)
    
    # Loan-to-Value Ratio (50-100%, normally distributed)
    ltv_ratio = np.clip(np.random.normal(78, 12, n_samples), 50, 100)
    
    # Annual Income (log-normal, $20k-$500k)
    annual_income = np.clip(np.random.lognormal(11, 0.5, n_samples), 20000, 500000)
    
    # Loan Amount (correlated with income)
    loan_amount = annual_income * np.random.uniform(2, 5, n_samples)
    loan_amount = np.clip(loan_amount, 50000, 1000000)
    
    # ===================
    # Employment Features
    # ===================
    
    # Employment Length (years, 0-40)
    employment_length = np.clip(np.random.exponential(5, n_samples), 0, 40)
    
    # Employment Type (categorical: 0=unemployed, 1=part-time, 2=full-time, 3=self-employed)
    employment_type = np.random.choice([0, 1, 2, 3], n_samples, p=[0.03, 0.07, 0.80, 0.10])
    
    # ===================
    # Credit History Features
    # ===================
    
    # Number of delinquencies (0-10, heavily right-skewed)
    num_delinquencies = np.clip(np.random.poisson(0.5, n_samples), 0, 10)
    
    # Months since last delinquency (0-120, 0 means never)
    months_since_delinq = np.where(
        num_delinquencies > 0,
        np.random.exponential(24, n_samples),
        0
    )
    months_since_delinq = np.clip(months_since_delinq, 0, 120)
    
    # Number of credit inquiries (last 6 months)
    num_inquiries = np.clip(np.random.poisson(1.5, n_samples), 0, 15)
    
    # Number of open accounts
    num_open_accounts = np.clip(np.random.poisson(8, n_samples) + 2, 1, 30)
    
    # Credit utilization ratio (0-100%)
    credit_utilization = np.clip(np.random.beta(2, 5, n_samples) * 100, 0, 100)
    
    # Total credit limit
    total_credit_limit = annual_income * np.random.uniform(0.3, 1.5, n_samples)
    
    # Revolving balance
    revolving_balance = total_credit_limit * (credit_utilization / 100)
    
    # ===================
    # Loan Characteristics
    # ===================
    
    # Interest Rate (correlated with credit score)
    base_rate = 4.0
    credit_adjustment = (750 - credit_score) / 100  # Higher score = lower rate
    interest_rate = np.clip(base_rate + credit_adjustment + np.random.normal(0, 0.5, n_samples), 2.5, 12)
    
    # Loan Term (months: 180, 240, 360)
    loan_term = np.random.choice([180, 240, 360], n_samples, p=[0.15, 0.25, 0.60])
    
    # Loan Purpose (categorical: 0=purchase, 1=refinance, 2=cash-out)
    loan_purpose = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.40, 0.15])
    
    # ===================
    # Property Features
    # ===================
    
    # Property Value
    property_value = loan_amount / (ltv_ratio / 100)
    
    # Property Type (0=single family, 1=condo, 2=multi-family)
    property_type = np.random.choice([0, 1, 2], n_samples, p=[0.70, 0.20, 0.10])
    
    # ===================
    # Derived Features
    # ===================
    
    # Payment to Income ratio
    monthly_payment = (loan_amount * (interest_rate/100/12) * 
                      (1 + interest_rate/100/12)**loan_term) / \
                      ((1 + interest_rate/100/12)**loan_term - 1)
    pti_ratio = (monthly_payment * 12) / annual_income * 100
    
    # ===================
    # Noise Features (no predictive power)
    # ===================
    noise_features = {}
    for i in range(n_noise_features):
        noise_features[f'noise_{i+1}'] = np.random.normal(0, 1, n_samples)
    
    # ===================
    # Correlated Features (to test multicollinearity handling)
    # ===================
    
    # Income variant (highly correlated with annual_income)
    monthly_income = annual_income / 12
    
    # Credit score variant
    fico_score = credit_score + np.random.normal(0, 5, n_samples)
    
    # ===================
    # Generate Target Variable
    # ===================
    
    # Create log-odds based on key risk factors
    log_odds = (
        -4.5  # Intercept (controls overall default rate)
        - 0.015 * (credit_score - 680)  # Higher score = lower default
        + 0.08 * (dti_ratio - 35)  # Higher DTI = higher default
        + 0.05 * (ltv_ratio - 80)  # Higher LTV = higher default
        - 0.00001 * (annual_income - 75000)  # Higher income = lower default
        + 0.3 * num_delinquencies  # More delinquencies = higher default
        + 0.1 * num_inquiries  # More inquiries = higher default
        + 0.02 * credit_utilization  # Higher utilization = higher default
        - 0.03 * employment_length  # Longer employment = lower default
        + 0.15 * (employment_type == 0)  # Unemployed = higher default
        + np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Convert to probability and generate binary outcome
    prob_default = expit(log_odds)
    y = (np.random.random(n_samples) < prob_default).astype(int)
    
    # ===================
    # Date Variable
    # ===================
    # 20% in 2021, 60% in 2022-2024, 20% in 2025
    n_2021 = int(n_samples * 0.2)
    n_2025 = int(n_samples * 0.2)
    n_2022_2024 = n_samples - n_2021 - n_2025  # 60%
    # Split 2022-2024 evenly
    n_2022 = n_2022_2024 // 3
    n_2023 = n_2022_2024 // 3
    n_2024 = n_2022_2024 - n_2022 - n_2023  # remainder to 2024

    date_list = (
        [2021] * n_2021 +
        [2022] * n_2022 +
        [2023] * n_2023 +
        [2024] * n_2024 +
        [2025] * n_2025
    )
    # Shuffle to randomize assignment
    np.random.shuffle(date_list)
    date_series = pd.to_datetime([f"{year}-01-01" for year in date_list])

    # ===================
    # Create DataFrame
    # ===================
    
    data = {
        # Core credit features
        'credit_score': credit_score,
        'dti_ratio': dti_ratio,
        'ltv_ratio': ltv_ratio,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        
        # Employment features
        'employment_length': employment_length,
        'employment_type': employment_type,
        
        # Credit history
        'num_delinquencies': num_delinquencies,
        'months_since_delinq': months_since_delinq,
        'num_inquiries': num_inquiries,
        'num_open_accounts': num_open_accounts,
        'credit_utilization': credit_utilization,
        'total_credit_limit': total_credit_limit,
        'revolving_balance': revolving_balance,
        
        # Loan characteristics
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'loan_purpose': loan_purpose,
        
        # Property features
        'property_value': property_value,
        'property_type': property_type,
        
        # Derived features
        'pti_ratio': pti_ratio,
        'monthly_payment': monthly_payment,
        
        # Correlated features (for multicollinearity testing)
        'monthly_income': monthly_income,
        'fico_score': fico_score,

        # Date variable
        'application_date': date_series,
    }
    
    # Add noise features
    data.update(noise_features)
    
    X = pd.DataFrame(data)
    
    print(f"Generated dataset: {n_samples} samples, {X.shape[1]} features")
    print(f"Default rate: {y.mean():.2%}")
    print("Date distribution:")
    print(X['application_date'].dt.year.value_counts().sort_index())
    
    return X, y


# =============================================================================
# ASSUMPTION VALIDATION UTILITIES
# =============================================================================

def calculate_epv(y: np.ndarray, n_vars: int) -> Dict[str, float]:
    """
    Calculate Events Per Variable (EPV) ratio.
    
    Parameters
    ----------
    y : np.ndarray
        Binary target
    n_vars : int
        Number of candidate variables
        
    Returns
    -------
    dict : EPV metrics and recommendations
    """
    n_events = y.sum()
    n_non_events = len(y) - n_events
    min_class = min(n_events, n_non_events)
    
    epv = min_class / n_vars if n_vars > 0 else float('inf')
    
    return {
        'n_events': int(n_events),
        'n_non_events': int(n_non_events),
        'n_variables': n_vars,
        'epv_ratio': epv,
        'meets_minimum_10': epv >= 10,
        'meets_robust_20': epv >= 20,
        'max_recommended_vars': int(min_class / 10)
    }


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for all features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
        
    Returns
    -------
    pd.DataFrame : VIF values for each feature
    """
    # Handle any infinite or missing values
    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_clean.values, i)
            vif_data.append({'feature': col, 'vif': vif})
        except Exception as e:
            vif_data.append({'feature': col, 'vif': np.nan})
    
    return pd.DataFrame(vif_data).sort_values('vif', ascending=False)


def iterative_vif_elimination(
    X: pd.DataFrame, 
    threshold: float = 5.0,
    verbose: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Iteratively remove features with VIF above threshold.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float
        VIF threshold (default 5.0)
    verbose : bool
        Print removal progress
        
    Returns
    -------
    tuple : (list of retained features, removal history DataFrame)
    """
    X_working = X.copy()
    removal_history = []
    
    while True:
        vif_df = calculate_vif(X_working)
        max_vif = vif_df['vif'].max()
        
        if max_vif < threshold or np.isnan(max_vif):
            break
            
        # Remove feature with highest VIF
        drop_feature = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
        removal_history.append({
            'feature': drop_feature,
            'vif': max_vif,
            'remaining_features': len(X_working.columns) - 1
        })
        
        if verbose:
            print(f"  Removing {drop_feature} (VIF={max_vif:.2f})")
            
        X_working = X_working.drop(columns=[drop_feature])
    
    return X_working.columns.tolist(), pd.DataFrame(removal_history)


def check_separation(X: pd.DataFrame, y: np.ndarray) -> Dict[str, List[str]]:
    """
    Check for perfect or quasi-perfect separation in the data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
        
    Returns
    -------
    dict : Features with separation issues
    """
    issues = {
        'perfect_separation': [],
        'quasi_separation': []
    }
    
    for col in X.columns:
        # Check for perfect separation
        by_target = X.groupby(y)[col].agg(['min', 'max'])
        # Skip date/time columns
        if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_timedelta64_dtype(X[col]):
            print(f'Skipping date/time column: {col}')
            continue
        if pd.api.types.is_period_dtype(X[col]) or pd.api.types.is_interval_dtype(X[col]):
            print(f'Skipping date/time column: {col}')
            continue
        if pd.api.types.is_datetime64tz_dtype(X[col]):
            print(f'Skipping date/time column: {col}')
            continue
        
        if len(by_target) == 2:
            # If max of class 0 < min of class 1 (or vice versa)
            if by_target.loc[0, 'max'] < by_target.loc[1, 'min']:
                issues['perfect_separation'].append(col)
            elif by_target.loc[1, 'max'] < by_target.loc[0, 'min']:
                issues['perfect_separation'].append(col)
            # Check for quasi-separation (very limited overlap)
            elif by_target.loc[0, 'max'] <= by_target.loc[1, 'min'] + 0.01:
                issues['quasi_separation'].append(col)
    
    return issues


def box_tidwell_test(
    X: pd.DataFrame, 
    y: np.ndarray, 
    continuous_vars: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform Box-Tidwell test for linearity in the logit.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
    continuous_vars : list
        List of continuous variable names to test
    alpha : float
        Significance level
        
    Returns
    -------
    pd.DataFrame : Test results for each variable
    """
    results = []
    
    for var in continuous_vars:
        try:
            X_test = X[[var]].copy()
            
            # Add small constant to avoid log(0)
            min_val = X_test[var].min()
            if min_val <= 0:
                X_test[var] = X_test[var] - min_val + 1
            
            # Create interaction term: X * log(X)
            X_test[f'{var}_log_int'] = X_test[var] * np.log(X_test[var])
            X_test = sm.add_constant(X_test)
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y, X_test).fit(disp=0, maxiter=100)
            
            interaction_pvalue = model.pvalues.get(f'{var}_log_int', np.nan)
            
            results.append({
                'variable': var,
                'interaction_coef': model.params.get(f'{var}_log_int', np.nan),
                'interaction_pvalue': interaction_pvalue,
                'linearity_holds': interaction_pvalue > alpha if not np.isnan(interaction_pvalue) else None
            })
        except Exception as e:
            results.append({
                'variable': var,
                'interaction_coef': np.nan,
                'interaction_pvalue': np.nan,
                'linearity_holds': None,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# FILTER METHODS
# =============================================================================

def calculate_iv(
    X: pd.DataFrame, 
    y: np.ndarray, 
    feature: str, 
    bins: int = 10
) -> float:
    """
    Calculate Information Value (IV) for a single feature.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
    feature : str
        Feature name
    bins : int
        Number of bins for continuous variables
        
    Returns
    -------
    float : Information Value
    """
    df = pd.DataFrame({'feature': X[feature], 'target': y})
    
    # Bin continuous variables
    n_unique = df['feature'].nunique()
    if n_unique > bins:
        try:
            df['bin'] = pd.qcut(df['feature'], q=bins, duplicates='drop')
        except ValueError:
            df['bin'] = pd.cut(df['feature'], bins=bins)
    else:
        df['bin'] = df['feature']
    
    # Calculate WoE and IV
    grouped = df.groupby('bin', observed=True)['target'].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']
    
    # Add small constant to avoid division by zero
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    
    if total_events == 0 or total_non_events == 0:
        return 0.0
    
    grouped['pct_events'] = (grouped['events'] + 0.5) / (total_events + 0.5 * len(grouped))
    grouped['pct_non_events'] = (grouped['non_events'] + 0.5) / (total_non_events + 0.5 * len(grouped))
    
    grouped['woe'] = np.log(grouped['pct_non_events'] / grouped['pct_events'])
    grouped['iv'] = (grouped['pct_non_events'] - grouped['pct_events']) * grouped['woe']
    
    return grouped['iv'].sum()


def calculate_all_iv(X: pd.DataFrame, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Calculate Information Value for all features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
    bins : int
        Number of bins for continuous variables
        
    Returns
    -------
    pd.DataFrame : IV values with interpretation
    """
    iv_values = []
    
    for col in X.columns:
        try:
            iv = calculate_iv(X, y, col, bins)
            
            # Interpretation
            if iv < 0.02:
                strength = 'Not Predictive'
            elif iv < 0.1:
                strength = 'Weak'
            elif iv < 0.3:
                strength = 'Medium'
            elif iv < 0.5:
                strength = 'Strong'
            else:
                strength = 'Suspicious (check for leakage)'
                
            iv_values.append({
                'feature': col,
                'iv': iv,
                'predictive_strength': strength
            })
        except Exception as e:
            iv_values.append({
                'feature': col,
                'iv': np.nan,
                'predictive_strength': 'Error',
                'error': str(e)
            })
    
    return pd.DataFrame(iv_values).sort_values('iv', ascending=False)


def calculate_mutual_information(
    X: pd.DataFrame, 
    y: np.ndarray,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate Mutual Information for all features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame : MI values for each feature
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_information': mi_scores
    }).sort_values('mutual_information', ascending=False)
    
    return mi_df


def correlation_filter(
    X: pd.DataFrame, 
    y: np.ndarray,
    threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Identify highly correlated feature pairs.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Binary target
    threshold : float
        Correlation threshold
        
    Returns
    -------
    tuple : (correlation matrix, list of high correlation pairs)
    """
    corr_matrix = X.corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    return corr_matrix, high_corr_pairs


# =============================================================================
# SELECTION METHODS (for use in Nested CV)
# =============================================================================

def lasso_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42
) -> List[str]:
    """
    LASSO (L1) regularization for variable selection.
    """
    # Scale features for regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        Cs=np.logspace(-4, 2, 30),
        cv=cv,
        scoring='roc_auc',
        max_iter=2000,
        random_state=random_state,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)
    
    selected_mask = model.coef_[0] != 0
    return X.columns[selected_mask].tolist()


def elastic_net_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    l1_ratio: float = 0.5
) -> List[str]:
    """
    Elastic Net regularization for variable selection.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {'alpha': np.logspace(-5, 1, 15)}
    
    sgd = SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',
        l1_ratio=l1_ratio,
        max_iter=2000,
        random_state=random_state
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = GridSearchCV(sgd, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_scaled, y)
    
    selected_mask = grid.best_estimator_.coef_[0] != 0
    return X.columns[selected_mask].tolist()


def rfe_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    min_features: int = 5
) -> List[str]:
    """
    Recursive Feature Elimination with CV.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    estimator = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )
    
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring='roc_auc',
        min_features_to_select=min_features,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X_scaled, y)
    
    return X.columns[rfecv.support_].tolist()


def forward_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    max_features: int = 15
) -> List[str]:
    """
    Sequential Forward Selection.
    """
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        estimator = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )
        
        sfs = SequentialFeatureSelector(
            estimator,
            k_features=(5, min(max_features, X.shape[1])),
            forward=True,
            floating=False,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sfs.fit(X_scaled, y)
        
        return list(sfs.k_feature_names_)
    except ImportError:
        # Fallback to simple forward selection
        return _simple_forward_selection(X, y, cv, random_state, max_features)


def _simple_forward_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    max_features: int = 15
) -> List[str]:
    """
    Simple forward selection without mlxtend.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    selected = []
    remaining = list(X.columns)
    best_score = 0
    
    while len(selected) < max_features and remaining:
        best_candidate = None
        best_candidate_score = best_score
        
        for candidate in remaining:
            features = selected + [candidate]
            
            model = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
            
            scores = cross_val_score(
                model, 
                X_scaled[features], 
                y, 
                cv=cv, 
                scoring='roc_auc'
            )
            score = scores.mean()
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = candidate
        
        if best_candidate is None or best_candidate_score <= best_score + 0.001:
            break
            
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score
    
    return selected


def mutual_info_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    top_k: int = 15
) -> List[str]:
    """
    Mutual Information based selection with CV validation.
    """
    # Calculate MI scores
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    # Get top-k features
    top_indices = np.argsort(mi_scores)[-top_k:]
    
    return X.columns[top_indices].tolist()


def tree_importance_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    top_k: int = 15
) -> List[str]:
    """
    Random Forest permutation importance with CV stability.
    """
    importance_accumulator = np.zeros(X.shape[1])
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        perm_imp = permutation_importance(
            rf, X_val, y_val,
            n_repeats=5,
            random_state=random_state,
            scoring='roc_auc',
            n_jobs=-1
        )
        importance_accumulator += perm_imp.importances_mean
    
    # Average importance across CV folds
    avg_importance = importance_accumulator / cv
    top_indices = np.argsort(avg_importance)[-top_k:]
    
    return X.columns[top_indices].tolist()


def iv_selection(
    X: pd.DataFrame, 
    y: np.ndarray, 
    cv: int = 5, 
    random_state: int = 42,
    iv_threshold: float = 0.02
) -> List[str]:
    """
    Information Value based selection.
    """
    iv_df = calculate_all_iv(X, y)
    
    # Select features above threshold
    selected = iv_df[iv_df['iv'] >= iv_threshold]['feature'].tolist()
    
    return selected


# =============================================================================
# NESTED CV FRAMEWORK
# =============================================================================

class NestedCVVariableSelector:
    """
    Nested cross-validation framework for unbiased variable selection
    and performance estimation in logistic regression.
    """
    
    def __init__(
        self,
        outer_cv: int = 5,
        inner_cv: int = 5,
        random_state: int = 42,
        min_votes: int = None,
        verbose: bool = True
    ):
        """
        Initialize the nested CV selector.
        
        Parameters
        ----------
        outer_cv : int
            Number of outer CV folds
        inner_cv : int
            Number of inner CV folds
        random_state : int
            Random seed
        min_votes : int
            Minimum votes for consensus (default: majority)
        verbose : bool
            Print progress
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.min_votes = min_votes
        self.verbose = verbose
        
        # Results storage
        self.outer_fold_results_ = []
        self.selected_features_per_fold_ = []
        self.method_selections_per_fold_ = []
        self.feature_selection_stability_ = None
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        selection_methods: Dict[str, Callable]
    ):
        """
        Run nested CV with ensemble variable selection.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Binary target
        selection_methods : dict
            Dictionary of {method_name: method_function}
        """
        outer_splitter = StratifiedKFold(
            n_splits=self.outer_cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Set default min_votes if not specified
        if self.min_votes is None:
            self.min_votes = len(selection_methods) // 2 + 1
        
        feature_names = X.columns.tolist()
        
        for fold_idx, (dev_idx, test_idx) in enumerate(outer_splitter.split(X, y)):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"OUTER FOLD {fold_idx + 1}/{self.outer_cv}")
                print(f"{'='*60}")
            
            # Split into development and held-out test
            X_dev, X_test = X.iloc[dev_idx].copy(), X.iloc[test_idx].copy()
            y_dev, y_test = y[dev_idx], y[test_idx]
            
            # ============================================
            # INNER LOOP: Variable selection on dev set only
            # ============================================
            fold_selections = {}
            
            for method_name, method_func in selection_methods.items():
                if self.verbose:
                    print(f"\n  Running {method_name}...")
                
                try:
                    selected = method_func(
                        X_dev, y_dev,
                        cv=self.inner_cv,
                        random_state=self.random_state
                    )
                    fold_selections[method_name] = selected
                    if self.verbose:
                        print(f"    Selected {len(selected)} features")
                except Exception as e:
                    if self.verbose:
                        print(f"    Failed: {e}")
                    fold_selections[method_name] = []
            
            self.method_selections_per_fold_.append(fold_selections)
            
            # Ensemble aggregation within this fold
            consensus_features = self._aggregate_selections(fold_selections)
            
            if self.verbose:
                print(f"\n  Consensus features ({len(consensus_features)}): {consensus_features[:5]}...")
            
            self.selected_features_per_fold_.append(consensus_features)
            
            # ============================================
            # Train final model and evaluate on TRUE test set
            # ============================================
            if len(consensus_features) == 0:
                if self.verbose:
                    print("  WARNING: No features selected, using all features")
                consensus_features = feature_names
            
            X_dev_selected = X_dev[consensus_features]
            X_test_selected = X_test[consensus_features]
            
            # Scale features
            scaler = StandardScaler()
            X_dev_scaled = scaler.fit_transform(X_dev_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Final model with light regularization for stability
            final_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )
            final_model.fit(X_dev_scaled, y_dev)
            
            # Evaluate on held-out test
            y_prob = final_model.predict_proba(X_test_scaled)[:, 1]
            
            fold_metrics = {
                'fold': fold_idx + 1,
                'n_features': len(consensus_features),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'brier_score': brier_score_loss(y_test, y_prob),
                'log_loss': log_loss(y_test, y_prob),
                'features': consensus_features
            }
            self.outer_fold_results_.append(fold_metrics)
            
            if self.verbose:
                print(f"\n  OUTER FOLD {fold_idx + 1} RESULTS:")
                print(f"    ROC-AUC: {fold_metrics['roc_auc']:.4f}")
                print(f"    Brier Score: {fold_metrics['brier_score']:.4f}")
        
        # Calculate feature stability across folds
        self._calculate_stability(feature_names)
        
        return self
    
    def _aggregate_selections(self, method_selections: Dict) -> List[str]:
        """Majority voting across selection methods."""
        all_features = []
        for features in method_selections.values():
            all_features.extend(features)
        
        vote_counts = Counter(all_features)
        consensus = [f for f, count in vote_counts.items() if count >= self.min_votes]
        
        return consensus
    
    def _calculate_stability(self, all_features: List[str]):
        """Calculate stability metrics including Kuncheva's index."""
        # Feature frequency across folds
        feature_counts = defaultdict(int)
        for fold_features in self.selected_features_per_fold_:
            for f in fold_features:
                feature_counts[f] += 1
        
        # Normalize by number of folds
        n_folds = len(self.selected_features_per_fold_)
        stability_scores = {
            f: count / n_folds
            for f, count in feature_counts.items()
        }
        
        # Kuncheva's index (pairwise consistency)
        n_total_features = len(all_features)
        kuncheva_values = []
        
        for (s1, s2) in combinations(self.selected_features_per_fold_, 2):
            set1, set2 = set(s1), set(s2)
            r = len(set1 & set2)  # intersection
            k1, k2 = len(set1), len(set2)
            
            if k1 == 0 or k2 == 0:
                continue
            
            # Kuncheva's index
            expected = (k1 * k2) / n_total_features if n_total_features > 0 else 0
            max_val = min(k1, k2) - expected
            if max_val > 0:
                ki = (r - expected) / max_val
                kuncheva_values.append(ki)
        
        self.feature_selection_stability_ = {
            'feature_frequencies': dict(sorted(
                stability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'kuncheva_index': np.mean(kuncheva_values) if kuncheva_values else 0.0,
            'n_stable_features_80': sum(1 for v in stability_scores.values() if v >= 0.8),
            'n_stable_features_60': sum(1 for v in stability_scores.values() if v >= 0.6)
        }
    
    def get_results_summary(self) -> Tuple[pd.DataFrame, Dict]:
        """Return summary of outer fold performance."""
        df = pd.DataFrame(self.outer_fold_results_)
        
        summary = {
            'mean_roc_auc': df['roc_auc'].mean(),
            'std_roc_auc': df['roc_auc'].std(),
            'mean_brier': df['brier_score'].mean(),
            'std_brier': df['brier_score'].std(),
            'mean_log_loss': df['log_loss'].mean(),
            'std_log_loss': df['log_loss'].std(),
            'mean_n_features': df['n_features'].mean(),
            'kuncheva_stability': self.feature_selection_stability_['kuncheva_index']
        }
        
        return df, summary
    
    def get_stable_features(self, threshold: float = 0.8) -> List[str]:
        """Return features selected in >= threshold fraction of outer folds."""
        return [
            f for f, freq in self.feature_selection_stability_['feature_frequencies'].items()
            if freq >= threshold
        ]
    
    def get_feature_votes(self) -> pd.DataFrame:
        """Get vote counts and frequencies for all features."""
        freqs = self.feature_selection_stability_['feature_frequencies']
        
        data = [
            {
                'feature': f,
                'selection_frequency': freq,
                'n_folds_selected': int(freq * self.outer_cv)
            }
            for f, freq in freqs.items()
        ]
        
        return pd.DataFrame(data).sort_values('selection_frequency', ascending=False)


# =============================================================================
# FINAL MODEL DIAGNOSTICS
# =============================================================================

def final_model_diagnostics(
    X: pd.DataFrame, 
    y: np.ndarray,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Run comprehensive diagnostics on final variable set.
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix
    y : np.ndarray
        Binary target
    feature_names : list
        Selected features to evaluate
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    dict : Comprehensive diagnostic results
    """
    from sklearn.model_selection import train_test_split
    
    X_selected = X[feature_names]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit statsmodels for detailed diagnostics
    X_train_sm = sm.add_constant(X_train_scaled)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_model = sm.Logit(y_train, X_train_sm).fit(disp=0, maxiter=200)
    
    # Sklearn model for predictions
    sk_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=random_state)
    sk_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_prob = sk_model.predict_proba(X_train_scaled)[:, 1]
    y_test_prob = sk_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate VIF on final features
    vif_df = calculate_vif(X_train)
    
    # EPV check
    epv = calculate_epv(y_train, len(feature_names))
    
    diagnostics = {
        'model_converged': sm_model.mle_retvals['converged'],
        'n_features': len(feature_names),
        'epv_ratio': epv['epv_ratio'],
        'meets_epv_10': epv['meets_minimum_10'],
        
        # Performance metrics
        'train_roc_auc': roc_auc_score(y_train, y_train_prob),
        'test_roc_auc': roc_auc_score(y_test, y_test_prob),
        'train_brier': brier_score_loss(y_train, y_train_prob),
        'test_brier': brier_score_loss(y_test, y_test_prob),
        
        # Overfitting check
        'auc_gap': roc_auc_score(y_train, y_train_prob) - roc_auc_score(y_test, y_test_prob),
        
        # VIF check
        'max_vif': vif_df['vif'].max(),
        'all_vif_below_5': (vif_df['vif'] < 5).all(),
        
        # Model statistics
        'pseudo_r2': sm_model.prsquared,
        'log_likelihood': sm_model.llf,
        'aic': sm_model.aic,
        'bic': sm_model.bic,
        
        # Coefficient summary
        'coefficients': pd.DataFrame({
            'feature': ['const'] + feature_names,
            'coef': sm_model.params,
            'std_err': sm_model.bse,
            'z_value': sm_model.tvalues,
            'p_value': sm_model.pvalues
        }),
        
        'vif_table': vif_df
    }
    
    return diagnostics


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_feature_importance_comparison(
    method_results: Dict[str, List[str]], 
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create a heatmap showing which features were selected by each method.
    """
    import matplotlib.pyplot as plt
    
    # Get all unique features
    all_features = set()
    for features in method_results.values():
        all_features.update(features)
    all_features = sorted(all_features)
    
    # Create binary matrix
    methods = list(method_results.keys())
    matrix = np.zeros((len(all_features), len(methods)))
    
    for j, method in enumerate(methods):
        for i, feature in enumerate(all_features):
            if feature in method_results[method]:
                matrix[i, j] = 1
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features)
    
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Selection by Method')
    
    plt.colorbar(im, ax=ax, label='Selected')
    plt.tight_layout()
    
    return fig, ax


def plot_stability_across_folds(
    nested_cv_results: NestedCVVariableSelector,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot feature selection stability across CV folds.
    """
    import matplotlib.pyplot as plt
    
    freq_df = nested_cv_results.get_feature_votes().head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2ecc71' if f >= 0.8 else '#f39c12' if f >= 0.6 else '#e74c3c' 
              for f in freq_df['selection_frequency']]
    
    bars = ax.barh(range(len(freq_df)), freq_df['selection_frequency'], color=colors)
    ax.set_yticks(range(len(freq_df)))
    ax.set_yticklabels(freq_df['feature'])
    ax.set_xlabel('Selection Frequency')
    ax.set_title(f'Top {top_n} Features by Selection Stability')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='60% threshold')
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig, ax


def plot_cv_performance(
    nested_cv_results: NestedCVVariableSelector,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Plot performance metrics across CV folds.
    """
    import matplotlib.pyplot as plt
    
    results_df, summary = nested_cv_results.get_results_summary()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC-AUC by fold
    ax1 = axes[0]
    ax1.bar(results_df['fold'], results_df['roc_auc'], color='steelblue', alpha=0.7)
    ax1.axhline(y=summary['mean_roc_auc'], color='red', linestyle='--', 
                label=f"Mean: {summary['mean_roc_auc']:.4f}")
    ax1.fill_between(
        [0.5, len(results_df) + 0.5],
        summary['mean_roc_auc'] - summary['std_roc_auc'],
        summary['mean_roc_auc'] + summary['std_roc_auc'],
        alpha=0.2, color='red'
    )
    ax1.set_xlabel('Outer Fold')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('ROC-AUC by Outer Fold')
    ax1.legend()
    
    # Number of features by fold
    ax2 = axes[1]
    ax2.bar(results_df['fold'], results_df['n_features'], color='coral', alpha=0.7)
    ax2.axhline(y=summary['mean_n_features'], color='red', linestyle='--',
                label=f"Mean: {summary['mean_n_features']:.1f}")
    ax2.set_xlabel('Outer Fold')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Features Selected by Outer Fold')
    ax2.legend()
    
    plt.tight_layout()
    
    return fig, axes
