#!/usr/bin/env python3
"""
04 Analysis: Create Joint Encoding-Decoding DataFrame (ICA or Atlas)

This script:
1. Can merge existing encoding and decoding outputs into one joint table
2. Works for both ICA and atlas feature spaces
3. Optionally runs full ICA encoding/decoding from raw ICA activations
4. Saves a standardized joint dataframe for downstream plotting/interpretation
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Paths
BASE_DIR = Path('/gpfs01/bartels/user/smuehlinghaus/causalcoding')
ENCODING_DIR = BASE_DIR / 'Results/Causal_Interpretation/Encoding'
DECODING_DIR = BASE_DIR / 'Results/Causal_Interpretation/Decoding'
OUTPUT_DIR = BASE_DIR / 'Results/Causal_Interpretation'

# Ensure output directories exist
ENCODING_DIR.mkdir(parents=True, exist_ok=True)
DECODING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(ENCODING_DIR / 'ICA').mkdir(parents=True, exist_ok=True)
(DECODING_DIR / 'ICA').mkdir(parents=True, exist_ok=True)


def load_ica_data():
    """Load ICA activations from pickle or CSV fallback."""
    pkl_path = ENCODING_DIR / 'ICA' / 'all_subjects_ICA.pkl'
    csv_path = ENCODING_DIR / 'ICA' / 'all_subjects_ICA.csv'
    
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
        print(f"✓ Loaded ICA data from {pkl_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded ICA data from {csv_path}")
    else:
        raise FileNotFoundError(
            f"Neither ICA input file exists:\n- {pkl_path}\n- {csv_path}"
        )
    
    # Parse comma-separated string back to numpy array when needed
    if isinstance(df['activations'].iloc[0], str):
        df['activations'] = df['activations'].apply(
            lambda s: np.array([float(x) for x in s.strip('[]').split(',')])
        )
    print(f"  Parsed {len(df['activations'].iloc[0])} activation values per sample")   
    
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def run_encoding_analysis(df_all_subjects_ica, alpha=0.05):
    """
    Perform encoding analysis using f_classif (ANOVA F-test)
    Tests which ICA components show significant differences between face/house conditions
    """
    print("\n" + "="*60)
    print("ENCODING ANALYSIS (f_classif)")
    print("="*60)
    
    encoding_results = []
    all_subjects = df_all_subjects_ica['subject'].unique()
    
    for subject in tqdm(all_subjects, desc="Encoding"):
        df_subject = df_all_subjects_ica[df_all_subjects_ica['subject'] == subject].copy()
        
        for cycle_pair in ['BR', 'Replay']:
            cycle_data = df_subject[df_subject['cycle_pair'] == cycle_pair].copy()
            
            # Need at least 4 samples (2 per class) for f_classif
            if len(cycle_data) < 4:
                print(f"  Skipping {subject} {cycle_pair}: insufficient samples ({len(cycle_data)} < 4)")
                continue
            
            X = np.vstack(cycle_data['activations'].values)
            y = cycle_data['condition'].map({'face': 1, 'house': 0}).values
            
            # Run f_classif
            stats, pvals = f_classif(X, y)
            significant_idx = np.where(pvals < alpha)[0]
            
            encoding_results.append({
                'subject': subject,
                'cycle_pair': cycle_pair,
                'n_samples': len(cycle_data),
                'num_significant_features': len(significant_idx),
                'significant_features': significant_idx.tolist(),
                'p_values': pvals.tolist()
            })
    
    df_encoding = pd.DataFrame(encoding_results)
    
    # Save encoding results
    encoding_output = ENCODING_DIR / 'ICA' / 'encoding_results_ICA.csv'
    df_encoding.to_csv(encoding_output, index=False)
    print(f"\n✓ Saved encoding results to {encoding_output}")
    
    # Summary statistics
    print("\nEncoding Results Summary:")
    summary = df_encoding.groupby('cycle_pair')['num_significant_features'].agg(['mean', 'std', 'count'])
    print(summary)
    print(f"\nTotal significant components: {df_encoding['num_significant_features'].sum()}")
    
    return df_encoding


def feature_importance_test(X, y, groups, n_permutations=1000):
    """
    Label permutation test for feature importance
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (ICA component activations)
    y : array-like, shape (n_samples,)
        Target labels (0=house, 1=face)
    groups : array-like, shape (n_samples,)
        Group labels for cross-validation (run numbers)
    n_permutations : int
        Number of permutations for null distribution
    
    Returns:
    --------
    original_mean_acc : float
        Mean accuracy across CV folds
    permuted_mean_acc : float
        Mean of permuted accuracies
    permuted_mean_coefs : array, shape (n_permutations, n_features)
        Permuted coefficient distributions
    original_mean_coefficients : array, shape (n_features,)
        Mean coefficients across CV folds
    p_values : list of float
        P-value for each feature
    """
    # Reshuffle data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    groups = groups[shuffle_idx]
    
    # Grid search for best hyperparameters
    param_grid = {
        'l1_ratio': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100]
    }
    
    logo = LeaveOneGroupOut()
    clf = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000)
    grid_search = GridSearchCV(clf, param_grid, cv=logo, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X, y, groups=groups)
    best_clf = grid_search.best_estimator_
    
    # Compute original performance
    accuracies = []
    original_coefficients = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        original_coefficients.append(best_clf.coef_.flatten())
    
    original_mean_acc = np.mean(accuracies)
    original_mean_coefficients = np.mean(original_coefficients, axis=0)
    
    # Permutation test
    mean_coefficient_permutations = []
    for i in range(n_permutations):
        y_permuted = shuffle(y, random_state=i)
        permuted_coefficients = []
        
        for train_idx, test_idx in logo.split(X, y_permuted, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_permuted[train_idx], y_permuted[test_idx]
            
            # Skip if only one class in training set
            if len(np.unique(y_train)) < 2:
                continue
            
            try:
                best_clf.fit(X_train, y_train)
                permuted_coefficients.append(best_clf.coef_.flatten())
            except ValueError:
                continue
        
        if len(permuted_coefficients) > 0:
            mean_fold_coefficients = np.mean(permuted_coefficients, axis=0)
            mean_coefficient_permutations.append(mean_fold_coefficients)
    
    permuted_mean_coefs = np.array(mean_coefficient_permutations)
    
    # Compute p-value for each feature
    p_values = []
    for feature_idx in range(X.shape[1]):
        p_value = np.mean([
            1 if abs(coef[feature_idx]) >= abs(original_mean_coefficients[feature_idx]) else 0 
            for coef in permuted_mean_coefs
        ])
        p_values.append(p_value)
    
    return original_mean_acc, permuted_mean_coefs.mean(), permuted_mean_coefs, original_mean_coefficients, p_values


def run_decoding_analysis(df_all_subjects_ica, n_permutations=1000):
    """
    Perform decoding analysis using label permutation test
    Tests which ICA components contribute to classification accuracy
    """
    print("\n" + "="*60)
    print("DECODING ANALYSIS (Label Permutation Test)")
    print("="*60)
    
    decoding_results = []
    all_subjects = df_all_subjects_ica['subject'].unique()
    
    for subject in tqdm(all_subjects, desc="Decoding"):
        subject_data = df_all_subjects_ica[df_all_subjects_ica['subject'] == subject]
        cycle_pairs = subject_data['cycle_pair'].unique()
        
        for cycle_pair in cycle_pairs:
            cycle_data = subject_data[subject_data['cycle_pair'] == cycle_pair]
            
            # Check minimum samples
            if len(cycle_data) < 4:
                print(f"  ⚠ Skipping {subject} {cycle_pair}: insufficient samples ({len(cycle_data)} < 4)")
                continue
            
            # Check minimum runs for LeaveOneGroupOut
            unique_runs = cycle_data['run'].unique()
            if len(unique_runs) < 2:
                print(f"  ⚠ Skipping {subject} {cycle_pair}: insufficient runs ({len(unique_runs)} < 2)")
                continue
            
            X = np.vstack(cycle_data['activations'].values)
            y = cycle_data['condition'].map({'face': 1, 'house': 0}).values
            groups = cycle_data['run'].values
            
            try:
                mean_acc, permuted_mean_acc, permuted_mean_coefs, original_mean_coefficients, p_values = \
                    feature_importance_test(X, y, groups, n_permutations=n_permutations)
                
                significant_features_idx = [i for i, p in enumerate(p_values) if p < 0.05]
                
                decoding_results.append({
                    'subject': subject,
                    'cycle_pair': cycle_pair,
                    'mean_accuracy': mean_acc,
                    'permuted_mean_accuracy': permuted_mean_acc,
                    'num_significant_features': len(significant_features_idx),
                    'significant_features': significant_features_idx,
                    'p_values': p_values
                })
                
            except Exception as e:
                print(f"  ⚠ Error for {subject} {cycle_pair}: {e}")
                continue
    
    df_decoding = pd.DataFrame(decoding_results)
    
    # Save decoding results
    decoding_output = DECODING_DIR / 'ICA' / 'decoding_results_ICA.csv'
    df_decoding.to_csv(decoding_output, index=False)
    print(f"\n✓ Saved decoding results to {decoding_output}")
    
    # Summary statistics
    print("\nDecoding Results Summary:")
    summary = df_decoding.groupby('cycle_pair')[['mean_accuracy', 'num_significant_features']].agg(['mean', 'std', 'count'])
    print(summary)
    
    return df_decoding


def create_joint_dataframe(df_encoding, df_decoding, output_csv=None):
    """
    Merge encoding and decoding results into joint dataframe
    """
    print("\n" + "="*60)
    print("CREATING JOINT DATAFRAME")
    print("="*60)
    
    df_joint = pd.merge(
        df_decoding[['subject', 'cycle_pair', 'mean_accuracy', 'num_significant_features', 'significant_features', 'p_values']],
        df_encoding[['subject', 'cycle_pair', 'n_samples', 'num_significant_features', 'significant_features', 'p_values']],
        on=['subject', 'cycle_pair'],
        how='inner',
        suffixes=('_dec', '_enc')
    )
    
    # Rename columns for clarity
    df_joint = df_joint.rename(columns={
        'num_significant_features_enc': 'n_enc_features',
        'significant_features_enc': 'enc_feature_idx',
        'p_values_enc': 'enc_p_values',
        'num_significant_features_dec': 'n_dec_features',
        'significant_features_dec': 'dec_feature_idx',
        'p_values_dec': 'dec_p_values'
    })
    
    # Save joint dataframe
    if output_csv is None:
        output_csv = OUTPUT_DIR / 'joint_enc_dec_ICA.csv'
    output_csv = Path(output_csv)
    df_joint.to_csv(output_csv, index=False)
    print(f"\n✓ Saved joint results to {output_csv}")
    
    print(f"\nJoint dataframe shape: {df_joint.shape}")
    print(f"Columns: {df_joint.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_joint.head())
    
    return df_joint


def _default_paths(feature_space='ica', atlas_resolution=400):
    """Return default encoding/decoding/output CSV paths for the selected feature space."""
    feature_space = feature_space.lower()

    if feature_space == 'ica':
        return (
            ENCODING_DIR / 'ICA' / 'encoding_results_ICA.csv',
            DECODING_DIR / 'ICA' / 'decoding_results_ICA.csv',
            OUTPUT_DIR / 'joint_enc_dec_ICA.csv',
        )

    if feature_space == 'atlas':
        return (
            ENCODING_DIR / 'Atlas' / 'df_ANOVA_feature_importance.csv',
            DECODING_DIR / 'Atlas' / f'df_feature_importance_schaefer_{atlas_resolution}regions.csv',
            OUTPUT_DIR / f'joint_enc_dec_atlas_{atlas_resolution}regions.csv',
        )

    raise ValueError("feature_space must be either 'ica' or 'atlas'")


def create_joint_dataframe_from_files(encoding_csv, decoding_csv, output_csv):
    """Load encoding/decoding CSV files and create a standardized joint dataframe."""
    encoding_csv = Path(encoding_csv)
    decoding_csv = Path(decoding_csv)

    if not encoding_csv.exists():
        raise FileNotFoundError(f"Encoding CSV not found: {encoding_csv}")
    if not decoding_csv.exists():
        raise FileNotFoundError(f"Decoding CSV not found: {decoding_csv}")

    df_encoding = pd.read_csv(encoding_csv)
    df_decoding = pd.read_csv(decoding_csv)

    encoding_required = {'subject', 'cycle_pair', 'n_samples', 'num_significant_features', 'significant_features', 'p_values'}
    decoding_required = {'subject', 'cycle_pair', 'mean_accuracy', 'num_significant_features', 'significant_features', 'p_values'}

    missing_enc = encoding_required - set(df_encoding.columns)
    missing_dec = decoding_required - set(df_decoding.columns)

    if missing_enc:
        raise ValueError(f"Encoding CSV is missing required columns: {sorted(missing_enc)}")
    if missing_dec:
        raise ValueError(f"Decoding CSV is missing required columns: {sorted(missing_dec)}")

    return create_joint_dataframe(df_encoding, df_decoding, output_csv=output_csv)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create joint encoding-decoding dataframe for ICA or atlas features.')
    parser.add_argument('--feature-space', choices=['ica', 'atlas'], default='ica', help='Feature space to process.')
    parser.add_argument('--mode', choices=['merge', 'run-ica-analysis'], default='merge',
                        help="'merge' combines existing CSV outputs; 'run-ica-analysis' computes ICA encoding/decoding first.")
    parser.add_argument('--encoding-csv', type=str, default=None, help='Path to encoding CSV (optional; uses defaults).')
    parser.add_argument('--decoding-csv', type=str, default=None, help='Path to decoding CSV (optional; uses defaults).')
    parser.add_argument('--output-csv', type=str, default=None, help='Path for output joint CSV (optional; uses defaults).')
    parser.add_argument('--atlas-resolution', type=int, default=400, help='Atlas region count used in atlas decoding filenames.')
    parser.add_argument('--n-permutations', type=int, default=1000, help='Permutation count for ICA decoding mode.')
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    default_encoding_csv, default_decoding_csv, default_output_csv = _default_paths(
        feature_space=args.feature_space,
        atlas_resolution=args.atlas_resolution,
    )

    encoding_csv = Path(args.encoding_csv) if args.encoding_csv else default_encoding_csv
    decoding_csv = Path(args.decoding_csv) if args.decoding_csv else default_decoding_csv
    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv

    print("="*60)
    print(f"JOINT ENCODING-DECODING ({args.feature_space.upper()})")
    print(f"Mode: {args.mode}")
    print("="*60)

    if args.mode == 'run-ica-analysis':
        if args.feature_space != 'ica':
            raise ValueError("run-ica-analysis mode is only supported for --feature-space ica")

        # Step 1: Load ICA data
        df_ica = load_ica_data()

        # Step 2: Run encoding analysis
        df_encoding = run_encoding_analysis(df_ica)

        # Step 3: Run decoding analysis
        df_decoding = run_decoding_analysis(df_ica, n_permutations=args.n_permutations)

        # Step 4: Create joint dataframe
        create_joint_dataframe(df_encoding, df_decoding, output_csv=output_csv)
    else:
        # Merge mode: use existing encoding/decoding tables for ICA or Atlas
        create_joint_dataframe_from_files(
            encoding_csv=encoding_csv,
            decoding_csv=decoding_csv,
            output_csv=output_csv,
        )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print(f"  • Encoding input: {encoding_csv}")
    print(f"  • Decoding input: {decoding_csv}")
    print(f"  • Joint output:   {output_csv}")


if __name__ == "__main__":
    main()
