"""03 Analysis: Decode Subjects.

This script contains the subject-level decoding pipeline used to evaluate
classification performance from GLM- and ICA/atlas-derived features.

Included functions:
- load_canica: load or compute the shared CanICA model
- fit_glm_per_run: fit the first-level GLM and persist contrasts
- project_contrasts_to_ica: map contrasts into ICA space
- build_dataset: stack run-level samples for classification
- make_classifier: construct the requested classifier
- decode_subject_with_gridsearch: run subject-specific decoding with CV tuning
"""

import warnings
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from nilearn.decomposition import CanICA
from utils import create_overview_df


# -----------------------------
# Configuration
# -----------------------------
base_path = Path('/gpfs01/bartels/user/smuehlinghaus/causalcoding')
data_dir = Path('/gpfs01/bartels/group/br_insideout/data/sourcedata')
ica_path = base_path / 'ICA'
################################
glm_base_path = base_path / 'GLM_pattern_estimation'
################################
# glm_base_path = base_path / 'GLM_switches'
################################
results_path = base_path / 'code/decoding/feature_selection'
################################
# results_path = base_path / 'code/decoding/switch_classification'

# ica and classifier configuration for unique file naming
n_ica_components = 1024  # number of components in canica object
ic_selection_classifier = 'lda'  # classifier used for ic selection (lda or rf)
ic_selection_task = 'br_replay'  # task used for ic selection (br_replay or face_house)
file_suffix = f'{n_ica_components}ICA_{ic_selection_classifier}_{ic_selection_task}'

# -----------------------------
# Helpers
# -----------------------------

def load_canica(n_ica_components):
    """Load the stored CanICA object."""
    import pickle as pkl

    # check if file exists
    ica_file = ica_path / f'canica_object_{n_ica_components}.pkl'
    if not ica_file.exists():
        print('CanICA object will be computed and saved...')

        func_filenames = []

        for subj_idx, session in enumerate(sessions):
            session_path = data_dir / session
            
            # get NIfTI files (fMRI data) - exclude hidden macOS metadata files
            nii_files = sorted([session_path / f for f in os.listdir(session_path) 
                            if f.endswith('.nii') and not f.startswith('._')])
            
            if nii_files:
                func_filenames.extend([str(f) for f in nii_files])

        canica = CanICA(
            n_components=n_ica_components,
            verbose=1,
            random_state=0,
            standardize="zscore_sample",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            # silence warnings about ICA not converging
            # consider increasing tolerance or the maximum number of iterations.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            # silence deprecation warnings from nilearn about future changes in default parameters
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            canica.fit(func_filenames)
        with open(ica_file, 'wb') as f:
            pkl.dump(canica, f)
    else:
        print(f"Loading CanICA object...")
        with open(ica_file, 'rb') as f:
            canica = pkl.load(f)
    return canica


def fit_glm_per_run(subject, design_matrix_files, nii_files, subject_glm_path, contrast_definitions, smoothing_fwhm, contrast_type='_t'):
    """Fit GLM per run and compute contrasts; returns contrast_maps[run_idx][contrast_name].
    Uses t-stat or effect_size maps and saves contrast results per subject x run.
    Loads existing contrasts if already computed.
    """
    import pickle as pkl
    
    contrast_maps = {}
    contrasts_dir = subject_glm_path / 'contrasts'
    contrasts_dir.mkdir(parents=True, exist_ok=True)

    for dm_file in design_matrix_files:
        run_idx = int(Path(dm_file).stem.split('run')[-1])

        if run_idx >= len(nii_files):
            continue

        # check if contrast file for run exists
        all_contrasts_exist = True
        for contrast_name in contrast_definitions.keys():
            output_file = contrasts_dir / f"run{run_idx}_{contrast_name}{contrast_type}.pkl"
            if not output_file.exists():
                all_contrasts_exist = False
                break

        contrast_maps[run_idx] = {}

        if all_contrasts_exist:
            # load contrasts
            print(f"  Loading existing contrasts for run {run_idx}")
            for contrast_name in contrast_definitions.keys():
                output_file = contrasts_dir / f"run{run_idx}_{contrast_name}{contrast_type}.pkl"
                with open(output_file, 'rb') as f:
                    result = pkl.load(f)
                    contrast_maps[run_idx][contrast_name] = result['effect_size']
        else:
            # otherwise compute contrasts
            print(f"  Computing contrasts for run {run_idx}")
            design_matrix = pd.read_csv(dm_file, index_col=0)
            nii_path = nii_files[run_idx]
            nifti_img = load_img(nii_path)
            n_volumes = nifti_img.shape[3]

            if design_matrix.shape[0] != n_volumes:
                min_len = min(design_matrix.shape[0], n_volumes)
                design_matrix = design_matrix.iloc[:min_len, :]

            fmri_glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm)
            fmri_glm = fmri_glm.fit(str(nii_path), design_matrices=design_matrix)

            for contrast_name, contrast_formula in contrast_definitions.items():
                if contrast_type == '_t':
                    result = fmri_glm.compute_contrast(contrast_formula, stat_type='t', output_type='all')
                else:
                    result = fmri_glm.compute_contrast(contrast_formula, stat_type='F', output_type='all')
                contrast_maps[run_idx][contrast_name] = result['effect_size']

                output_file = contrasts_dir / f"run{run_idx}_{contrast_name}{contrast_type}.pkl"
                with open(output_file, 'wb') as f:
                    pkl.dump(result, f)

    return contrast_maps


def project_contrasts_to_ica(contrast_maps, canica, ics_to_use=None):
    """Project t-stat contrast maps onto ICA space per run and contrast."""
    ica_contrasts = {}
    for run_idx, contrasts in contrast_maps.items():
        ica_contrasts[run_idx] = {}
        for contrast_name, contrast_img in contrasts.items():
            proj = canica.transform(contrast_img)
            if isinstance(proj, list):
                proj = np.vstack(proj)
            else:
                proj = np.asarray(proj)

            if ics_to_use is not None:
                proj = proj[:, ics_to_use]

            ica_contrasts[run_idx][contrast_name] = proj
    return ica_contrasts


def build_dataset(ica_contrasts, run_indices, cond_list, task="face_house"):
    """Build dataset with run groups for cross-validation."""
    x_vals, y_labels, groups = [], [], []

    if task == "face_house":
        for cond_prefix in cond_list:
            face_key = f"{cond_prefix}_face" 
            house_key = f"{cond_prefix}_house"
            for run_idx in run_indices:
                x_vals.append(ica_contrasts[run_idx][face_key])
                y_labels.append(1)  # face
                groups.append(run_idx)
                x_vals.append(ica_contrasts[run_idx][house_key])
                y_labels.append(0)  # house
                groups.append(run_idx)
    elif task == "switches":
        # cond_list should be condition prefixes: ['BR', 'Replay']
        # or None to use both BR and Replay
        if cond_list is None:
            cond_list = ['BR', 'Replay']
        
        for cond_prefix in cond_list:
            htf_key = f"{cond_prefix}_house_to_face"
            fth_key = f"{cond_prefix}_face_to_house"
            for run_idx in run_indices:
                x_vals.append(ica_contrasts[run_idx][htf_key])
                y_labels.append(1)  # house to face
                groups.append(run_idx)
                x_vals.append(ica_contrasts[run_idx][fth_key])
                y_labels.append(0)  # face to house
                groups.append(run_idx)
    elif task == "br_replay":
        br_prefixes = ["c1", "c3"]
        replay_prefixes = ["c2", "c4"]

        for run_idx in run_indices:
            for cond_prefix in br_prefixes:
                face_key = f"{cond_prefix}_face"
                house_key = f"{cond_prefix}_house"
                x_vals.append(ica_contrasts[run_idx][face_key])
                y_labels.append(1)  # br
                groups.append(run_idx)
                x_vals.append(ica_contrasts[run_idx][house_key])
                y_labels.append(1)  # br
                groups.append(run_idx)

            for cond_prefix in replay_prefixes:
                face_key = f"{cond_prefix}_face"
                house_key = f"{cond_prefix}_house"
                x_vals.append(ica_contrasts[run_idx][face_key])
                y_labels.append(0)  # replay
                groups.append(run_idx)
                x_vals.append(ica_contrasts[run_idx][house_key])
                y_labels.append(0)  # replay
                groups.append(run_idx)
    else:
        raise ValueError(f"Unknown task: {task}")

    return np.vstack(x_vals), np.array(y_labels), np.array(groups)


def make_classifier(classifier_type):
    """Return estimator and optional GridSearchCV."""
    if classifier_type == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    if classifier_type == 'rf':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    if classifier_type == 'logreg':
        return LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv_folds=5, 
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            scoring='balanced_accuracy'
        )

    raise ValueError(f"Unknown classifier: {classifier_type}")


def decode_subject_with_gridsearch(
    subject,
    contrast_maps,
    canica,
    ics_to_use=None,
    task="face_house",
    min_runs_for_gridsearch=2,
):
    """Decode face vs house using logistic regression with elastic net and grid search.
    
    Performs per-subject grid search for optimal hyperparameters using leave-one-run-out CV.
    
    Args:
        subject: Subject ID
        contrast_maps: Dict of contrast maps for this subject
        canica: Fitted CanICA object  
        ics_to_use: Which ICs to use (None = all)
        task: 'face_house' for face vs house classification
        
    Returns:
        DataFrame with results per condition (BR and Replay)
    """
    from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
    from sklearn.metrics import accuracy_score
    
    # Project contrasts to ICA space
    ica_contrasts = project_contrasts_to_ica(contrast_maps, canica, ics_to_use=ics_to_use)
    run_indices = sorted(ica_contrasts.keys())
    n_runs = len(run_indices)
    
    if n_runs < min_runs_for_gridsearch:
        print(
            f"  Skipping {subject}: only {n_runs} runs "
            f"(need >={min_runs_for_gridsearch} for grid search)"
        )
        return pd.DataFrame()
    
    n_ics = len(ics_to_use) if ics_to_use is not None else canica.components_img_.shape[-1]
    
    # Parameter grid for grid search
    # Use a reduced grid for low-run subjects to improve robustness.
    if n_runs < 4:
        param_grid = {
            'l1_ratio': [0.1, 0.5, 0.9],
            'C': [0.01, 0.1, 1.0, 10.0],
        }
    else:
        param_grid = {
            'l1_ratio': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
        }
    
    condition_types = {
        'BR': ['c1', 'c3'],
        'Replay': ['c2', 'c4'],
    }
    
    results = []
    
    for cond_type, cond_prefixes in condition_types.items():
        # Build full dataset for this condition
        X, y, groups = build_dataset(ica_contrasts, run_indices, cond_prefixes, task=task)
        
        # Grid search with leave-one-run-out CV
        logo = LeaveOneGroupOut()
        clf = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, random_state=42)
        
        grid_search = GridSearchCV(
            clf, 
            param_grid, 
            cv=logo, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(X, y, groups=groups)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"  {subject} - {cond_type}: best params={best_params}, CV score={best_score:.3f}")
        
        # Re-train with best parameters and collect fold accuracies
        best_clf = LogisticRegression(
            penalty='elasticnet', 
            solver='saga',
            l1_ratio=best_params['l1_ratio'],
            C=best_params['C'],
            max_iter=1000,
            random_state=42
        )
        
        accuracies = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            best_clf.fit(X_train, y_train)
            y_pred = best_clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        
        results.append({
            'subject': subject,
            'condition': cond_type,
            'classifier': 'logreg_elasticnet',
            'task': task,
            'n_ics': n_ics,
            'n_runs': n_runs,
            'n_samples': len(y),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'best_l1_ratio': best_params['l1_ratio'],
            'best_C': best_params['C'],
            'fold_accuracies': accuracies,
        })
    
    return pd.DataFrame(results)


def find_optimal_ics_pooled(all_contrast_maps, subjects, canica, classifier_type='lda', min_features=8, 
                           task='br_replay', split_conditions=False):
    """find optimal ics by pooling all subjects' data.
    
    uses rfecv to automatically find the optimal number and indices of ics.
    
    args:
        all_contrast_maps: dict of contrast maps per subject
        subjects: list of subject ids
        canica: fitted canica object
        classifier_type: 'lda' or 'rf'
        min_features: minimum number of ics to select
        task: 'br_replay' for BR vs Replay classification, 
              'face_house' for face vs house classification
        split_conditions: if True and task='face_house', run separate IC selection 
                         for BR cycles (c1, c3) and Replay cycles (c2, c4) independently
    
    returns:
        - if task='br_replay' or split_conditions=False: 
          dict with optimal_ics, n_ics, cv_accuracy, cv_results
        - if task='face_house' and split_conditions=True:
          dict with 'BR' and 'Replay' keys, each containing a result dict
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.feature_selection import RFECV, RFE
    
    def _run_rfecv_on_pooled_data(X_pooled, y_pooled, groups_pooled, condition_label=""):
        """helper function to run rfecv on pooled data"""
        print(f"\n{'='*70}")
        print(f"running rfecv for {condition_label}")
        print(f"{'='*70}")
        print(f"pooled dataset: {X_pooled.shape[0]} samples from {len(np.unique(groups_pooled))} subjects")
        print(f"total ics available: {X_pooled.shape[1]}")
        
        clf = make_classifier(classifier_type)
        logo = LeaveOneGroupOut()
        
        print(f"running rfecv with leave-one-subject-out cv (min_features={min_features})...")
        rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=logo,
            scoring=make_scorer(balanced_accuracy_score),
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        rfecv.fit(X_pooled, y_pooled, groups=groups_pooled)
        
        # prepare results summary
        results_summary = []
        for n_feat, score in zip(
            range(X_pooled.shape[1], min_features - 1, -1),
            rfecv.cv_results_['mean_test_score']
        ):
            results_summary.append({
                'n_features': n_feat,
                'cv_accuracy': score
            })
        
        # find maximum accuracy and select configuration with fewer ics if tied
        max_accuracy = max(r['cv_accuracy'] for r in results_summary)
        optimal_config = min(
            [r for r in results_summary if r['cv_accuracy'] == max_accuracy],
            key=lambda x: x['n_features']
        )
        
        n_features_selected = optimal_config['n_features']
        optimal_score = optimal_config['cv_accuracy']
        
        # refit rfe with optimal number to get specific ic indices
        rfe_optimal = RFE(
            estimator=make_classifier(classifier_type),
            n_features_to_select=n_features_selected,
            step=1
        )
        rfe_optimal.fit(X_pooled, y_pooled)
        selected_ics = np.where(rfe_optimal.support_)[0]
        
        # mark optimal in results
        for r in results_summary:
            r['is_optimal'] = r['n_features'] == n_features_selected
        
        print(f"\noptimal number of ics: {n_features_selected}")
        print(f"optimal ics: {selected_ics.tolist()}")
        print(f"cv accuracy with optimal ics: {optimal_score:.3f}")
        
        return {
            'optimal_ics': selected_ics,
            'n_ics': n_features_selected,
            'cv_accuracy': optimal_score,
            'cv_results': results_summary
        }
    
    # pool data based on task and split_conditions
    if task == 'br_replay' or (task == 'face_house' and not split_conditions):
        # single pooling: either br vs replay, or face vs house (all cycles together)
        print("\npooling data from all subjects for ic selection...")
        
        all_X = []
        all_y = []
        all_groups = []
        
        for subject in subjects:
            if subject not in all_contrast_maps:
                continue
                
            contrast_maps = all_contrast_maps[subject]
            ica_contrasts = project_contrasts_to_ica(contrast_maps, canica, ics_to_use=None)
            run_indices = sorted(ica_contrasts.keys())
            
            if len(run_indices) < 1:
                continue
            
            # build dataset based on task
            if task == 'br_replay':
                X_subj, y_subj, _ = build_dataset(ica_contrasts, run_indices, None, task="br_replay")
            elif task == 'switches':
                # For switches: use condition prefixes only ['BR', 'Replay']
                cond_list = None  # use both BR and Replay by default
                X_subj, y_subj, _ = build_dataset(ica_contrasts, run_indices, cond_list, task="switches")
            else:  # face_house, all cycles
                cond_list = ['c1', 'c2', 'c3', 'c4']
                X_subj, y_subj, _ = build_dataset(ica_contrasts, run_indices, cond_list, task="face_house")
            
            all_X.append(X_subj)
            all_y.append(y_subj)
            all_groups.append(np.full(len(y_subj), subject))
        
        X_pooled = np.vstack(all_X)
        y_pooled = np.hstack(all_y)
        groups_pooled = np.hstack(all_groups)
        
        condition_label = "BR vs Replay" if task == 'br_replay' else "Switches" if task == 'switches' else "Face vs House (all cycles)"
        return _run_rfecv_on_pooled_data(X_pooled, y_pooled, groups_pooled, condition_label)
    
    elif task == 'face_house' and split_conditions:
        # separate pooling for BR cycles (c1, c3) and Replay cycles (c2, c4)
        print("\npooling data from all subjects with split conditions (BR and Replay separate)...")
        
        results = {}
        
        for condition_name, cycle_prefixes in [('BR', ['c1', 'c3']), ('Replay', ['c2', 'c4'])]:
            all_X = []
            all_y = []
            all_groups = []
            
            for subject in subjects:
                if subject not in all_contrast_maps:
                    continue
                    
                contrast_maps = all_contrast_maps[subject]
                ica_contrasts = project_contrasts_to_ica(contrast_maps, canica, ics_to_use=None)
                run_indices = sorted(ica_contrasts.keys())
                
                if len(run_indices) < 1:
                    continue
                
                # build face vs house dataset for this condition only
                X_subj, y_subj, _ = build_dataset(ica_contrasts, run_indices, cycle_prefixes, task="face_house")
                
                all_X.append(X_subj)
                all_y.append(y_subj)
                all_groups.append(np.full(len(y_subj), subject))
            
            X_pooled = np.vstack(all_X)
            y_pooled = np.hstack(all_y)
            groups_pooled = np.hstack(all_groups)
            
            condition_label = f"Face vs House ({condition_name} cycles only)"
            results[condition_name] = _run_rfecv_on_pooled_data(X_pooled, y_pooled, groups_pooled, condition_label)
        
        return results
    
    else:
        raise ValueError(f"Invalid combination: task={task}, split_conditions={split_conditions}")


def decode_subject(subject, contrast_maps, canica, classifier_type='lda', ics_to_use=None, task="face_house"):
    """Decode face vs house (default) or br vs replay from ICA-projected t-stat maps with GroupShuffleSplit CV."""
    from sklearn.metrics import balanced_accuracy_score, make_scorer

    ica_contrasts = project_contrasts_to_ica(contrast_maps, canica, ics_to_use=ics_to_use)
    run_indices = sorted(ica_contrasts.keys())
    n_runs = len(run_indices)

    if n_runs < 2:
        raise ValueError(f"Subject {subject} has insufficient runs for CV: {n_runs}")
    
    # For switches task with leave-one-run-out CV and LDA:
    # Each run contributes only 2 samples (face_to_house, house_to_face) per condition
    # LDA requires n_samples > n_classes (2 classes)
    # Training on 1 run gives 2 samples which equals 2 classes -> LDA fails
    # Skip subjects with fewer than 3 runs
    if task == "switches" and classifier_type == "lda" and n_runs < 3:
        print(f"  Skipping {subject}: only {n_runs} runs (need >=3 for switches task with LDA)")
        return pd.DataFrame()

    results = []
    n_ics = len(ics_to_use) if ics_to_use is not None else canica.components_img_.shape[-1]

    if task == "face_house":
        condition_types = {
            'BR': ['c1', 'c3'],
            'Replay': ['c2', 'c4'],
        }
        condition_items = list(condition_types.items())

    elif task == "br_replay":
        condition_items = [("BR_vs_Replay", None)]
    elif task == "switches":
        # For switches: each condition uses just the prefix ['BR'] or ['Replay']
        # build_dataset will append _face_to_house and _house_to_face automatically
        condition_types = {
            'BR': ['BR'],
            'Replay': ['Replay']
        }
        condition_items = list(condition_types.items())
    else:
        raise ValueError(f"Unknown task: {task}")

    for cond_type, cond_prefixes in condition_items:
        accuracies = []
        
        # leave-one-run-out cv
        for test_run in run_indices:
            train_runs = [r for r in run_indices if r != test_run]
            
            X_train, y_train, _ = build_dataset(ica_contrasts, train_runs, cond_prefixes, task=task)
            X_test, y_test, _ = build_dataset(ica_contrasts, [test_run], cond_prefixes, task=task)
            
            clf_fold = make_classifier(classifier_type)
            clf_fold.fit(X_train, y_train)
            y_pred = clf_fold.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))

        results.append({
            'subject': subject,
            'condition': cond_type,
            'classifier': classifier_type,
            'task': task,
            'n_ics': n_ics,
            'n_runs': n_runs,
            'mean_balanced_accuracy': mean_acc,
            'std_balanced_accuracy': std_acc,
            'fold_accuracies': accuracies,
        })

    return pd.DataFrame(results)


def overall_mean_ci(subject_means):
    """Compute mean and CI for subject means."""
    subject_means = np.asarray(subject_means, dtype=float)

    overall_mean = np.average(subject_means)
    variance = np.average((subject_means - overall_mean) ** 2)
    n_eff = len(subject_means)
    sem = np.sqrt(variance / n_eff) if n_eff > 1 else 0.0
    z = 1.96  # approx 95% CI
    return overall_mean, z * sem


def plot_decoding_results_bar(lda_df=None, svm_df=None, nb_df=None, rf_df=None, title_suffix=""):
    import matplotlib.pyplot as plt

    classifier_names_map = {
        'LDA': (lda_df, 'blue'),
        'SVM': (svm_df, 'green'),
        'Naive Bayes': (nb_df, 'orange'),
        'Random Forest': (rf_df, 'red')
    }

    dfs_to_combine = []
    classifiers_order = []
    colors = {}

    for clf_name, (df, color) in classifier_names_map.items():
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy['classifier'] = clf_name
            dfs_to_combine.append(df_copy)
            classifiers_order.append(clf_name)
            colors[clf_name] = color

    if not dfs_to_combine:
        print("No data to plot!")
        return

    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    subjects_order = sorted(combined_df['subject'].unique())
    conditions_order = sorted(combined_df['condition'].unique())
    if not conditions_order:
        print("No conditions found to plot!")
        return

    n_classifiers = len(classifiers_order)
    n_conditions = len(conditions_order)
    fig, axes = plt.subplots(n_conditions, n_classifiers, figsize=(6 * n_classifiers, 5 * n_conditions))
    if n_conditions == 1 and n_classifiers == 1:
        axes = np.array([[axes]])
    elif n_conditions == 1:
        axes = axes.reshape(1, n_classifiers)
    elif n_classifiers == 1:
        axes = axes.reshape(n_conditions, 1)

    for row_idx, condition in enumerate(conditions_order):
        for col_idx, clf_name in enumerate(classifiers_order):
            ax = axes[row_idx, col_idx]
            clf_cond = combined_df[
                (combined_df['classifier'] == clf_name) & (combined_df['condition'] == condition)
            ]

            means = []
            ci_vals = []
            n_runs_list = []
            for subj in subjects_order:
                subj_data = clf_cond[clf_cond['subject'] == subj]
                if subj_data.empty:
                    means.append(np.nan)
                    ci_vals.append(np.nan)
                    n_runs_list.append(np.nan)
                    continue
                mean_acc = subj_data['mean_balanced_accuracy'].values[0]
                std_acc = subj_data['std_balanced_accuracy'].values[0]
                n_runs = subj_data['n_runs'].values[0]
                # for leave-one-run-out CV, n_splits = n_runs
                sem = std_acc / np.sqrt(max(n_runs, 1))
                ci = 1.96 * sem
                means.append(mean_acc)
                ci_vals.append(ci)
                n_runs_list.append(n_runs)

            x = np.arange(len(subjects_order))
            ax.bar(x, means, yerr=ci_vals, color=colors[clf_name], alpha=0.7, capsize=4)

            valid_mask = ~np.isnan(means)
            if np.any(valid_mask):
                valid_means = np.array(means)[valid_mask]
                overall_mean, overall_ci = overall_mean_ci(valid_means)
                ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
                           label=f'Avg: {overall_mean:.3f}')
                ax.fill_between([-0.5, len(subjects_order) - 0.5],
                                overall_mean - overall_ci,
                                overall_mean + overall_ci,
                                color='red', alpha=0.1)

            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Chance')

            ax.set_xticks(x)
            ax.set_xticklabels(subjects_order, rotation=45, ha='right')
            ax.set_ylabel('Balanced Accuracy')
            ax.set_xlabel('Subject')
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{clf_name} - {condition}", fontsize=14)
            ax.legend(loc='lower right', fontsize=8)

    if len(conditions_order) == 1 and conditions_order[0] == 'BR_vs_Replay':
        title_prefix = 'BR vs Replay'
    else:
        title_prefix = 'Face vs House'
    plt.suptitle(f'{title_prefix}{title_suffix}', fontsize=16, y=0.995)
    plt.tight_layout()
    # save figure before show to avoid blank output in some backends
    plt.savefig(results_path / f"decoding_results_bar{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.show()


def summarize_overall_results(results_df):
    """Compute mean accuracy for each classifier/condition."""
    summaries = []
    for (classifier, condition), df in results_df.groupby(['classifier', 'condition']):
        means = df['mean_balanced_accuracy'].values
        overall_mean, overall_ci = overall_mean_ci(means)
        summaries.append({
            'classifier': classifier,
            'condition': condition,
            'mean_balanced_accuracy': overall_mean,
            'ci_95': overall_ci,
        })
    return pd.DataFrame(summaries)

def plot_combined_results(results_df, n_ics):
    """plot combined decoding results for lda and rf classifiers in separate subplots"""
    import matplotlib.pyplot as plt
    from scipy import stats
    
    conditions = sorted(results_df['condition'].unique())
    n_conditions = len(conditions)
    
    # create subplots: 2 rows (lda, rf) x n_conditions columns
    fig, axes = plt.subplots(2, n_conditions, figsize=(8 * n_conditions, 10))
    if n_conditions == 1:
        axes = axes.reshape(2, 1)
    
    classifiers = ['lda', 'rf']
    clf_colors = {'lda': 'blue', 'rf': 'orange'}
    clf_labels = {'lda': 'LDA', 'rf': 'RF'}
    
    for col_idx, condition in enumerate(conditions):
        cond_data = results_df[results_df['condition'] == condition]
        subjects_order = sorted(cond_data['subject'].unique())
        
        for row_idx, clf in enumerate(classifiers):
            ax = axes[row_idx, col_idx]
            clf_data = cond_data[cond_data['classifier'] == clf]
            
            # collect accuracies per subject
            subject_accs = []
            subject_labels = []
            
            for subj in subjects_order:
                subj_data = clf_data[clf_data['subject'] == subj]
                if not subj_data.empty:
                    mean_acc = subj_data['mean_balanced_accuracy'].values[0]
                    subject_accs.append(mean_acc)
                    subject_labels.append(subj)
            
            if not subject_accs:
                continue
                
            x = np.arange(len(subject_accs))
            
            # plot individual subject accuracies as bars
            ax.bar(x, subject_accs, color=clf_colors[clf], alpha=0.6, 
                   label=f'{clf_labels[clf]} (subjects)')
            
            # compute overall mean across subjects and confidence interval
            mean_acc = np.mean(subject_accs)
            std_acc = np.std(subject_accs, ddof=1)
            n_subjects = len(subject_accs)
            sem = std_acc / np.sqrt(n_subjects)
            ci = stats.t.ppf(0.975, n_subjects - 1) * sem  # 95% ci
            
            # plot mean line with shaded confidence interval
            ax.axhline(y=mean_acc, color=clf_colors[clf], linestyle='-', 
                      linewidth=2.5, label=f'Mean: {mean_acc:.3f} ± {ci:.3f}')
            ax.axhspan(mean_acc - ci, mean_acc + ci, color=clf_colors[clf], 
                      alpha=0.2, label=f'95% CI')
            
            # plot chance level
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, 
                      label='Chance (0.5)')
            
            # formatting
            ax.set_xticks(x)
            ax.set_xticklabels(subject_labels, rotation=45, ha='right')
            ax.set_ylabel('Balanced Accuracy', fontsize=12)
            ax.set_xlabel('Subject', fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.set_title(f'{clf_labels[clf]} - {condition}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Face vs House Decoding (optimal {n_ics} ICs)', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # save figure
    fig_path = results_path / f'decoding_results_optimal_{n_ics}ICs_{file_suffix}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nsaved figure to: {fig_path}")
    plt.show()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":

    # selection of ICs after visual inspection
    #selected_ics = [2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 19]
    # test subjects for pipeline validation
    test_subjects = ['s01', 's02']
    smoothing_fwhm = 6.0

    contrast_definitions = {
        'c1_face': 'c1_face',
        'c1_house': 'c1_house',
        'c2_face': 'c2_face',
        'c2_house': 'c2_house',
        'c3_face': 'c3_face',
        'c3_house': 'c3_house',
        'c4_face': 'c4_face',
        'c4_house': 'c4_house',
    }

    # contrast_definitions = {
    #     'BR_face_to_house': 'BR_face_to_house',
    #     'BR_house_to_face': 'BR_house_to_face',
    #     'Replay_face_to_house': 'Replay_face_to_house',
    #     'Replay_house_to_face': 'Replay_house_to_face', 
    # }
    warnings.filterwarnings('ignore', category=UserWarning, message='.*sklearn.utils.parallel.delayed.*')

    # ensure required directories exist before running
    for required_path in [base_path, data_dir, ica_path, glm_base_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    # ensure output directory exists
    results_path.mkdir(parents=True, exist_ok=True)

    sessions = sorted([d for d in data_dir.iterdir() if d.name.startswith('s')])
    overview_df = create_overview_df([s.name for s in sessions])
    # use only test subjects for pipeline validation
    #subjects = test_subjects
    subjects = sorted(overview_df['session_folder'].unique())

    canica = load_canica(n_ica_components)
    
    print("="*70)
    print("STEP 1: Load all subjects' contrast maps")
    print("="*70)
    
    # load contrast maps for all subjects
    all_contrast_maps = {}
    
    for subject in subjects:
        if subject not in overview_df['session_folder'].values:
            print(f"skipping {subject}: not found in overview_df")
            continue

        subject_glm_path = glm_base_path / subject
        design_matrix_files = sorted(subject_glm_path.glob(f'design_matrix_pattern_{subject}_run*.csv')) # change to "pattern" or "switches" 
        session_path = data_dir / subject
        nii_files = sorted([f for f in session_path.glob('*.nii') if not f.name.startswith('._')])

        if not design_matrix_files:
            print(f"skipping {subject}: no design matrices found")
            continue

        print(f"loading {subject}...")
        contrast_maps = fit_glm_per_run(subject, design_matrix_files, nii_files, subject_glm_path, contrast_definitions, smoothing_fwhm)
        all_contrast_maps[subject] = contrast_maps
    
    # print("\n" + "="*70)
    # print(f"STEP 2: Find optimal ICs using pooled {ic_selection_task.replace('_', ' ').title()} data")
    # print("="*70)
    
    # print(f"IC selection configuration: {n_ica_components} ICA components, classifier={ic_selection_classifier}")
    # # find optimal ics by pooling all subjects' data for specified task
    # ic_selection_result = find_optimal_ics_pooled(
    #     all_contrast_maps, 
    #     subjects, 
    #     canica, 
    #     classifier_type=ic_selection_classifier, 
    #     min_features=10,
    #     task=ic_selection_task,
    #     split_conditions=False  # set to True to run separate selection 
    # )

    # # handle split_conditions results structure
    # if isinstance(ic_selection_result, dict) and 'BR' in ic_selection_result:
    #     # split_conditions=True case: separate results for BR and Replay
    #     print('\n=== IC Selection Results (Split Conditions) ===')
        
    #     all_cv_results = []
    #     optimal_ics_br = ic_selection_result['BR']['optimal_ics']
    #     optimal_ics_replay = ic_selection_result['Replay']['optimal_ics']
        
    #     # combine cv_results from both conditions into single dataframe
    #     for condition_name in ['BR', 'Replay']:
    #         condition_result = ic_selection_result[condition_name]
    #         cv_results = pd.DataFrame(condition_result['cv_results'])
    #         cv_results['condition'] = condition_name
    #         cv_results['classifier'] = ic_selection_classifier
    #         cv_results['task'] = ic_selection_task
    #         cv_results['min_features'] = 10
    #         cv_results['optimal_n_ics'] = condition_result['n_ics']
    #         cv_results['optimal_cv_accuracy'] = condition_result['cv_accuracy']
    #         cv_results['optimal_ics'] = str(condition_result['optimal_ics'].tolist())
    #         all_cv_results.append(cv_results)
            
    #         print(f"\n{condition_name}: {condition_result['n_ics']} optimal ICs")
    #         print(f"  ICs: {condition_result['optimal_ics'].tolist()}")
    #         print(f"  CV accuracy: {condition_result['cv_accuracy']:.3f}")
        
    #     # save combined results
    #     ic_selection_df = pd.concat(all_cv_results, ignore_index=True)
    #     ic_selection_df.to_csv(results_path / f'ic_selection_results_{file_suffix}.csv', index=False)
    #     print(f"\nsaved IC selection results to: {results_path / f'ic_selection_results_{file_suffix}.csv'}")
        
    #     # for downstream analysis: use BR ICs (can be changed to replay or union)
    #     optimal_ics = optimal_ics_br
    #     n_ics_label = f"{len(optimal_ics_br)}BR+{len(optimal_ics_replay)}Replay"
        
    # else:
    #     # single result case (br_replay or face_house without split)
    #     print('\n=== IC Selection Results ===')
        
    #     optimal_ics = ic_selection_result['optimal_ics']
        
    #     # create comprehensive dataframe with cv_results and metadata
    #     ic_selection_df = pd.DataFrame(ic_selection_result['cv_results'])
    #     ic_selection_df['classifier'] = ic_selection_classifier
    #     ic_selection_df['task'] = ic_selection_task
    #     ic_selection_df['min_features'] = 10
    #     ic_selection_df['optimal_n_ics'] = ic_selection_result['n_ics']
    #     ic_selection_df['optimal_cv_accuracy'] = ic_selection_result['cv_accuracy']
    #     ic_selection_df['optimal_ics'] = str(ic_selection_result['optimal_ics'].tolist())
        
    #     ic_selection_df.to_csv(results_path / f'ic_selection_results_{file_suffix}.csv', index=False)
    #     print(f"\nsaved IC selection results to: {results_path / f'ic_selection_results_{file_suffix}.csv'}")
        
    #     print(f"\nOptimal: {ic_selection_result['n_ics']} ICs")
    #     print(f"ICs: {optimal_ics.tolist()}")
    #     print(f"CV accuracy: {ic_selection_result['cv_accuracy']:.3f}")
        
    #     n_ics_label = f"{len(optimal_ics)}"
    
    # print(f"\nAccuracy by number of features:")
    # print(ic_selection_df[['n_features', 'cv_accuracy', 'is_optimal']].head(20).to_string(index=False))
    
    print("\n" + "="*70)
    print("STEP 3: Classification using ICs")
    print("="*70)

    # optimal ICs all
    optimal_ics = None
    n_ics_label = "all"
    
    all_results_lda = []
    all_results_rf = []
    all_results_logreg = []
    
    for subject in subjects:
        if subject not in all_contrast_maps:
            print(f"skipping {subject}: no contrast maps available")
            continue

        print(f"processing {subject}...")
        contrast_maps = all_contrast_maps[subject]

        # decode with optimal ics using lda and rf
        for clf in ['lda']:
            results = decode_subject(
                subject,
                contrast_maps,
                canica,
                classifier_type=clf,
                ics_to_use=optimal_ics, # None: all ICs are used
                task="face_house"
            )
            
            # Skip if empty (e.g., subject had too few runs)
            if results.empty:
                continue
            
            if clf == 'lda':
                all_results_lda.append(results)
            elif clf == 'rf':
                all_results_rf.append(results)
            elif clf == 'logreg':
                all_results_logreg.append(results)
    
    results_df_lda = pd.concat(all_results_lda, ignore_index=True) if all_results_lda else pd.DataFrame()
    results_df_rf = pd.concat(all_results_rf, ignore_index=True) if all_results_rf else pd.DataFrame()
    results_df_logreg = pd.concat(all_results_logreg, ignore_index=True) if all_results_logreg else pd.DataFrame()

    
    # determine n_ics for file naming (handle None case)
    if optimal_ics is None:
        n_ics_file = f"all{n_ica_components}"
    else:
        n_ics_file = n_ics_label
    
    output_path_combined = results_path / f'decoding_results_optimal_{n_ics_file}ICs_{file_suffix}.csv'
    results_df_lda.to_csv(output_path_combined, index=False)
    
    print(f"\nsaved LDA results to: {output_path_combined}")
    
    # print summaries
    if not results_df_lda.empty:
        summary = summarize_overall_results(results_df_lda)
        print("\noverall summary:")
        print(summary)
    
    # plot results
    if not results_df_lda.empty:
        n_ics_display = n_ics_label if optimal_ics is not None else f"all {n_ica_components}"
        plot_decoding_results_bar(lda_df=results_df_lda, title_suffix=f" (optimal {n_ics_display} ICs)")
    
    print("\n" + "="*70)
    print("analysis complete")
    print("="*70)
    
    # =========================================================================
    # TEST: Logistic Regression with Grid Search on s01 using ICA 1024
    # =========================================================================
    print("\n" + "="*70)
    print("TEST: Logistic Regression with Grid Search (s01, ICA 1024)")
    print("="*70)
    
    test_subject = 's01'
    if test_subject in all_contrast_maps:
        print(f"\nRunning grid search for {test_subject}...")
        test_results = decode_subject_with_gridsearch(
            subject=test_subject,
            contrast_maps=all_contrast_maps[test_subject],
            canica=canica,
            ics_to_use=None,  # Use all ICs (1024)
            task="face_house"
        )
        
        if not test_results.empty:
            print(f"\nResults for {test_subject}:")
            print(test_results[['subject', 'condition', 'n_ics', 'n_samples', 'mean_accuracy', 
                               'best_l1_ratio', 'best_C']].to_string(index=False))
            
            # Save results
            test_output_path = results_path / f'gridsearch_test_{test_subject}_1024ICs.csv'
            test_results.to_csv(test_output_path, index=False)
            print(f"\nSaved test results to: {test_output_path}")
        else:
            print(f"No results generated for {test_subject}")
    else:
        print(f"Subject {test_subject} not found in contrast maps")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)