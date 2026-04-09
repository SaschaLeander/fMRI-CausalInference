"""01 Preprocessing: Fit GLM.

This preprocessing pipeline builds run-level events, design matrices, and
first-level contrasts for the binocular rivalry / replay analysis.

Included functions:
- compute_adaptive_cutoffs: derive BR and Replay cutoffs from `.mat` timings
- create_events_df: build per-run condition tables for c1/c2/c3/c4 events
- find_motion_file: locate motion regressors for a run
- get_scan_times: align scan times and motion rows to the NIfTI length
- create_switch_events_df: build switch-event tables for BR and Replay
- create_design_matrix / create_design_matrices helpers: assemble nilearn-ready matrices
"""

# This script:
# - splits fMRI data from Zaretskaya et al. (2010) into c1/c2/c3/c4 face and house conditions
# - creates the run-level events table
# - creates a design matrix for each run of a subject
# - fits a General Linear Model using nilearn's FirstLevelModel
# - creates first-level contrasts
# - supports group-level and second-level GLM workflows

from matplotlib import colorbar
import scipy.io
import numpy as np
import pandas as pd
import os
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from scipy.stats import pearsonr
import re
import subprocess
from nilearn.image import load_img, index_img, mean_img
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.plotting import plot_design_matrix, plot_stat_map, view_img_on_surf
from utils import plot_temporal_distribution, create_overview_df

# data directory 
data_dir = Path('/your/path/') # update this to your data directory containing session folders (e.g., s01, s02, etc.)

# output directory 
output_dir = Path('/your/path/') # update this to your desired output directory

# load sessions s01-s31 
sessions = sorted([d for d in os.listdir(data_dir) if d.startswith('s')])

def compute_adaptive_cutoffs(session_folder, run_idx, gap_threshold_ms=100):
    """
    Compute adaptive cutoffs for BR and Replay conditions using gap detection + mean.
    
    Uses a two-pronged approach:
    1. Look for a significant gap (>gap_threshold_ms) in sorted onsets to define cutoff
    2. If no clear gap, fall back to mean of onsets
    
    Parameters:
    -----------
    session_folder : str
        Session folder name (e.g., 's01')
    run_idx : int
        Index of the run (0-based)
    gap_threshold_ms : float, default=100
        Minimum gap (ms) to consider as a natural division between early/late events
    
    Returns:
    --------
    br_cutoff, replay_cutoff : float, float
        Cutoff values computed from data
    dict
        Metadata about which method was used for each cutoff
    """
    session_path = data_dir / session_folder
    mat_files = sorted([f for f in os.listdir(session_path) 
                       if f.endswith('.mat') and not f.startswith('._')])
    
    mat_file_path = session_path / mat_files[run_idx]
    data = scipy.io.loadmat(mat_file_path)
    exp_log = data['exp_log'][0, 0]
    
    # extract onsets
    onsets_A_br = exp_log['onsets_A'].flatten()
    onsets_B_br = exp_log['onsets_B'].flatten()
    onsets_A_rep = exp_log['onsets_repA'].flatten()
    onsets_B_rep = exp_log['onsets_repB'].flatten()
    durs_A_rep = exp_log['durs_repA'].flatten() if 'durs_repA' in exp_log.dtype.names else np.zeros_like(onsets_A_rep)
    durs_B_rep = exp_log['durs_repB'].flatten() if 'durs_repB' in exp_log.dtype.names else np.zeros_like(onsets_B_rep)
    
    metadata = {}
    
    # ===== br cutoff =====
    br_onsets = np.concatenate([onsets_A_br, onsets_B_br])
    br_min = np.min(br_onsets)
    br_max = np.max(br_onsets)
    br_cutoff = (br_min + br_max) / 2
    metadata['br_method'] = 'midpoint'
    metadata['br_span'] = br_max - br_min
    
    # ===== replay cutoff =====
    replay_onsets_end = np.sort(np.concatenate([onsets_A_rep + durs_A_rep, onsets_B_rep + durs_B_rep]))
    replay_gaps = np.diff(replay_onsets_end)
    max_gap_idx_replay = np.argmax(replay_gaps)
    max_gap_replay = replay_gaps[max_gap_idx_replay]
    
    if max_gap_replay > gap_threshold_ms:
        # use gap-based cutoff: midpoint between the largest gap
        replay_cutoff = (replay_onsets_end[max_gap_idx_replay] + replay_onsets_end[max_gap_idx_replay + 1]) / 2
        metadata['replay_method'] = 'gap-based'
        metadata['replay_gap'] = max_gap_replay
    else:
        # fall back to mean
        replay_cutoff = np.mean(replay_onsets_end)
        metadata['replay_method'] = 'mean-based'
        metadata['replay_gap'] = max_gap_replay
    
    return br_cutoff, replay_cutoff, metadata

def create_events_df(session_folder, run_idx, br_cutoff_ms=None, replay_cutoff_ms=None):
    """
    Create events dataframe for a specific run, preserving temporal order.
    
    Events are classified into 8 conditions based on:
    - BR early (onset < br_cutoff) → c1_house/c1_face
    - BR late (onset >= br_cutoff) → c3_house/c3_face
    - Replay short (onset+dur < replay_cutoff) → c2_house/c2_face
    - Replay long (onset+dur >= replay_cutoff) → c4_house/c4_face
    
    Parameters:
    -----------
    session_folder : str
        Session folder name (e.g., 's01')
    run_idx : int
        Index of the run (0-based)
    br_cutoff_ms : float, optional
        Cutoff onset (ms) for BR events (early vs late block).
        If None, uses adaptive cutoff = mean of BR onsets.
    replay_cutoff_ms : float, optional
        Cutoff offset (onset+dur, ms) for Replay events (short vs long).
        If None, uses adaptive cutoff = mean of replay offset.
    
    Returns:
    --------
    pd.DataFrame
        Events dataframe with columns: onset, duration, trial_type
        Sorted by onset time (preserving temporal order)
    dict
        Metadata with actual cutoffs used
    """
    
    # load data
    session_path = data_dir / session_folder
    mat_files = sorted([f for f in os.listdir(session_path) 
                       if f.endswith('.mat') and not f.startswith('._')])
    
    if run_idx >= len(mat_files):
        raise ValueError(f"Run index {run_idx} out of range. Session has {len(mat_files)} runs.")
    
    mat_file_path = session_path / mat_files[run_idx]
    data = scipy.io.loadmat(mat_file_path)
    exp_log = data['exp_log'][0, 0]
    
    # extract event data
    onsets_A_br = exp_log['onsets_A'].flatten()
    onsets_B_br = exp_log['onsets_B'].flatten()
    durs_A_br = exp_log['durs_A'].flatten() if 'durs_A' in exp_log.dtype.names else np.zeros_like(onsets_A_br)
    durs_B_br = exp_log['durs_B'].flatten() if 'durs_B' in exp_log.dtype.names else np.zeros_like(onsets_B_br)
    
    onsets_A_rep = exp_log['onsets_repA'].flatten()
    onsets_B_rep = exp_log['onsets_repB'].flatten()
    durs_A_rep = exp_log['durs_repA'].flatten() if 'durs_repA' in exp_log.dtype.names else np.zeros_like(onsets_A_rep)
    durs_B_rep = exp_log['durs_repB'].flatten() if 'durs_repB' in exp_log.dtype.names else np.zeros_like(onsets_B_rep)
    
    # compute cutoffs if not provided
    if br_cutoff_ms is None or replay_cutoff_ms is None:
        br_cutoff_computed, replay_cutoff_computed, cutoff_metadata = compute_adaptive_cutoffs(session_folder, run_idx)
        if br_cutoff_ms is None:
            br_cutoff_ms = br_cutoff_computed
        if replay_cutoff_ms is None:
            replay_cutoff_ms = replay_cutoff_computed
    else:
        cutoff_metadata = {}
    
    # create list of all events with metadata
    events_list = []
    
    # br events - image a (house)
    for onset, dur in zip(onsets_A_br, durs_A_br):
        if dur > 0:
            trial_type = 'c1_house' if onset < br_cutoff_ms else 'c3_house'
            events_list.append({'onset': onset, 'duration': dur, 'trial_type': trial_type})
    
    # br events - image b (face)
    for onset, dur in zip(onsets_B_br, durs_B_br):
        if dur > 0:
            trial_type = 'c1_face' if onset < br_cutoff_ms else 'c3_face'
            events_list.append({'onset': onset, 'duration': dur, 'trial_type': trial_type})
    
    # replay events - image a (house)
    for onset, dur in zip(onsets_A_rep, durs_A_rep):
        if dur > 0:
            trial_type = 'c2_house' if (onset + dur) < replay_cutoff_ms else 'c4_house'
            events_list.append({'onset': onset, 'duration': dur, 'trial_type': trial_type})
    
    # replay events - image b (face)
    for onset, dur in zip(onsets_B_rep, durs_B_rep):
        if dur > 0:
            trial_type = 'c2_face' if (onset + dur) < replay_cutoff_ms else 'c4_face'
            events_list.append({'onset': onset, 'duration': dur, 'trial_type': trial_type})
    
    # create dataframe and sort by onset time (preserves temporal order from plot)
    events = pd.DataFrame(events_list)
    events = events.sort_values('onset').reset_index(drop=True)
    
    # return metadata
    metadata = {
        'br_cutoff': br_cutoff_ms,
        'replay_cutoff': replay_cutoff_ms,
        'n_events': len(events),
        'condition_counts': events['trial_type'].value_counts().to_dict(),
        'cutoff_method': cutoff_metadata
    }
    
    return events, metadata

def find_motion_file(session_folder, run_idx):
    """
    Find motion parameter file for a given session and run.
    
    Parameters:
    -----------
    session_folder : str
        Session folder name (e.g., 's01')
    run_idx : int
        Index of the run (0-based)
    scan_times : np.ndarray
        Array of scan times in seconds
    Returns:
    --------
    str
        Path to the motion parameter file
    """
    # load motion .txt file 
    session_num = session_folder.replace('s', '')

    # search for motion parameter file in the backup directory structure
    search_base = Path('/your/path/')

    # find all rp*.txt files recursively
    result = subprocess.run(
        ['find', str(search_base), '-name', 'rp*.txt'],
        capture_output=True,
        text=True
    )

    # filter for files matching the current session and run
    motion_files = [f for f in result.stdout.strip().split('\n') if f]

    # match based on session folder pattern (look for files in sess_X directories)
    matched_file = None
    for f in motion_files:
        # check if file path contains sess_ pattern matching our run
        if f'sess_{run_idx + 1}' in f or f'sess-{run_idx + 1}' in f:
            matched_file = f
            break

    if matched_file:
        motion_file = matched_file
    else:
        raise FileNotFoundError(f"Could not find motion file for {session_folder}, run {run_idx}")

    motion_data = np.loadtxt(motion_file)

    return motion_data

def get_scan_times(session_folder, run_idx, motion_data, nifti_path=None, verbose=False):
    """
    Get scan times for a given session and run, ensuring they match the NIfTI file length.
    
    Parameters:
    -----------
    session_folder : str
        Session folder name (e.g., 's01')
    run_idx : int
        Index of the run (0-based)
    motion_data : np.ndarray
        Motion parameter data
    nifti_path : str, optional
        Path to the NIfTI file to check the number of volumes
    verbose : bool, default=False
        Print debug information
    Returns:
    --------
    tuple
        (scan_times, motion_data) - both adjusted to match NIfTI length if needed
    """
    session_path = data_dir / session_folder
    mat_files = sorted([f for f in os.listdir(session_path) 
                       if f.endswith('.mat') and not f.startswith('._')])
    
    mat_file_path = session_path / mat_files[run_idx]
    data = scipy.io.loadmat(mat_file_path)
    exp_log = data['exp_log'][0, 0]
    
    scan_times = exp_log['scan_times'].flatten() / 1000.0  # convert to seconds
    
    # get NIfTI file length if path provided
    nifti_length = None
    if nifti_path is not None:
        nifti_img = load_img(nifti_path)
        nifti_length = nifti_img.shape[3]  # 4th dimension is time/volumes
        if verbose:
            print(f'NIfTI volumes: {nifti_length}')
    
    if verbose:
        print('before adaptation:', motion_data.shape, scan_times.shape)
    
    # determine target length (NIfTI length if available, otherwise motion data length)
    target_length = nifti_length if nifti_length is not None else motion_data.shape[0]
    
    # estimate TR from existing scan times
    tr = np.median(np.diff(scan_times))
    if verbose:
        print(f'estimated TR: {tr}')
    
    # adapt scan_times to target length
    if target_length > scan_times.shape[0]:
        # add scan times at the beginning to match target length
        diff = target_length - scan_times.shape[0]
        prepend_times = scan_times[0] - np.arange(diff, 0, -1) * tr
        scan_times = np.concatenate([prepend_times, scan_times])
    elif target_length < scan_times.shape[0]:
        # trim scan times from the end to match target length
        scan_times = scan_times[:target_length]
    
    # adapt motion_data to target length
    if target_length > motion_data.shape[0]:
        # add motion parameters at the beginning (replicate first row)
        diff = target_length - motion_data.shape[0]
        first_row = motion_data[0:1, :]  # keep as 2D
        prepend_motion = np.repeat(first_row, diff, axis=0)
        motion_data = np.vstack([prepend_motion, motion_data])
    elif target_length < motion_data.shape[0]:
        # trim motion data from the end to match target length
        motion_data = motion_data[:target_length, :]
    
    if verbose:
        print('after adaptation:', motion_data.shape, scan_times.shape)
    
    return scan_times, motion_data

def create_switch_events_df(session_folder, run_idx):
    """
    Create events dataframe for switches between face and house percepts.
    
    Identifies transitions in BR and Replay conditions:
    - BR_face_to_house: switch from face to house during BR
    - BR_house_to_face: switch from house to face during BR
    - Replay_face_to_house: switch from face to house during Replay
    - Replay_house_to_face: switch from house to face during Replay
    
    Parameters:
    -----------
    session_folder : str
        Session folder name (e.g., 's01')
    run_idx : int
        Index of the run (0-based)
    
    Returns:
    --------
    pd.DataFrame
        Events dataframe with columns: onset, duration, trial_type
        Duration is always 0 (instantaneous events)
        Sorted by onset time
    dict
        Metadata with switch counts
    """
    # load data
    session_path = data_dir / session_folder
    mat_files = sorted([f for f in os.listdir(session_path) 
                       if f.endswith('.mat') and not f.startswith('._')])
    
    if run_idx >= len(mat_files):
        raise ValueError(f"Run index {run_idx} out of range. Session has {len(mat_files)} runs.")
    
    mat_file_path = session_path / mat_files[run_idx]
    data = scipy.io.loadmat(mat_file_path)
    exp_log = data['exp_log'][0, 0]
    
    # extract event data
    onsets_A_br = exp_log['onsets_A'].flatten()
    onsets_B_br = exp_log['onsets_B'].flatten()
    durs_A_br = exp_log['durs_A'].flatten() if 'durs_A' in exp_log.dtype.names else np.zeros_like(onsets_A_br)
    durs_B_br = exp_log['durs_B'].flatten() if 'durs_B' in exp_log.dtype.names else np.zeros_like(onsets_B_br)
    
    onsets_A_rep = exp_log['onsets_repA'].flatten()
    onsets_B_rep = exp_log['onsets_repB'].flatten()
    durs_A_rep = exp_log['durs_repA'].flatten() if 'durs_repA' in exp_log.dtype.names else np.zeros_like(onsets_A_rep)
    durs_B_rep = exp_log['durs_repB'].flatten() if 'durs_repB' in exp_log.dtype.names else np.zeros_like(onsets_B_rep)
    
    # create combined event list for BR and Replay to identify switches
    events_list = []
    
    # BR events
    br_events = []
    for onset, dur in zip(onsets_A_br, durs_A_br):
        if dur > 0:
            br_events.append({'onset': onset, 'duration': dur, 'percept': 'house'})
    for onset, dur in zip(onsets_B_br, durs_B_br):
        if dur > 0:
            br_events.append({'onset': onset, 'duration': dur, 'percept': 'face'})
    br_events = sorted(br_events, key=lambda x: x['onset'])
    
    # Replay events
    replay_events = []
    for onset, dur in zip(onsets_A_rep, durs_A_rep):
        if dur > 0:
            replay_events.append({'onset': onset, 'duration': dur, 'percept': 'house'})
    for onset, dur in zip(onsets_B_rep, durs_B_rep):
        if dur > 0:
            replay_events.append({'onset': onset, 'duration': dur, 'percept': 'face'})
    replay_events = sorted(replay_events, key=lambda x: x['onset'])
    
    # identify switches in BR
    for i in range(1, len(br_events)):
        prev_percept = br_events[i-1]['percept']
        curr_percept = br_events[i]['percept']
        
        if prev_percept != curr_percept:
            # switch detected at the onset of the new percept
            switch_onset = br_events[i]['onset']
            trial_type = f"BR_{prev_percept}_to_{curr_percept}"
            events_list.append({
                'onset': switch_onset,
                'duration': 0,  # instantaneous event
                'trial_type': trial_type
            })
    
    # identify switches in Replay
    for i in range(1, len(replay_events)):
        prev_percept = replay_events[i-1]['percept']
        curr_percept = replay_events[i]['percept']
        
        if prev_percept != curr_percept:
            # switch detected at the onset of the new percept
            switch_onset = replay_events[i]['onset']
            trial_type = f"Replay_{prev_percept}_to_{curr_percept}"
            events_list.append({
                'onset': switch_onset,
                'duration': 0,  # instantaneous event
                'trial_type': trial_type
            })
    
    # create dataframe and sort by onset time
    events = pd.DataFrame(events_list)
    if len(events) > 0:
        events = events.sort_values('onset').reset_index(drop=True)
    
    # create metadata
    metadata = {
        'n_events': len(events),
        'condition_counts': events['trial_type'].value_counts().to_dict() if len(events) > 0 else {},
        'n_br_switches': sum(1 for e in events_list if e['trial_type'].startswith('BR_')),
        'n_replay_switches': sum(1 for e in events_list if e['trial_type'].startswith('Replay_'))
    }
    
    return events, metadata

def create_switch_design_matrices(subjects=[], runs=[], high_pass_thrsh=1/128, verbose=False):
    """
    Create design matrices for switch events (face↔house transitions).
    
    This function creates design matrices focused on perceptual switches rather than
    sustained percepts. All events have 0 duration (instantaneous).
    
    Parameters:
    -----------
    subjects : list of str or str
        List of subject/session folder names (e.g., ['s01', 's02']) or single subject as str.
    runs : list of int or int
        List of run indices (0-based) or single run index as int. If empty, processes all runs for each subject.
    high_pass_thrsh : float, default=1/128
        High-pass filter threshold for design matrix
    verbose : bool, default=False
        Print debug information

    Returns:
    --------
    dict
        Nested dictionary with structure: {subject: {run_idx: design_matrix_df}}
    """
    design_matrices = {}
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(runs, int):
        runs = [runs]
    
    for subject in subjects:
        nr_runs = overview_df[overview_df['session_folder'] == subject]['n_mat_files'].values[0]
        
        # get nifti files for this subject
        session_path = data_dir / subject
        nii_files = sorted([f for f in os.listdir(session_path) 
                           if f.endswith('.nii') and not f.startswith('._')])
        
        for run_idx in (runs if runs else range(nr_runs)):
            # get nifti file path
            nifti_path = str(session_path / nii_files[run_idx]) if run_idx < len(nii_files) else None
            
            # create switch events
            events_data, switch_metadata = create_switch_events_df(subject, run_idx)
            
            if len(events_data) == 0:
                if verbose:
                    print(f'Warning: No switch events found for {subject} run {run_idx}')
                continue
            
            motion_data = find_motion_file(subject, run_idx)
            scan_times, motion_data = get_scan_times(subject, run_idx, motion_data, nifti_path=nifti_path, verbose=verbose)
            
            design_mat = make_first_level_design_matrix(
                frame_times=scan_times, 
                events=events_data,
                hrf_model='spm',
                drift_model=None,
                high_pass=high_pass_thrsh,
                add_regs=motion_data,
                add_reg_names=[f'motion_param_{i+1}' for i in range(motion_data.shape[1])]
            )
            
            if subject not in design_matrices:
                design_matrices[subject] = {}
            
            design_matrices[subject][run_idx] = design_mat
            
            if verbose:
                print(f'{subject} run {run_idx}: {switch_metadata["n_events"]} switches detected')
                print(f'  Condition counts: {switch_metadata["condition_counts"]}')
    
    # save and plot design matrices
    glm_output_dir = output_dir / 'GLM_switches'
    glm_output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject in subjects:
        # create session-specific folder
        session_dir = glm_output_dir / subject
        session_dir.mkdir(parents=True, exist_ok=True)
        
        nr_runs = overview_df[overview_df['session_folder'] == subject]['n_mat_files'].values[0]
        
        # save design matrices as CSV files
        for run_idx in range(nr_runs):
            if run_idx in design_matrices.get(subject, {}):
                dm = design_matrices[subject][run_idx]
                csv_path = session_dir / f'design_matrix_switches_{subject}_run{run_idx}.csv'
                dm.to_csv(csv_path, index=True)
                if verbose:
                    print(f'saved design matrix: {csv_path}')
        
        # plot all design matrices for this subject
        fig_dm, ax_dm = plt.subplots(2, 3, figsize=(18, 10))
        for i, run_idx in enumerate(range(nr_runs)):
            if run_idx in design_matrices.get(subject, {}):
                dm = design_matrices[subject][run_idx]
                plot_design_matrix(dm, axes=ax_dm.flatten()[i])
                ax_dm.flatten()[i].set_title(f'{subject} - Run {run_idx} (Switches)')
        plt.tight_layout()
        dm_plot_path = session_dir / f'design_matrices_switches_{subject}.png'
        plt.savefig(dm_plot_path, dpi=100)
        plt.close(fig_dm)
        if verbose:
            print(f'saved design matrix plot: {dm_plot_path}')
        
        # create concatenated switch events plot for all runs
        events_list = []
        session_labels = []
        for run_idx in range(nr_runs):
            events_data, _ = create_switch_events_df(subject, run_idx)
            events_list.append(events_data)
            session_labels.append(f'{subject}_run{run_idx}')
        
        if events_list:
            fig_concat, ax_concat = plot_concatenated_events(events_list, session_labels)
            concat_plot_path = session_dir / f'concatenated_switch_events_{subject}.png'
            fig_concat.savefig(concat_plot_path, dpi=100)
            plt.close(fig_concat)
            if verbose:
                print(f'saved concatenated switch events plot: {concat_plot_path}')
    
    return design_matrices

def create_design_matrices(subjects=[], runs=[], high_pass_thrsh=1/128, verbose=False):
    """
    Create design matrices for specified subjects and runs.
    
    Parameters:
    -----------
    subjects : list of str or str
        List of subject/session folder names (e.g., ['s01', 's02']) or single subject as str.
    runs : list of int or int
        List of run indices (0-based) or single run index as int. If empty, processes all runs for each subject.

    Returns:
    --------
    dict
        Nested dictionary with structure: {subject: {run_idx: design_matrix_df}}
    """

    design_matrices = {}
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(runs, int):
        runs = [runs]
    for subject in subjects:
        nr_runs = overview_df[overview_df['session_folder'] == subject]['n_mat_files'].values[0]
        
        # get nifti files for this subject
        session_path = data_dir / subject
        nii_files = sorted([f for f in os.listdir(session_path) 
                           if f.endswith('.nii') and not f.startswith('._')])
        
        for run_idx in (runs if runs else range(nr_runs)):
            # get nifti file path
            nifti_path = str(session_path / nii_files[run_idx]) if run_idx < len(nii_files) else None
            
            events_data = create_events_df(subject, run_idx)[0]
            motion_data = find_motion_file(subject, run_idx)
            scan_times, motion_data = get_scan_times(subject, run_idx, motion_data, nifti_path=nifti_path, verbose=verbose)
            design_mat = make_first_level_design_matrix(
                frame_times=scan_times, 
                events=events_data,
                hrf_model='spm',
                drift_model=None,
                high_pass=high_pass_thrsh,
                add_regs=motion_data,
                add_reg_names=[f'motion_param_{i+1}' for i in range(motion_data.shape[1])]
                )
            if subject not in design_matrices:
                design_matrices[subject] = {}
            
            design_matrices[subject][run_idx] = design_mat
    
    # save and plot design matrices and concatenated events
    glm_output_dir = output_dir / 'GLM_pattern_estimation'
    glm_output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject in subjects:
        # create session-specific folder
        session_dir = glm_output_dir / subject
        session_dir.mkdir(parents=True, exist_ok=True)
        
        nr_runs = overview_df[overview_df['session_folder'] == subject]['n_mat_files'].values[0]
        
        # save design matrices as CSV files
        for run_idx in range(nr_runs):
            if run_idx in design_matrices.get(subject, {}):
                dm = design_matrices[subject][run_idx]
                csv_path = session_dir / f'design_matrix_pattern_{subject}_run{run_idx}.csv'
                dm.to_csv(csv_path, index=True)
                if verbose:
                    print(f'saved design matrix: {csv_path}')
        
        # plot all design matrices for this subject
        fig_dm, ax_dm = plt.subplots(2, 3, figsize=(18, 10))
        for i, run_idx in enumerate(range(nr_runs)):
            if run_idx in design_matrices.get(subject, {}):
                dm = design_matrices[subject][run_idx]
                plot_design_matrix(dm, axes=ax_dm.flatten()[i])
                ax_dm.flatten()[i].set_title(f'{subject} - Run {run_idx}')
        plt.tight_layout()
        dm_plot_path = session_dir / f'design_matrices_{subject}.png'
        plt.savefig(dm_plot_path, dpi=100)
        plt.close(fig_dm)
        if verbose:
            print(f'saved design matrix plot: {dm_plot_path}')
        
        # create concatenated events plot for all runs
        events_list = []
        session_labels = []
        for run_idx in range(nr_runs):
            events_data, _ = create_events_df(subject, run_idx)
            events_list.append(events_data)
            session_labels.append(f'{subject}_run{run_idx}')
        
        if events_list:
            fig_concat, ax_concat = plot_concatenated_events(events_list, session_labels)
            concat_plot_path = session_dir / f'concatenated_events_{subject}.png'
            fig_concat.savefig(concat_plot_path, dpi=100)
            plt.close(fig_concat)
            if verbose:
                print(f'saved concatenated events plot: {concat_plot_path}')
            
    return design_matrices

def plot_concatenated_events(events_list, session_labels, figsize=(18, 8)):
    """
    Plot temporal distribution of multiple events dataframes concatenated sequentially.
    
    Each session is displayed one after another on the x-axis with a time offset.
    Automatically detects event type (c1-c4 conditions or switch events) and adjusts layout.
    
    Parameters:
    -----------
    events_list : list of pd.DataFrame
        List of events dataframes, each with columns: onset, duration, trial_type
    session_labels : list of str
        List of session identifiers/names (e.g., ['s01_run1', 's01_run2', ...])
    figsize : tuple, default=(18, 8)
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # detect event type from first non-empty dataframe
    event_type = 'c1-c4'  # default
    for events in events_list:
        if len(events) > 0:
            first_trial_type = events['trial_type'].iloc[0]
            if 'BR_' in first_trial_type or 'Replay_' in first_trial_type:
                event_type = 'switches'
            break
    
    if event_type == 'switches':
        # define y-positions for switch conditions
        y_positions = {
            'BR_face_to_house': 3,
            'BR_house_to_face': 2.5,
            'Replay_face_to_house': 1.5,
            'Replay_house_to_face': 1,
        }
        
        # color mapping: to_house = blue, to_face = red
        colors = {
            'BR_face_to_house': 'blue',
            'BR_house_to_face': 'red',
            'Replay_face_to_house': 'blue',
            'Replay_house_to_face': 'red',
        }
        
        # line styles: BR = solid, Replay = dashed
        linestyles = {
            'BR_face_to_house': '-',
            'BR_house_to_face': '-',
            'Replay_face_to_house': '--',
            'Replay_house_to_face': '--',
        }
        
        # marker styles for switches
        markers = {
            'BR_face_to_house': '>',
            'BR_house_to_face': '<',
            'Replay_face_to_house': '>',
            'Replay_house_to_face': '<',
        }
        
        plot_title = 'Concatenated Switch Events'
        subtitle = '4 Conditions (BR/Replay × face→house/house→face)'
        
    else:  # c1-c4 conditions
        # define y-positions for each condition group
        y_positions = {
            'c1_house': 3.5,
            'c1_face': 3,
            'c2_house': 2.5,
            'c2_face': 2,
            'c3_house': 1.5,
            'c3_face': 1,
            'c4_house': 0.5,
            'c4_face': 0,
        }
        
        # color mapping: house = blue, face = red
        colors = {
            'c1_house': 'blue',
            'c1_face': 'red',
            'c2_house': 'blue',
            'c2_face': 'red',
            'c3_house': 'blue',
            'c3_face': 'red',
            'c4_house': 'blue',
            'c4_face': 'red',
        }
        
        # line styles: BR = solid, Replay = dashed
        linestyles = {
            'c1_house': '-', 'c1_face': '-',
            'c2_house': '--', 'c2_face': '--',
            'c3_house': '-', 'c3_face': '-',
            'c4_house': '--', 'c4_face': '--',
        }
        
        markers = None
        plot_title = 'Concatenated Events'
        subtitle = '8 Conditions (BR early/late × Replay short/long × House/Face)'
    
    bar_height = 0.25
    
    # calculate cumulative offsets for each session
    session_offsets = [0]
    for i, events in enumerate(events_list[:-1]):
        if len(events) > 0:
            max_time = events['onset'].max() + events['duration'].max()
        else:
            max_time = 0
        session_offsets.append(session_offsets[-1] + max_time + 50)  # 50ms gap between sessions
    
    # plot events from all sessions with offsets
    for session_idx, (events, label, offset) in enumerate(zip(events_list, session_labels, session_offsets)):
        for _, row in events.iterrows():
            trial_type = row['trial_type']
            
            # skip if trial_type not in our mapping (shouldn't happen but be safe)
            if trial_type not in y_positions:
                continue
                
            onset = row['onset'] + offset
            duration = row['duration']
            y_pos = y_positions[trial_type]
            color = colors[trial_type]
            linestyle = linestyles[trial_type]
            
            if event_type == 'switches':
                # for switches (duration=0), plot as markers with vertical lines
                marker = markers[trial_type]
                ax.plot([onset, onset], [y_pos - bar_height/2, y_pos + bar_height/2], 
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.7)
                ax.plot(onset, y_pos, marker=marker, color=color, markersize=8, alpha=0.9)
            else:
                # for c1-c4 conditions, plot onset as vertical line
                ax.plot([onset, onset], [y_pos - bar_height/2, y_pos + bar_height/2], 
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.7)
                
                # plot duration as horizontal bar
                if duration > 0:
                    ax.barh(y_pos, duration, height=bar_height, left=onset, 
                           color=color, alpha=0.3, edgecolor=color, linewidth=0.5)
    
    # add session boundaries and labels
    for session_idx, offset in enumerate(session_offsets):
        if session_idx > 0:
            ax.axvline(offset - 25, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        if session_idx < len(session_offsets):
            # calculate mid-point for label placement
            if session_idx < len(session_offsets) - 1:
                mid_offset = (session_offsets[session_idx] + session_offsets[session_idx + 1]) / 2
            else:
                if len(events_list[session_idx]) > 0:
                    max_time = events_list[session_idx]['onset'].max() + events_list[session_idx]['duration'].max()
                else:
                    max_time = 0
                mid_offset = (session_offsets[session_idx] + session_offsets[session_idx] + max_time) / 2
            y_label_pos = -1.2 if event_type == 'c1-c4' else -0.5
            ax.text(mid_offset, y_label_pos, session_labels[session_idx], ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
    
    # formatting
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=10)
    ax.set_xlabel('Time (ms, offset per session)', fontsize=12)
    ax.set_title(f'{plot_title}: {", ".join(session_labels)}\n{subtitle}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    if event_type == 'switches':
        ax.set_ylim(-1, 3.5)
        # add legend for switches
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, marker='>', markersize=8, label='Switch to House'),
            Line2D([0], [0], color='red', lw=2, marker='<', markersize=8, label='Switch to Face'),
            Line2D([0], [0], color='gray', lw=2, linestyle='-', label='BR'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Replay'),
        ]
    else:
        ax.set_ylim(-1.5, 4.5)
        # add legend for c1-c4 conditions
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='House (Image A)'),
            Line2D([0], [0], color='red', lw=2, label='Face (Image B)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='-', label='BR (early/late)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Replay (short/long)'),
            Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, edgecolor='gray', label='Duration')
        ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig, ax

def fit_glm_and_compute_contrasts(subjects, design_matrices, overview_df, smoothing_fwhm=8.0, switches=False, verbose=False):
    """
    Fit GLM for each subject and compute contrasts.
    
    Parameters:
    -----------
    subjects : list of str
        List of subject/session folder names
    design_matrices : dict
        Nested dictionary with design matrices {subject: {run_idx: design_matrix_df}}
    overview_df : pd.DataFrame
        Overview dataframe with session information
    smoothing_fwhm : float, default=8.0
        FWHM for spatial smoothing
    verbose : bool, default=False
        Print progress information
        
    Returns:
    --------
    dict
        GLM results for each subject {subject: fmri_glm}
    """
    
    # define contrasts
    if switches:
        contrasts = {
            'BR_vs_replay': 'BR_face_to_house + BR_house_to_face - Replay_face_to_house - Replay_house_to_face',
            'replay_vs_BR': 'Replay_face_to_house + Replay_house_to_face - BR_face_to_house - BR_house_to_face',
        }
        glm_output_dir = output_dir / 'GLM_switches'
        glm_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        glm_output_dir = output_dir / 'GLM_pattern_estimation'
        glm_output_dir.mkdir(parents=True, exist_ok=True)
        contrasts = {
            'c1_face': 'c1_face',
            'c1_house': 'c1_house',
            'c2_face': 'c2_face',
            'c2_house': 'c2_house',
            'c3_face': 'c3_face',
            'c3_house': 'c3_house',
            'c4_face': 'c4_face',
            'c4_house': 'c4_house',
        }
    glm_results = {}
    
    for subject in subjects:
        if verbose:
            print(f'\nprocessing subject: {subject}')
        
        # get nifti files for this subject
        session_path = Path('/gpfs01/bartels/group/br_insideout/data/sourcedata') / subject
        nii_files = sorted([f for f in os.listdir(session_path) 
                           if f.endswith('.nii') and not f.startswith('._')])
        
        nr_runs = overview_df[overview_df['session_folder'] == subject]['n_mat_files'].values[0]
        
        # Collect all nii files and design matrices for all runs
        all_nii_files = []
        all_design_matrices = []
        for run_idx in range(nr_runs):
            if run_idx < len(nii_files) and run_idx in design_matrices.get(subject, {}):
                nii_file_path = str(session_path / nii_files[run_idx])
                print(nii_file_path)
                design_mat = design_matrices[subject][run_idx]
                all_nii_files.append(nii_file_path)
                all_design_matrices.append(design_mat)
        
        # Fit GLM with all runs at once
        if all_nii_files:
            # compute mean image for plotting (using first run)
            mean_image = mean_img(all_nii_files[0])
            
            # plot parameters
            plot_param = {
                "vmin": 0,
                "display_mode": "z",
                "cut_coords": 3,
                'bg_img': mean_image,
                "cmap": "inferno",
                "transparency_range": [0, 10],
            }
            
            # fit GLM with all runs
            fmri_glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm)
            fmri_glm = fmri_glm.fit(all_nii_files, design_matrices=all_design_matrices)
            
            if verbose:
                print(f'  fitted GLM for {len(all_nii_files)} runs')
            
            # compute and plot contrasts
            fig_contrasts, axes = plt.subplots(2, 1 if switches else 4, figsize=(12, 10))
            axes = axes.flatten()

            session_dir = glm_output_dir / subject
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # compute contrasts across all runs
            for idx, (contrast_name, contrast_expr) in enumerate(contrasts.items()):
                if verbose:
                    print(f'  computing contrast across all runs: {contrast_name}')
                results = fmri_glm.compute_contrast(
                    contrast_expr,
                    output_type="all"
                )
                
                # save effect size map for second-level analysis
                effect_size_path = session_dir / f'contrast_{contrast_name}.nii.gz'
                nib.save(results, effect_size_path)
                if verbose:
                    print(f'    saved effect size map: {effect_size_path}')
                
                # plot stat map
                plot_stat_map(
                    results["stat"],
                    title=f'{subject}: {contrast_name}',
                    transparency=results["z_score"],
                    axes=axes[idx],
                    **plot_param,
                )
                # save interactive viewer
                view_img_on_surf(
                    results["stat"],
                    colorbar=True,
                    title=f'{subject}: {contrast_name}',
                    cmap='inferno',
                ).save_as_html(
                    session_dir / f'contrast_switches_{contrast_name.replace(" ", "_")}_{subject}_all_runs.html' if switches else session_dir / f'contrast_pattern_{contrast_name.replace(" ", "_")}_{subject}_all_runs.html'
                )

            contrast_plot_path = session_dir / f'contrasts_switches_{subject}_all_runs.png' if switches else session_dir / f'contrasts_pattern_{subject}_all_runs.png'
            plt.savefig(contrast_plot_path, dpi=100)
            plt.close(fig_contrasts)
            if verbose:
                print(f'  saved contrast plot: {contrast_plot_path}')
        
        glm_results[subject] = fmri_glm if all_nii_files else None
    
    return glm_results

def fit_second_level_glm(subjects, smoothing_fwhm=8.0, switches=False, verbose=False):
    """
    Creates second-level design matrix, fits second-level GLM and computes specified contrasts

    Parameters:
    -----------
    subjects : list of str
        List of subject/session folder names
    design_matrices : dict
        Nested dictionary with design matrices {subject: {run_idx: design_matrix_df}}
    overview_df : pd.DataFrame
        Overview dataframe with session information
    smoothing_fwhm : float, default=8.0
        FWHM for spatial smoothing
    switches : bool, default=False
        Whether to use switch-based contrasts
    verbose : bool, default=False

    Returns:
    --------
    dict
        Second-level GLM results
    """
    from nilearn.glm.second_level import SecondLevelModel
    import nibabel as nib
    
    # define contrasts based on analysis type
    if switches:
        contrasts = {
            'BR_vs_replay': 'BR_face_to_house + BR_house_to_face - Replay_face_to_house - Replay_house_to_face',
            'replay_vs_BR': 'Replay_face_to_house + Replay_house_to_face - BR_face_to_house - BR_house_to_face',
        }
        glm_output_dir = output_dir / 'GLM_switches'
    else:
        contrasts = {
            'BR_face_vs_house': '(c1_face + c3_face) - (c1_house + c3_house)',
            'Replay_face_vs_house': '(c2_face + c4_face) - (c2_house + c4_house)',
        }
        glm_output_dir = output_dir / 'GLM_pattern_estimation'
    
    glm_output_dir.mkdir(parents=True, exist_ok=True)
    
    # load existing first-level contrast maps
    contrast_maps = {contrast_name: [] for contrast_name in contrasts.keys()}
    
    if verbose:
        print("loading first-level contrast maps...")
    
    for subject in subjects:
        if verbose:
            print(f'  loading maps for subject: {subject}')
        
        # look for saved contrast maps in subject directory
        subject_dir = glm_output_dir / subject
        
        if not subject_dir.exists():
            if verbose:
                print(f'    WARNING: subject directory not found: {subject_dir}')
            continue
        
        for contrast_name in contrasts.keys():
            # standard naming from fit_glm_and_compute_contrasts
            contrast_file = subject_dir / f'contrast_{contrast_name}_effect_size.nii.gz'
            
            if contrast_file.exists():
                contrast_maps[contrast_name].append(str(contrast_file))
                if verbose:
                    print(f'    loaded {contrast_name}')
            else:
                if verbose:
                    print(f'    WARNING: could not find effect size map for {contrast_name} in {subject_dir}')
    
    # fit second-level GLM for each contrast
    if verbose:
        print("\nfitting second-level GLM...")
    
    second_level_results = {}
    
    for contrast_name, maps in contrast_maps.items():
        if len(maps) > 0:
            if verbose:
                print(f'\nprocessing second-level for contrast: {contrast_name}')
                print(f'  number of subjects: {len(maps)}')
            
            # create simple design matrix (intercept only for one-sample t-test)
            design_matrix_2nd = pd.DataFrame([1] * len(maps), columns=['intercept'])
            
            # fit second-level model
            second_level_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
            second_level_model = second_level_model.fit(maps, design_matrix=design_matrix_2nd)
            
            # compute group-level contrast (intercept = one-sample t-test)
            group_contrast = second_level_model.compute_contrast(output_type='all')
            
            # save results
            second_level_results[contrast_name] = {
                'model': second_level_model,
                'contrast': group_contrast
            }
            
            # create visualization
            plot_param = {
                "vmin": 0,
                "display_mode": "z",
                "cut_coords": 5,
                "cmap": "inferno",
                "threshold": 3.0,
            }
            
            fig = plt.figure(figsize=(12, 4))
            plot_stat_map(
                group_contrast['stat'],
                title=f'Second-Level: {contrast_name} (n={len(maps)} subjects)',
                **plot_param,
                figure=fig
            )
            
            # save plot
            second_level_dir = glm_output_dir / 'second_level'
            second_level_dir.mkdir(parents=True, exist_ok=True)
            plot_path = second_level_dir / f'second_level_{contrast_name}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if verbose:
                print(f'  saved second-level plot: {plot_path}')
            
            # save stat map
            stat_map_path = second_level_dir / f'second_level_{contrast_name}_stat.nii.gz'
            nib.save(group_contrast['stat'], stat_map_path)
            if verbose:
                print(f'  saved stat map: {stat_map_path}')
            
            # save interactive viewer
            html_view = view_img_on_surf(
                group_contrast['stat'],
                colorbar=True,
                title=f'Second-Level: {contrast_name}',
                cmap='inferno',
                threshold=3.0
            )
            html_path = second_level_dir / f'second_level_{contrast_name}.html'
            html_view.save_as_html(html_path)
            
            if verbose:
                print(f'  saved interactive viewer: {html_path}')
        else:
            if verbose:
                print(f'\nWARNING: no maps found for contrast {contrast_name}, skipping')
    
    return second_level_results

if __name__ == "__main__":
    # create overview over sessions and runs
    overview_df = create_overview_df(sessions)
    # all sessions folders
    all_subjs = ['s01', 's07', 's08', 's09', 's10',
     's11', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
     's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 
     's30', 's31']
    # all_sessions = overview_df.session_folder.unique().tolist()
    # session_idx = ['s03', 's02', 's04', 's05',] 
    # define run or process all runs if run_index=None
    run_index = None
    
    # # create pattern estimation design_matrices
    print("creating design matrices...")
    print(f"  subjects: {len(all_subjs)}")
    design_matrices = create_design_matrices(subjects=all_subjs, runs=run_index, verbose=False)

    # fit GLM and compute pattern estimation contrasts
    # print("\nfitting GLM and computing contrasts...")
    # glm_results = fit_glm_and_compute_contrasts(
    #     subjects=session_idx, 
    #     design_matrices=design_matrices,  
    #     overview_df=overview_df,
    #     switches=False,
    #     verbose=False
    # )
    
    # create switch-based design matrices
    # print("\ncreating switch-based design matrices...")
    # switch_design_matrices = create_switch_design_matrices(subjects=all_subjs, runs=run_index, verbose=False)
    
    # # fit GLM and compute BR / Replay contrasts
    # print("\nfitting GLM and computing contrasts...")
    # glm_results = fit_glm_and_compute_contrasts(
    #     subjects=all_subjs, 
    #     design_matrices=switch_design_matrices, 
    #     overview_df=overview_df,
    #     switches=True,
    #     verbose=False
    # )
    # # fit second-level GLM for switch-based contrasts
    # print("\nfitting second-level GLM for switch-based contrasts...")
    # second_level_results = fit_second_level_glm(
    #     subjects=all_subjs,
    #     smoothing_fwhm=8.0,
    #     switches=True,
    #     verbose=False
    # )
    print("\nprocessing complete!")

