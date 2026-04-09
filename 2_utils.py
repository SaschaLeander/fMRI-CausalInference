"""02 Analysis: Utility Functions.

This module collects the small shared helpers used across the analysis
pipeline.

Included functions:
- create_overview_df: map session folders to subject names and scan counts
- plot_temporal_distribution: visualize BR, Replay, and Physical Replay timing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import re

data_dir = Path('/gpfs01/bartels/group/br_insideout/data/sourcedata')

def create_overview_df(sessions):

    session_subject_mapping = []

    for session_folder in sessions:
        session_path = data_dir / session_folder
        
        # find mat files in this session
        mat_files_in_session = [f for f in os.listdir(session_path) 
                                if f.endswith('.mat') and not f.startswith('._')]
        
        if mat_files_in_session:
            # extract subject name from first mat file (format: subjectname_number_...)
            first_mat = mat_files_in_session[0]
            # subject name is everything before the first digit
            match = re.match(r'([a-zA-Z]+)\d*_', first_mat)
            subject_name = match.group(1) if match else first_mat.split('_')[0]
            # remove any trailing numbers from subject name
            subject_name = re.sub(r'\d+$', '', subject_name)
            
            session_subject_mapping.append({
                'session_folder': session_folder,
                'subject': subject_name,
                'n_mat_files': len(mat_files_in_session),
                'n_nii_files': len([f for f in os.listdir(session_path) 
                                if f.endswith('.nii')])
            })

    overview_df = pd.DataFrame(session_subject_mapping)
    return overview_df

def plot_temporal_distribution(subject_name=None, session_folder=None, overview_df=None, run_idx=0, figsize=(15, 6)):

    """
    Plot temporal distribution of BR, Replay, and Physical Replay conditions.
    
    Parameters:
    -----------
    subject_name : str, optional
        Subject name (e.g., 'agnes', 'sonja'). If provided, finds the corresponding session.
    session_folder : str, optional
        Session folder name (e.g., 's01', 's02'). Takes precedence over subject_name.
    run_idx : int, default=0
        Index of the run to plot (0 for first run, 1 for second, etc.)
    figsize : tuple, default=(15, 6)
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import scipy.io
    
    # Determine session folder
    if session_folder is None:
        if subject_name is None:
            raise ValueError("Either subject_name or session_folder must be provided")
        
        # Find session folder for this subject
        matching_sessions = [row['session_folder'] for idx, row in overview_df.iterrows() 
                           if row['subject'] == subject_name]
        if not matching_sessions:
            raise ValueError(f"No session found for subject '{subject_name}'")
        session_folder = matching_sessions[0]
        print(f"Using session folder: {session_folder} for subject: {subject_name}")
    
    # Load data
    session_path = data_dir / session_folder
    mat_files = sorted([f for f in os.listdir(session_path) 
                       if f.endswith('.mat') and not f.startswith('._')])
    
    if run_idx >= len(mat_files):
        raise ValueError(f"Run index {run_idx} out of range. Session has {len(mat_files)} runs.")
    
    mat_file_path = session_path / mat_files[run_idx]
    data = scipy.io.loadmat(mat_file_path)
    exp_log = data['exp_log'][0, 0]
    
    # extract timing data
    scan_times = exp_log['scan_times'].flatten() / 1000.0
    
    # BR data
    onsets_A_br = exp_log['onsets_A'].flatten()
    onsets_B_br = exp_log['onsets_B'].flatten()
    durs_A_br = exp_log['durs_A'].flatten() if 'durs_A' in exp_log.dtype.names else np.array([])
    durs_B_br = exp_log['durs_B'].flatten() if 'durs_B' in exp_log.dtype.names else np.array([])
    
    # replay data
    onsets_A_rep = exp_log['onsets_repA'].flatten()
    # print('onsets_A_rep:', onsets_A_rep)
    onsets_B_rep = exp_log['onsets_repB'].flatten()
    # print('onsets_B_rep:', onsets_B_rep)
    durs_A_rep = exp_log['durs_repA'].flatten() if 'durs_repA' in exp_log.dtype.names else np.array([])
    durs_B_rep = exp_log['durs_repB'].flatten() if 'durs_repB' in exp_log.dtype.names else np.array([])
    
    # physical replay data
    onsets_A_phys = exp_log['onsets_phys_replay_A'].flatten() / 1000.0
    # print('onsets_A_phys:', onsets_A_phys)
    # print('onsets_A_phys from exp_log:', exp_log['onsets_phys_replay_A'].flatten())
    onsets_B_phys = exp_log['onsets_phys_replay_B'].flatten() / 1000.0
    # print('onsets_B_phys:', onsets_B_phys)
    # print('onsets_B_phys from exp_log:', exp_log['onsets_phys_replay_B'].flatten())
    durs_A_phys = exp_log['durs_phys_replay_A'].flatten() if 'durs_phys_replay_A' in exp_log.dtype.names else np.array([])
    durs_B_phys = exp_log['durs_phys_replay_B'].flatten() if 'durs_phys_replay_B' in exp_log.dtype.names else np.array([])
    
    # create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_br = 1
    y_rep = 2
    y_phys = 3
    bar_height = 0.15
    
    # plot BR events - onsets as vertical lines, durations as horizontal bars
    for i, onset in enumerate(onsets_A_br):
        ax.plot([onset, onset], [y_br-0.3, y_br+0.3], 'b-', linewidth=2, alpha=0.7)
        if i < len(durs_A_br) and durs_A_br[i] > 0:
            ax.barh(y_br + bar_height/2, durs_A_br[i], height=bar_height, 
                    left=onset, color='blue', alpha=0.3, edgecolor='blue', linewidth=0.5)
    
    for i, onset in enumerate(onsets_B_br):
        ax.plot([onset, onset], [y_br-0.3, y_br+0.3], 'r-', linewidth=2, alpha=0.7)
        if i < len(durs_B_br) and durs_B_br[i] > 0:
            ax.barh(y_br - bar_height/2, durs_B_br[i], height=bar_height, 
                    left=onset, color='red', alpha=0.3, edgecolor='red', linewidth=0.5)
    
    # plot Replay events
    for i, onset in enumerate(onsets_A_rep):
        ax.plot([onset, onset], [y_rep-0.3, y_rep+0.3], 'b--', linewidth=2, alpha=0.7)
        if i < len(durs_A_rep) and durs_A_rep[i] > 0:
            ax.barh(y_rep + bar_height/2, durs_A_rep[i], height=bar_height, 
                    left=onset, color='blue', alpha=0.3, edgecolor='blue', linewidth=0.5)
    
    for i, onset in enumerate(onsets_B_rep):
        ax.plot([onset, onset], [y_rep-0.3, y_rep+0.3], 'r--', linewidth=2, alpha=0.7)
        if i < len(durs_B_rep) and durs_B_rep[i] > 0:
            ax.barh(y_rep - bar_height/2, durs_B_rep[i], height=bar_height, 
                    left=onset, color='red', alpha=0.3, edgecolor='red', linewidth=0.5)
    
    # plot Physical Replay events
    for i, onset in enumerate(onsets_A_phys):
        ax.plot([onset, onset], [y_phys-0.3, y_phys+0.3], 'b:', linewidth=2, alpha=0.7)
        if i < len(durs_A_phys) and durs_A_phys[i] > 0:
            ax.barh(y_phys + bar_height/2, durs_A_phys[i], height=bar_height, 
                    left=onset, color='blue', alpha=0.3, edgecolor='blue', linewidth=0.5)
    
    for i, onset in enumerate(onsets_B_phys):
        ax.plot([onset, onset], [y_phys-0.3, y_phys+0.3], 'r:', linewidth=2, alpha=0.7)
        if i < len(durs_B_phys) and durs_B_phys[i] > 0:
            ax.barh(y_phys - bar_height/2, durs_B_phys[i], height=bar_height, 
                    left=onset, color='red', alpha=0.3, edgecolor='red', linewidth=0.5)
    
    # plot scan times as tick marks at the bottom
    for scan_time in scan_times:
        ax.axvline(scan_time, color='green', alpha=0.15, linewidth=0.5, linestyle='-')
    
    # formatting
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['BR', 'Replay', 'Physical Replay'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'{session_folder} - Run {run_idx+1}: Temporal Distribution\nVertical lines = onsets, Horizontal bars = durations\n{mat_files[run_idx]}')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Image A'),
        Line2D([0], [0], color='red', lw=2, label='Image B'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='BR'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Replay'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Physical Replay'),
        Line2D([0], [0], color='green', lw=0.5, alpha=0.5, label='Scan Times'),
        Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, edgecolor='gray', label='Duration')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # load sessions s01-s31 (except s07, s13)
    sessions = sorted([d for d in os.listdir(data_dir) if d.startswith('s')])
    overview_df = create_overview_df(sessions)
    # plot temporal distribution for s01 run 0
    fig, ax = plot_temporal_distribution(session_folder='s01', run_idx=4, overview_df=overview_df)