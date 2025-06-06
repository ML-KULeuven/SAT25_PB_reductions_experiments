import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration (updated paths to averaged files)
benchmark_paths = {
    'DEC-LIN': '../results/DEC-LIN/',
    'OPT-LIN': '../results/OPT-LIN/',
    'KNAP': '../results/KNAP/'
}
TIMEOUT = 3600 # 3600 for comp benchmarks
AVG_SUBDIR = "averaged"  # New subdirectory for averaged files

ACTIVE_GROUPS = [  # List of groups to include in plotting
    'division-based',
    'saturation-based',
    'SOTA'
]

solver_groups = {
    'division-based': {
        'title': '',
        'output_file': 'cactus_avg.png',
        'solvers': [
            {'name': 'Exact', 'file': '{benchmark_path}exact/base/{subdir}/averaged.csv', 'color': 'blue'},
            {'name': 'Exact+AW', 'file': '{benchmark_path}exact/base+AW/{subdir}/averaged.csv', 'color': 'green'},
            {'name': 'Exact+WS', 'file': '{benchmark_path}exact/base+WS/{subdir}/averaged.csv', 'color': 'orange'},
        ]
    },
    'saturation-based': {
        'title': '',
        'output_file': 'cactus_avg.png',
        'solvers': [
            {'name': 'Exact', 'file': '{benchmark_path}exact/base/{subdir}/averaged.csv', 'color': 'blue'},
            {'name': 'Exact+MWD', 'file': '{benchmark_path}exact/base+MWD/{subdir}/averaged.csv', 'color': 'red'},
            {'name': 'Exact+MWD+MWI', 'file': '{benchmark_path}exact/base+MWD+MWI/{subdir}/averaged.csv', 'color': 'olive'},
        ]
    },
    'SOTA': {
        'title': '',
        'output_file': 'cactus_sota.png',
        'solvers': [
            {'name': 'Exact', 'file': '{benchmark_path}exact/base/seed1.csv', 'color': 'blue'},
            {'name': 'RoundingSat', 'file': '{benchmark_path}roundingsat/base/seed1.csv', 'color': 'red'},
            {'name': 'Exact+AW+WS+MWD+MWI', 'file': '{benchmark_path}exact/base+AW+WS+MWD+MWI/seed1.csv', 'color': 'purple'},
            {'name': 'Exact+AW+WS+MWD', 'file': '{benchmark_path}exact/base+AW+WS+MWD/seed1.csv', 'color': 'cyan'},
        ]
    },

    # ... other groups updated similarly ...
}

def create_cactus_plot(group_name, group_config, benchmark):
    """Create cactus plot with confidence bands from raw seed data"""
    benchmark_path = benchmark_paths[benchmark]
    plt.figure(figsize=(12, 8))

    # Store all solver data with confidence bands
    solver_stats = {}
    # vbs_times = []

    # Process each solver configuration
    for solver in group_config['solvers']:
        solver_name = solver['name']
        solver_color = solver['color']
        solver_dir = Path(solver['file'].format(
            benchmark_path=benchmark_path,
            subdir=''  # Go directly to solver directory
        )).parent

        # Collect all seed files for this solver
        seed_files = list(solver_dir.glob("seed*.csv"))
        if not seed_files:
            print(f"⚠️ No seed files found for {solver_name} in {solver_dir}")
            continue

        # Process each seed file
        seed_times = []
        for seed_file in seed_files:
            df = pd.read_csv(seed_file)
            # Filter and process times
            solved_mask = df['result'].isin(['OPTIMUM FOUND', 'UNSATISFIABLE'])
            times = np.where(solved_mask, df['elapsed time'], TIMEOUT)
            sorted_times = np.sort(times)
            seed_times.append(sorted_times)

        # Calculate statistics across seeds
        max_length = max(len(t) for t in seed_times)
        padded = [np.pad(t, (0, max_length - len(t)), constant_values=TIMEOUT)
                for t in seed_times]
        stacked = np.vstack(padded)

        low_percentile = 2.5
        high_percentile = 100-low_percentile

        solver_stats[solver_name] = {
            'median': np.median(stacked, axis=0),
            'q25': np.percentile(stacked, low_percentile, axis=0),
            'q75': np.percentile(stacked, high_percentile, axis=0),
            'color': solver_color,
        }

    # Calculate x-axis limits
    min_at_10s = None
    max_solved = 0

    for solver_name, stats in solver_stats.items():
        # Count instances solved within 10 seconds
        fast_solved = np.sum(stats['median'] <= 50)
        # Count total solved instances (under timeout)
        total_solved = np.sum(stats['median'] < TIMEOUT)

        # Update limits
        min_at_10s = fast_solved if min_at_10s is None else min(min_at_10s, fast_solved)
        max_solved = max(max_solved, total_solved)

    # Configure plot labels and style
    plt.xlabel('Number of instances solved', fontsize=30)
    plt.ylabel('Solve time (seconds)', fontsize=30)
    # plt.title(group_config['title'].format(benchmark=benchmark, low=low_percentile, high=high_percentile))

    # Set grid and limits
    # plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(left=max(1, min_at_10s -5), right=max_solved +5)
    plt.ylim(bottom=1, top=TIMEOUT)
    plt.tick_params(labelsize=26)

    # Plot all solvers first
    for solver_name, stats in solver_stats.items():
        x = np.arange(1, len(stats['median']) + 1)
        linestyle = '--' if solver_name == 'VBS' else '-'

        # Plot median line WITH LABEL
        plt.plot(x, stats['median'],
                label=solver_name,
                linestyle=linestyle,
                 color=stats['color'],)

        # Add confidence band between quartiles
        if solver_name != 'VBS':
            plt.fill_between(x, stats['q25'], stats['q75'], alpha=0.2, color=stats['color'])

    # Get solver performance for sorting
    solver_performance = []
    for solver_name, stats in solver_stats.items():
        if solver_name == 'VBS':
            continue  # Handle VBS separately
        solved = np.sum(stats['median'] < TIMEOUT)
        low_solved = np.sum(stats['q25'] < TIMEOUT)
        high_solved = np.sum(stats['q75'] < TIMEOUT)
        solver_performance.append((solved, low_solved, high_solved, solver_name))

    print(solver_performance)
    
    # Sort solvers by solved count (descending)
    sorted_solvers = sorted(solver_performance, key=lambda x: -x[0])
    
    # Get handles and labels in sorted order
    handles, labels = plt.gca().get_legend_handles_labels()
    handle_map = {label: handle for handle, label in zip(handles, labels)}
    
    # Rebuild handles/labels list with sorting
    sorted_handles = []
    sorted_labels = []
    for solved, _, _, solver_name in sorted_solvers:
        if solver_name in handle_map:
            sorted_handles.append(handle_map[solver_name])
            sorted_labels.append(solver_name)
    
    # Create sorted legend
    if sorted_handles:
        plt.legend(
            sorted_handles, sorted_labels,
            loc='upper left',
            title=f"Solvers",
            title_fontsize=26,
            frameon=True,
            framealpha=0.9,
            prop={'size': 20}
            # edgecolor='black'
        )
    else:
        print("No plottable data found - check:")
        print(f"- Solver paths: {[s['file'] for s in group_config['solvers']]}")
        print(f"- Seed files: {[list(Path(s['file']).parent.glob('seed*')) for s in group_config['solvers']]}")

    # Adjust layout to prevent legend cutoff
    # plt.tight_layout()  # Comment out or remove this line

    # Update output directory to include benchmark
    plot_dir = f'/home/orestis_ubuntu/work/CP25_PBCuts/plots/{group_name}/{benchmark}'
    os.makedirs(plot_dir, exist_ok=True)

    # Save plot with appropriate name
    output_filename = f"{plot_dir}/cactus_with_ci_{benchmark}.png"
    plt.savefig(output_filename, bbox_inches='tight')  # Critical for external legend
    print(f"Saved plot with legend to {output_filename}")

    return {solver['name']: len(stats['median']) for solver in group_config['solvers']}

def main():
    # Process both benchmarks
    benchmarks = ['DEC-LIN', 'OPT-LIN']
    # benchmarks = ['KNAP'] # comment this line to process competition benchmarks
    
    # Example solver configurations (modify as needed)
    selected_groups = ['division-based', 'saturation-based', 'SOTA']

    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark} benchmark")
        base_dir = Path(benchmark_paths[benchmark]) / "exact"
        
        # Create output filename with benchmark name
        output_template = "cactus_{group}_{benchmark}.png"
        
        # Process each solver group
        for group_name in selected_groups:
            group_config = solver_groups[group_name].copy()
            
            # Update output path with benchmark name
            group_config['output_file'] = output_template.format(
                group=group_name,
                benchmark=benchmark.lower()
            )
            
            # Create plot for this benchmark
            create_cactus_plot(
                group_name=group_name,
                group_config=group_config,
                benchmark=benchmark
            )

if __name__ == "__main__":
    main()