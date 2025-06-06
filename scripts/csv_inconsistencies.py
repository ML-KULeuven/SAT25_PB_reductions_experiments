import pandas as pd
import os

# Configuration
TIMEOUT = 3600  # Timeout in seconds
CSV_DIR = "/home/orestis_ubuntu/work/CP25_PBCuts/results/"  # Directory containing CSV files

def analyze_csv(csv_file):
    """
    Analyze a single CSV file and print detailed statistics.
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"\nAnalyzing {csv_file}:")
        
        # Total number of rows
        total_rows = len(df)
        print(f"Total rows: {total_rows}")
        
        # Number of solved instances (exclude 'NO_OUTPUT')
        solved = df[df['result'].isin(['UNSATISFIABLE', 'OPTIMUM FOUND'])]
        print(f"Solved instances: {len(solved)}")
        
        # Number of timeouts
        timeouts = df[df['result'] == 'TIMEOUT']
        print(f"Timeouts: {len(timeouts)}")
        
        # Number of errors
        errors = df[df['result'] == 'ERROR']
        if len(errors) > 0:
            print(f"Errors: {len(errors)}")
        
        # Number of 'NO_OUTPUT' cases
        no_output = df[df['result'] == 'NO_OUTPUT']
        if len(no_output) > 0:
            print(f"NO_OUTPUT cases: {len(no_output)}")

        # Number of 'MEMORY_ERROR'
        mem_error = df[df['result'] == 'MEMORY_ERROR']
        if len(mem_error) > 0:
            print(f"MEMORY_ERROR cases: {len(mem_error)}")

        unknowns = df[df['result'] == 'UNKNOWN']
        if len(unknowns) > 0:
            print(f"UNKNOWN cases: {len(unknowns)}")
            print(unknowns['filename'])

        
        # Instances where elapsed time > TIMEOUT but result isn't 'TIMEOUT'
            inconsistencies = df[(df['elapsed time'] > TIMEOUT) & (~df['result'].isin(['SATISFIABLE', 'TIMEOUT', 'UNKNOWN']))]
            if not inconsistencies.empty:
                print(f"WARNING: Found {len(inconsistencies)} inconsistencies:")
                print(inconsistencies)
        else:
            print("No inconsistencies found.")
        
        print("-" * 50)
        return set(unknowns['filename'])
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return set()

def main():
    # Get all CSV files in the directory and its subdirectories
    csv_files = []
    for root, _, files in os.walk(CSV_DIR):
        if 'roundingsat' not in root or 'base' not in root:
            continue
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files in {CSV_DIR} and its subdirectories")

    all_unknowns = set()
    # Analyze each CSV file
    for csv_file in csv_files:
        unknowns = analyze_csv(csv_file)
        all_unknowns.update(unknowns)

    return all_unknowns

if __name__ == "__main__":
    print(main())
