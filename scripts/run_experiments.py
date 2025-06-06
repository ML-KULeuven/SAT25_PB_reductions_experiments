import lzma
import tempfile
import os
import subprocess
import glob
import re
import csv
import resource
import concurrent.futures
import time
import psutil
import resource
import platform
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global configuration
DEFAULT_TIMEOUT = 3600  # 1 hour timeout
PHYSICAL_CORES = psutil.cpu_count(logical=False)

# UNKNOWNS = {'normalized-NG.dot_fibo_0null50mast_rapportOE2.0_TMAX180_K192_cod2.opb.xz', 'normalized-comppebphp_opt_pyrheightequalsnholes_50.opb.xz', 'normalized-hw128-vm98p-dec.opb.negationfix.opb.xz', 'normalized-neos-631710.opb.xz', 'normalized-hw256-vm75p-dec.opb.negationfix.opb.xz', 'normalized-NG.dot_fibo_10null50mast_rapportOE2.0_TMAX15_K192_cod1.opb.xz', 'normalized-hw512-vm50p-dec.opb.negationfix.opb.xz', 'normalized-172.opb.xz', 'normalized-hw256-vm95p-dec.opb.negationfix.opb.xz', 'normalized-NG.dot_unif_50null10mast_rapportOE2.0_TMAX180_K192_cod2.opb.xz', 'normalized-NG.dot_unif_0null10mast_rapportOE1.0_TMAX15_K192_cod1.opb.xz', 'normalized-hw256-vm90p-dec.opb.negationfix.opb.xz', 'normalized-minisat100_16_6_3_mh.opb.xz', 'normalized-minisat100_16_6_5_mh.opb.xz', 'normalized-neos-948346.0.s.opb.xz', 'normalized-hw256-vm100p-dec.opb.negationfix.opb.xz', 'normalized-hw256-vm98p-dec.opb.negationfix.opb.xz', 'normalized-NG.dot_unif_50null10mast_rapportOE2.0_TMAX180_K192_cod1.opb.xz', 'normalized-hw512-vm90p-dec.opb.negationfix.opb.xz', 'normalized-NG.dot_fibo_20null20mast_rapportOE2.0_TMAX60_K192_cod2.opb.xz', 'normalized-minisat100_16_6_8_mh.opb.xz', 'normalized-NG.dot_luby_20null50mast_rapportOE2.0_TMAX60_K192_cod2.opb.xz', 'normalized-hw512-vm25p-dec.opb.negationfix.opb.xz', 'normalized-comppebphp_opt_pyrheightequalsnholes_54.opb.xz', 'normalized-hw512-vm75p-dec.opb.negationfix.opb.xz', 'normalized-NG.dot_fibo_10null0mast_rapportOE1.0_TMAX180_K192_cod2.opb.xz', 'normalized-NG.dot_fibo_20null10mast_rapportOE1.0_TMAX180_K192_cod1.opb.xz', 'normalized-hw256-vm50p-dec.opb.negationfix.opb.xz', 'normalized-hw512-vm99p-dec.opb.negationfix.opb.xz', 'normalized-minisat100_16_6_2_mh.opb.xz', 'normalized-hw128-vm99p-opt.opb.negationfix.opb.xz', 'normalized-hw128-vm98p-opt.opb.negationfix.opb.xz', 'normalized-aries-da_network_2000_5__647_1792__512.opb.xz'}

class SolverConfig:
    def __init__(self, name, path, command, flag_configs=None, parser=None, timeout=3600):
        """
        Initialize solver configuration
        name: identifier for the solver
        path: path to the solver executable
        command: command template to run the solver
        flag_configs: list of different flag configurations for this solver
        parser: custom parser function for this solver's output
        """
        self.name = name
        self.path = path
        self.command = command
        self.flag_configs = flag_configs or [{}]
        self.parser = parser or default_parser
        self.timeout = timeout


class MemoryAwareExecutor:
    def __init__(self, timeout):
        self.timeout = timeout
        self.phases = [
            {'limit_gb': 31, 'retries': 0}
        ]
        self.max_cores = 32  # Your physical cores
        self.total_mem = psutil.virtual_memory().total

    def calculate_parallelism(self, limit_gb):
        available_mem = self.total_mem * 0.9  # 10% safety margin
        mem_parallel = int(available_mem // (limit_gb * 1024 ** 3))
        return min(self.max_cores, mem_parallel)

    @staticmethod
    def set_memory_limit(limit_gb):
        try:
            soft = hard = limit_gb * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except Exception as e:
            print(f"Memory limit error: {e}")

    def run_job(self, job, limit_gb):  # limit_gb comes from current phase
        """Modified version of your original run_solver function"""
        solver_name, instance_path, timeout, flags, output_file = job
        solver = SOLVERS[solver_name]

        def set_subprocess_limits():
            """🛠️ Set limits for the solver subprocess"""
            MemoryAwareExecutor.set_memory_limit(limit_gb)
            set_core_affinity()

        try:
            # Create temporary file for decompressed instance
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name

                # Decompress the .xz file and write to the temporary file
                with lzma.open(instance_path, 'rb') as xz_file:
                    while chunk := xz_file.read(1024):
                        temp_file.write(chunk)

            # Format flags with current memory limit (convert GB to MB)
            formatted_flags = ' '.join([
                f.format(mem_limit=limit_gb * 1024)  # Convert GB to MB
                for f in flags.values()
            ])

            command = solver.command.format(
                path=solver.path,
                instance=temp_file_path,
                flags=formatted_flags,
                timeout=self.timeout
            )

            print(f"Running {command=}")

            # Run solver and measure time
            start_time = time.time()

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=set_subprocess_limits
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Parse and write results
            csvline = solver.parser(result.stdout, instance_path, elapsed_time)
            write_to_csv(output_file, csvline)
            return {'status': 'success'}

        except MemoryError:
            return {'status': 'memory_fail', 'limit_gb': limit_gb}
        except subprocess.TimeoutExpired:
            write_to_csv(output_file, [os.path.basename(instance_path), 2 * self.timeout, "TIMEOUT"])
            return {'status': 'timeout'}
        except Exception as e:
            print(f"Error running {solver_name}: {e}")
            write_to_csv(output_file, [os.path.basename(instance_path), 2 * self.timeout, "ERROR"])
            return {'status': 'error'}
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    def execute_phase(self, jobs, phase_idx):
        phase = self.phases[phase_idx]
        limit_gb = phase['limit_gb']

        # 🛠️ Pass limit_gb to each job
        phase_jobs = [(job, limit_gb) for job in jobs]

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(self.run_job, job, limit_gb): job
                for job, limit_gb in phase_jobs
            }

            retry_jobs = []
            for future in as_completed(futures):
                result = future.result()
                job = futures[future]

                if result['status'] == 'memory_fail':
                    if phase_idx < len(self.phases) - 1:
                        retry_jobs.append(job)


            return retry_jobs

    def _wrap_run_job(self, job_data, limit_gb):
        """Wrapper to set memory limits before execution"""
        self.set_memory_limit(limit_gb)
        return self.run_job(job_data)

    def process_group(self, group_name, tasks):
        """Replacement for your original process_group function"""
        remaining_jobs = tasks.copy()

        for phase_idx in range(len(self.phases)):
            if not remaining_jobs:
                break

            remaining_jobs = self.execute_phase(remaining_jobs, phase_idx)

            print(f"Phase {phase_idx + 1} completed. "
                  f"Remaining jobs: {len(remaining_jobs)}")

        if remaining_jobs:
            print(f"{len(remaining_jobs)} jobs failed all memory tiers")

def default_parser(output_text, instance_path, elapsed):
    # NOTE: most solvers give a different result between OPTIMUM FOUND, SATISFIABLE and UNSATISFIABLE, this can be fixed afterwards and is probable the easiest way to go about things instead of a different parser for each solver
    """
    Default parser for solver output.
    Looks for lines starting with "s " to extract the result.
    """
    # find line starting with "s "
    match = re.search(r"^s\s+(.*)", output_text, re.MULTILINE)
    if match:
        # Extract the line starting with 's' and construct the CSV line
        csv_line = [os.path.basename(instance_path), elapsed, match.group(1)]
        return csv_line
    return [f'"{os.path.basename(instance_path)}"', elapsed, "NO_OUTPUT"]

def parse_exact(output_text, instance_path, elapsed):
    """
    Custom parser for Exact solver output.
    Extracts detailed statistics from csvline output.
    """
    csv_lines = re.findall(r"c csvline,(.*)", output_text)
    match = re.search(r"^s\s+(.*)", output_text, re.MULTILINE)
    if csv_lines and match:
        # Use the final csv line and add filename and elapsed time
        csv_line = [os.path.basename(instance_path), elapsed, match.group(1)] + csv_lines[-1].split(",")
        return csv_line
    return [os.path.basename(instance_path), elapsed, "NO_OUTPUT"] + [0] * (123 - 3)

def parse_SCIP(output_text, instance_path, elapsed):
    """
    Custom parser for SCIP output.

    Extracts the SCIP status from the output. It searches for a line starting with 
    "SCIP Status" and extracts the status message. If the status contains content 
    within square brackets (e.g., "[infeasible]"), that is used as the result.

    New behavior:
      - If the extracted result is "infeasible" or "unsat", it overrides the 
        elapsed time with 2 * timeout (i.e., the UNSAT time), matching how timeouts
        are reported elsewhere.

    Args:
        output_text (str): The output from the SCIP solver.
        instance_path (str): Path to the benchmark instance.
        elapsed (float): The measured elapsed CPU time.

    
    Returns:
        list: A CSV line formatted as [basename(instance_path), elapsed time, result]
    """
    # Try to find a line with "SCIP Status:" which should contain the result message.
    match = re.search(r"SCIP Status:\s*(.*)", output_text, re.IGNORECASE)
    if match:
        result_text = match.group(1).strip()
        # Try to extract a result within square brackets (e.g., "[infeasible]")
        bracket_match = re.search(r"\[([^\]]+)\]", result_text)
        if bracket_match:
            result = bracket_match.group(1).strip()
        else:
            result = result_text


    if result.lower() == "infeasible":
        return [os.path.basename(instance_path), elapsed, "UNSATISFIABLE"]
    if result.lower() == "optimal solution found": 
        return [os.path.basename(instance_path), elapsed, "OPTIMUM FOUND"]

    # If nothing was parsed, default to "UNKNOWN"
    return [os.path.basename(instance_path), elapsed, "UNKNOWN"]

def parse_gurobi(output_text, instance_path, elapsed):
    """
    Custom parser for Gurobi output.

    Extracts the result from Gurobi's output by searching for lines indicating 
    the solution status (e.g., "Model is infeasible" or "Optimal solution found").
    
    If the result is "infeasible", the elapsed time is overridden to 2 * timeout
    (i.e., the UNSAT time), matching how timeouts are reported elsewhere.

    Args:
        output_text (str): The output from the Gurobi solver.
        instance_path (str): Path to the benchmark instance.
        elapsed (float): The measured elapsed CPU time.
    
    Returns:
        list: A CSV line formatted as [basename(instance_path), elapsed time, result]
    """

    # Extract the result from Gurobi's output
    if "Model is infeasible" in output_text:
        result = "UNSATISFIABLE"  # Use UNSAT time for infeasible cases
    elif "Optimal solution found" in output_text:
        result = "OPTIMUM FOUND"
    else:
        result = "UNKNOWN"

    return [os.path.basename(instance_path), elapsed, result]

# Define solver configurations
SOLVERS = {
    'exact': SolverConfig(
        name='exact',
        path='../../solvers/PB/exact/build/Exact', # YOUR PATH TO EXACT SOLVER HERE
        command='{path} {instance} {flags}',
        flag_configs=[
            {   # Base configuration
                'name': 'base',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=0',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=0',
                    'mwi': '--ca-mwi=0',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=0'
                }
            },
            {   # Base+MWD configuration
                'name': 'base+MWD',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=0',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=1',
                    'mwi': '--ca-mwi=0',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=0'
                }
            },
            {   # Base configuration
                'name': 'base+MWD+MWI',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=0',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=1',
                    'mwi': '--ca-mwi=1',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=0'
                }
            },
            {   # Base configuration
                'name': 'base+AW',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=1',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=0',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=0'
                }
            },
            {   # Base configuration
                'name': 'base+WS',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=0',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=0',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=1'
                }
            },
            {   # Base configuration
                'name': 'base+AW+WS',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=1',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=0',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=1'
                }
            },
            {   # Base configuration
                'name': 'base+AW+WS+MWD+MWI',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=1',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=1',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=1',
                }
            },
            {  # Base configuration
                'name': 'base+AW+WS+MWD',
                'flags': {
                    'seed': '--seed={seed}',
                    'timeout': '--timeout={timeout}',
                    'print_csv': '--print-csv',
                    'anti-weaken': '--ca-anti-weaken=1',
                    'multiply': '--ca-multiply=0',
                    'multweaken': '--ca-multweaken=1',
                    'partial': '--ca-partial-weakening=1',
                    'superfluous': '--ca-weaken-superfluous=1',
                    'mwi': '--ca-mwi=0'
                }
            },
        ],
        parser=parse_exact,
    ),
    'roundingsat': SolverConfig(
        name='roundingsat',
        path='../../solvers/PB/roundingsat/build_new/roundingsat', # YOUR PATH TO ROUNDINGSAT SOLVER HERE
        command='{path} {flags} {instance}',
        flag_configs=[
            {   # Base configuration
                'name': 'base',
                'flags': {
                    'timeout': '--\'time limit\'={timeout}',
                    'print': '--print-sol=1',
                    'vsids': '--vsids-var=0.8',
                    'cg-encoding': '--cg-encoding=reified',
                    'cg-resprop': '--cg-resprop=1'
                }
            },
        ]
    )
}

def generate_output_file_name(solver_name, config_name, seed, subdir):
    """
    Generate output file name based on solver configuration and seed
    """
    folder_path = f"../results/{subdir}/{solver_name}/{config_name}"
    os.makedirs(folder_path, exist_ok=True)
    
    return f"{folder_path}_new/seed{seed}.csv"

# Experiment group definitions
EXPERIMENT_GROUPS = {
    'base': {
        # Run base configuration for all solvers
        'exact': ['base'],
        'roundingsat': ['base'],
    },
    'multweaken': {
        'exact': ['base+MWD', 'base+MWD+MWI']
    },
    'all_options': {
        'exact': ['base+AW+WS+MWD', 'base+AW+WS+MWD+MWI']
    },
    'WS+AW': {
        'exact': ['base+WS+AW', 'base+WS', 'base+AW']
    }
    # Add more groups as needed
}

# Select which groups to run
ACTIVE_GROUPS = ['base']

# Generate active experiments from selected groups
ACTIVE_EXPERIMENTS = {}
for group in ACTIVE_GROUPS:
    group_config = EXPERIMENT_GROUPS[group]
    for solver, configs in group_config.items():
        if solver not in ACTIVE_EXPERIMENTS:
            ACTIVE_EXPERIMENTS[solver] = {
                'enabled': True,
                'configs': set()
            }
        ACTIVE_EXPERIMENTS[solver]['configs'].update(configs)

print(ACTIVE_EXPERIMENTS)

# Convert sets to lists
for solver in ACTIVE_EXPERIMENTS:
    ACTIVE_EXPERIMENTS[solver]['configs'] = list(ACTIVE_EXPERIMENTS[solver]['configs'])

# Define headers for different solvers
EXACT_HEADER = header = ["filename", "elapsed time", "result",
    "cpu time", "parse time", "solve time", "solve time det", "optimization time", "top-down time", "top-down time det", 
    "bottom-up solve time", "bottom-up solve time det", "conflict analysis time", "learned minimize time", "propagation time", 
    "constraint cleanup time", "inprocessing time", "garbage collection time", "constraint learning time", 
    "time spent in activity heuristic", "at-most-one detection time", "at-most-one detection time det", 
    "time spent in lift degree optimization", "number of lifted degrees", "LP solve time", "LP total time", "LP total time det", 
    "cores", "solutions", "propagations", "decisions", "conflicts", "restarts", "inprocessing phases", "original variables", 
    "auxiliary variables", "input clauses", "input cardinalities", "input general constraints", "input length average", 
    "input degree average", "input strength average", "learned clauses", "learned cardinalities", "learned general constraints", 
    "learned length average", "learned degree average", "learned strength average", "learned LBD average", "unit literals derived", 
    "pure literals", "constraints satisfied at root", "constraints simplified during database reduction", "small coef constraints", 
    "large coef constraints", "arbitrary coef constraints", "probing calls", "probing inprocessing time", "unit lits due to probing", 
    "equalities due to probing", "implications added due to probing", "max implications in memory due to probing", 
    "detected at-most-ones", "units derived during at-most-one detection", "resolve steps", "self-subsumptions", 
    "gcd simplifications", "detected cardinalities", "weakened non-implied", "weakened non-implying", 
    "number of multiply-weakens on reason", "number of multiply-weakens on conflict", "number of direct multiply-weakens", 
    "number of indirect multiply-weakens", "clausal propagations", "cardinality propagations", "watched propagations", 
    "counting propagations", "watch lookups", "watch backjump lookups", "watch checks", "propagation checks", 
    "blocking literal success", "blocking literal fails", "literal additions", "saturation steps", "unknown literals rounded up", 
    "trail pops", "formula constraints", "dominance breaking constraints", "learned constraints", "bound constraints", 
    "core-guided constraints", "reduced constraints", "encountered formula constraints", "encountered dominance breaking constraints", 
    "encountered learned constraints", "encountered bound constraints", "encountered core-guided constraints", 
    "encountered reduced constraints", "encountered detected at-most-ones", "encountered detected equalities", 
    "encountered detected implications", "CG unit cores", "CG non-clausal cores", "best upper bound", "best lower bound", 
    "LP relaxation objective", "LP constraints added", "LP constraints removed", "LP pivots", "LP approximate operations", 
    "LP literal additions", "LP calls", "LP optimalities", "LP no pivot count", "LP infeasibilities", "LP Farkas constraints", 
    "LP dual constraints", "LP basis resets", "LP cycling count", "LP singular count", "LP no primal count", "LP no dual count", 
    "LP no farkas count", "LP other issue count", "LP Gomory cuts", "LP learned cuts", "LP deleted cuts", 
    "LP encountered Gomory constraints", "LP encountered Farkas constraints", "LP encountered dual constraints"
]

DEFAULT_HEADER = ["filename", "elapsed time", "result"]

def get_header_for_solver(solver_name):
    if solver_name == 'exact':
        return EXACT_HEADER
    return DEFAULT_HEADER


def set_core_affinity():
    """Bind process to specific physical cores"""
    try:
        proc = psutil.Process()
        core = proc.pid % PHYSICAL_CORES
        proc.cpu_affinity([core])
    except Exception as e:
        print(f"Core affinity error: {e}")

def set_memory_limit():
    """Set 31GB virtual memory limit (soft/hard)"""
    try:
        mem_limit = 31 * 1024**3  # 31GB in bytes
        resource.setrlimit(
            resource.RLIMIT_AS,
            (mem_limit, mem_limit)
        )
    except Exception as e:
        print(f"Memory limit error: {e}")

def main():
    # Print system resource info
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Total memory: {psutil.virtual_memory().total/1024**3:.1f}GB")
    print(f"Safe parallelism: {min(PHYSICAL_CORES, psutil.virtual_memory().total//(31*1024**3))}")
    benchmark_base_dir = "../../benchmarks/filtered-PB24/"
    subdirs = ["DEC-LIN", "OPT-LIN"]
    timeout = 3600  # Override default timeout for specific runs
    executor = MemoryAwareExecutor(timeout=timeout)
    SEEDS = range(1, 6)

    # Debug: Check if benchmark directory exists
    if not os.path.exists(benchmark_base_dir):
        print(f"Error: Benchmark directory does not exist: {benchmark_base_dir}")
        return

    # Debug: Check if subdirectories exist
    for subdir in subdirs:
        subdir_path = os.path.join(benchmark_base_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Error: Subdirectory does not exist: {subdir_path}")
            return

    # Debug: Check if benchmarks are found
    benchmarks = []
    for subdir in subdirs:
        benchmarks.extend(glob.glob(f"{benchmark_base_dir}{subdir}/*.xz"))
    if not benchmarks:
        print(f"Error: No .xz files found in {benchmark_base_dir}{subdirs}")
        return

    # Debug: Check active experiments
    print(f"Active groups: {ACTIVE_GROUPS}")
    for group in ACTIVE_GROUPS:
        print(f"Group {group}: {EXPERIMENT_GROUPS[group]}")

    # Generate tasks for all solver configurations and seeds
    tasks = []
    output_files = set()

    # Only process solvers that are in ACTIVE_EXPERIMENTS
    for solver_name, solver_info in ACTIVE_EXPERIMENTS.items():
        if not solver_info['enabled']:
            continue
            
        solver = SOLVERS[solver_name]
        active_configs = solver_info['configs']
        
        for config in solver.flag_configs:
            config_name = config.get('name', 'base')
            # Skip configs that aren't in our active set
            if config_name not in active_configs:
                continue
                
            base_flags = config.get('flags', {}).copy()
            
            for seed in SEEDS:
                flags = base_flags.copy()
                if 'timeout' in flags:
                    flags['timeout'] = flags['timeout'].format(timeout=timeout)
                if 'seed' in flags:
                    flags['seed'] = flags['seed'].format(seed=seed)
                
                for subdir in subdirs:
                    output_file = generate_output_file_name(
                        solver_name=solver_name,
                        config_name=config_name,
                        seed=seed,
                        subdir=subdir
                    )
                    output_files.add(output_file)
                    
                    benchmarks = glob.glob(f"{benchmark_base_dir}{subdir}/*.xz")
                    for benchmark in benchmarks:
                        basename = os.path.basename(benchmark)
                        tasks.append((solver_name, benchmark, timeout, flags, output_file))
                        # print(basename)

    # Debug: Print number of tasks
    # print(f"Number of tasks: {len(tasks)}")
    # print(f"Tasks: {tasks}")

    # Initialize output files with headers
    for output_file in output_files:
        solver_name = output_file.split('/')[-3]  # Extract solver name from path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(get_header_for_solver(solver_name))

    print(f"{tasks=}")

    # Group tasks by solver and config
    task_groups = {}
    for task in tasks:
        solver_name, _, _, _, output_file = task
        config_name = output_file.split('/')[-2]  # Extract config name from path
        key = (solver_name, config_name)
        if key not in task_groups:
            task_groups[key] = []
        task_groups[key].append(task)

    # Process groups using the memory-aware executor
    for (solver_name, config_name), group_tasks in task_groups.items():
        print(f"\nProcessing {solver_name} - {config_name}")
        executor.process_group(f"{solver_name}-{config_name}", group_tasks)

def write_to_csv(file_path, csv_line):
    """
    Write a new csv_line to the CSV file.
    """
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_line)

if __name__ == "__main__":
    print("Starting main")
    main()
