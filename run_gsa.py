import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli

from SALib.analyze import sobol
from autogluon.tabular import TabularPredictor
import seaborn as sns 

from src.problem_definitions import problem_minimum
from src.sobol import remove_groups_from_problem, run_sobol_analysis, save_results_to_csv_sobol, plot_sobol_heatmap, plot_sobol_indices
# Configurat
# Configuration
col_setting = int(os.getenv('COL_SETTING'))
folder = os.getenv('MODEL_FOLDER')
if folder is None:
    folder = 'nebula'
dataset_name = 'NEBULA_englandwales_domestic_filtered'

region = 'None'
label = os.getenv('LABEL')
if label is None:
    raise ValueError('no target')
time_lim = 15000 

# Define regions to analyze
REGIONS = ['SE', 'LN', 'SW', 'EE', 'EM', 'YH', 'WM', 'NE', 'NW', 'WA']

# Base model and output paths
BASE_MODEL_PATH = '/home/gb669/rds/hpc-work/energy_map/data/automl_models'
BASE_OUTPUT_PATH = '/home/gb669/rds/hpc-work/energy_map/UKPostcodePrediction/sobol'

# Number of samples for Sobol analysis
N = int(os.getenv('N', 1024))


def get_model_path(folder, dataset_name, label, time_lim, col_setting, region):
    return f'{BASE_MODEL_PATH}/{folder}/{dataset_name}__global__{label}__{time_lim}__colset_{col_setting}__best_quality___tsp_1.0__all__{region}'

def get_output_path(label, region_id, folder, col_setting, grouped):
    base = f'{BASE_OUTPUT_PATH}/{label}/{region_id}/{folder}/colset_{col_setting}'
    return os.path.join(base, 'grouped' if grouped else 'ungrouped', f'N{N}')


       
def process_region(region_id, problem, grouped, folder, col_setting, label):
    print(f"\nProcessing region: {region_id}")
    start_time = time.time()
    
    # Create output directory
    result_folder = get_output_path(label, region_id, folder, col_setting, grouped)
    os.makedirs(result_folder, exist_ok=True)
    
    # Get model path and load predictor
    model_path = get_model_path(folder, dataset_name, label, time_lim, col_setting, region)
    print(f'Loading predictor for region {region_id}')
    predictor = TabularPredictor.load(model_path, require_version_match=True)
    
    # Run analysis
    print(f'Starting Sobol analysis for region {region_id} with N={N}')
    sobol_results = run_sobol_analysis(N, predictor, region_id, problem)
    
    # Save results
    print(f'Saving results for region {region_id}')
    s1_data, st_data = save_results_to_csv_sobol(sobol_results, result_folder, problem)
    
    # Create plots
    group_map = None
    plot_sobol_heatmap(sobol_results, result_folder, problem, group_map)
    plot_sobol_indices(s1_data, st_data, result_folder, problem, group_map)
    
    # Save problem configuration
    with open(os.path.join(result_folder, 'problem_config.json'), 'w') as f:
        json.dump(problem, f, indent=4)
    
    # Save execution time
    execution_time = time.time() - start_time
    with open(os.path.join(result_folder, 'execution_time.txt'), 'w') as f:
        f.write(f"Execution time: {execution_time:.2f} seconds")
    
    print(f"Completed analysis for region {region_id} in {execution_time:.2f} seconds")
    return execution_time



if __name__ == "__main__":
    # Initialize problem definition
    grouped = False
    problem = problem_minimum
    if not grouped:
        problem = remove_groups_from_problem(problem)
    
    print('Problem configuration:')
    print('num vars:', problem['num_vars'])
    print('var names:', problem['names'])
    print('bounds:', problem['bounds'])
    print('keys:', problem.keys())
    
    # Process all regions
    total_start_time = time.time()
    execution_times = {}
    
    for region_id in REGIONS:
        try:
            execution_times[region_id] = process_region(
                region_id, problem, grouped, folder, col_setting, label
            )
        except Exception as e:
            print(f"Error processing region {region_id}: {e}")
            execution_times[region_id] = -1  # Mark failed regions
    
    # Save summary of execution times
    total_time = time.time() - total_start_time
    summary_path = os.path.join(BASE_OUTPUT_PATH, label, 'execution_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        f.write("Region-wise execution times:\n")
        for region_id, exec_time in execution_times.items():
            status = f"{exec_time:.2f} seconds" if exec_time > 0 else "FAILED"
            f.write(f"{region_id}: {status}\n")
    
    print(f"\nAnalysis completed for all regions in {total_time:.2f} seconds")
    print(f"Execution summary saved to {summary_path}") 
        
