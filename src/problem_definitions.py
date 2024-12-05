

    # problem_definitions.py
from .column_settings import settings_dict
import numpy as np 


import numpy as np
import pandas as pd
from scipy.stats import qmc
from SALib.sample import saltelli



problem_minimum = {'num_vars': 11,
                   'names': [ 
'all_res_total_fl_area_H_total',
'Pre 1919_pct',
'Standard size detached_pct',
'postcode_area',
'HDD_winter',  
'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'socio_class_perc_L1, L2 and L3: Higher managerial, administrative and professional occupations',
 'household_comp_perc_One-person household',
 'Domestic outbuilding_pct',
 '3-4 storey and smaller flats_pct',
 ] ,
 'bounds': [
        [740.0, 6780.0], 
          [5.6, 100.0],          # Pre 1919_pct
          [3.6, 100] ,
          [1830, 35460], 
          [39.65, 52.34],
          [0.2, 0.4],
          [0.3, 1 ],
          [0, 0.2] ,
          [0.1, 0.5], 
          [3.2, 40.9],
          [3.2, 100]
    
 ]
      

}
problem_fi = {
        'num_vars': 16,
        'names': [
            'all_types_uprn_count_total',
            'clean_res_uprn_count_total',
            'all_res_total_fl_area_H_total',
            'Pre 1919_pct',
            'all_types_total_fl_area_H_total',
            'clean_res_premise_area_total',
            'all_types_total_fl_area_FC_total',
            '1919-1944_pct',
            'all_types_premise_area_total',
            'clean_res_total_buildings',
            'all_res_total_fl_area_FC_total',
            'clean_res_total_fl_area_H_total',
            'Standard size detached_pct',
            'clean_res_total_fl_area_FC_total',
            'postcode_density',
            'Post 1999_pct'
        ],
        'bounds': [
            [6.0, 50.0],           # all_types_uprn_count_total
            [6.0, 49.0],           # clean_res_uprn_count_total
            [740.0, 6780.0],       # all_res_total_fl_area_H_total
            [5.6, 100.0],          # Pre 1919_pct
            [740.0, 6830.0],       # all_types_total_fl_area_H_total
            [340.0, 3170.0],       # clean_res_premise_area_total
            [740.0, 6600.0],       # all_types_total_fl_area_FC_total
            [5.3, 100.0],          # 1919-1944_pct
            [350.0, 3340.0],       # all_types_premise_area_total
            [2.0, 43.0],           # clean_res_total_buildings
            [740.0, 6550.0],       # all_res_total_fl_area_FC_total
            [730.0, 6650.0],       # clean_res_total_fl_area_H_total
            [3.6, 100.0],          # Standard size detached_pct
            [730.0, 6410.0],       # clean_res_total_fl_area_FC_total
            [0.046, 0.33],         # postcode_density
            [2.7, 100.0]           # Post 1999_pct
        ]
    }
    
 

def check_and_enforce_heating_volume_constraint(x, problem_def):
    """
    Check and enforce the constraint that clean_res_heated_vol_h_total should equal
    all_res_heated_vol_h_total * (1 - Domestic_outbuilding_pct/100)
    
    Args:
        x: List of values corresponding to the variables in problem_def
        problem_def: Dictionary containing problem definition
        
    Returns:
        bool: Whether the constraint is satisfied
        float: The difference between actual and expected values
        float: The expected value of clean_res_heated_vol_h_total
    """
    # Get indices of relevant variables
    names = problem_def['names']
    all_res_idx = names.index('all_res_heated_vol_h_total')
    clean_res_idx = names.index('clean_res_heated_vol_h_total')
    outbuilding_idx = names.index('Domestic outbuilding_pct')
    
    # Calculate expected clean_res_heated_vol_h_total
    all_res_vol = x[all_res_idx]
    outbuilding_pct = x[outbuilding_idx]
    expected_clean_res = all_res_vol * (1 - outbuilding_pct/100)
    
    # Get actual value
    actual_clean_res = x[clean_res_idx]
    
    # Calculate difference
    difference = abs(actual_clean_res - expected_clean_res)
    
    # Check if constraint is satisfied (using small tolerance due to floating point arithmetic)
    is_satisfied = difference < 1e-10
    
    return is_satisfied, difference, expected_clean_res

def enforce_pc_area(x, problem_def):
    """
    Enforce the constraint by updating clean_res_heated_vol_h_total
  
    """
    # Create a copy of the input list
    x_new = x.copy()
    
    # Get indices of relevant variables
    names = problem_def['names']
    all_res_idx = names.index('postcode_area')
    clean_res_idx = names.index('log_pc_area')
    
    
    # Calculate and set the correct value
    all_res_vol = x[all_res_idx]
    
    x_new[clean_res_idx] = [np.log(x) for x in all_res_vol]
    print('new log pc',x_new[clean_res_idx] )
    return x_new

def enforce_heating_volume_constraint(x, problem_def):
    """
    Enforce the constraint by updating clean_res_heated_vol_h_total
    
    Args:
        x: List of values corresponding to the variables in problem_def
        problem_def: Dictionary containing problem definition
        
    Returns:
        List: Updated values with constraint enforced
    """
    # Create a copy of the input list
    x_new = x.copy()
    
    # Get indices of relevant variables
    names = problem_def['names']
    all_res_idx = names.index('all_res_heated_vol_h_total')
    clean_res_idx = names.index('clean_res_heated_vol_h_total')
    outbuilding_idx = names.index('Domestic outbuilding_pct')
    
    # Calculate and set the correct value
    all_res_vol = x[all_res_idx]
    outbuilding_pct = x[outbuilding_idx]
    print(outbuilding_pct)
    fraction = (1 - outbuilding_pct / 100 )
    if fraction.all() < 0 or fraction.all() > 1 : 
        raise ValueError('Fraction should be between 0 and 1')
    x_new[clean_res_idx] = all_res_vol * fraction 
    
    if x_new[clean_res_idx].all() < 0:
        raise ValueError('clean_res_heated_vol_h_total should be positive')
    return x_new

def generate_problem(col_setting):
    name, cols = settings_dict[col_setting]
    
    new_problem = {
        'num_vars': len(cols),
        'names': cols,
        'bounds': [base_params[col] for col in cols]
    }
    
    return new_problem

