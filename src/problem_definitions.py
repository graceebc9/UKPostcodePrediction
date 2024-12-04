

    # problem_definitions.py
from .column_settings import settings_dict
import numpy as np 

# Base dictionary of all possible parameters and their bounds
base_params = {
    'all_res_heated_vol_h_total': [0, 7600],
    'all_types_total_buildings': [0, 150],
    'clean_res_total_buildings': [0, 150],
    'all_res_total_buildings': [0, 150],
    'clean_res_heated_vol_h_total': [0, 7400],
    'Domestic outbuilding_pct': [0, 52],
    'Standard size detached_pct': [0, 100],
    'Standard size semi detached_pct': [0, 100],
    'Small low terraces_pct': [0, 100],
    '2 storeys terraces with t rear extension_pct': [0, 100],
    'Pre 1919_pct': [0, 100],
    'Unknown_age_pct': [0, 20],
    '1960-1979_pct': [0, 100],
    '1919-1944_pct': [0, 100],
    'Post 1999_pct': [0, 100],
    '1945-1959_pct': [0, 100],
    '1980-1989_pct': [0, 100],
    '1990-1999_pct': [0, 100],
    'Post 1999_pct': [0, 100],  
    'None_age_pct': [0, 15],
    'HDD': [30, 80],
    'CDD': [0, 8],
    'HDD_summer': [3, 15],
    'HDD_winter': [30, 60],
    'postcode_area': [1, 26000],
    'postcode_density': [0, 0.5],
    'log_pc_area': [5, 12],
    'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British': [0, 1],
    'central_heating_perc_Mains gas only': [0, 1],
    'household_siz_perc_perc_1 person in household': [0, 1],
    'Average Household Size': [1, 5],
    'clean_res_premise_area_total': [0, 2000],
    'all_res_premise_area_total': [0, 2000],
    'all_res_base_floor_total': [0, 1000]
}


problem_feat_cols = {
    'num_vars': 26,
    'names': [
        'all_types_total_fl_area_H_total',
        'all_types_premise_area_total',
        'clean_res_total_buildings',
        'clean_res_premise_area_total',
        'clean_res_total_fl_area_H_total',
        'clean_res_base_floor_total',
        'Domestic outbuilding_pct',
        'Standard size detached_pct',
        'Standard size semi detached_pct',
        'Small low terraces_pct',
        '2 storeys terraces with t rear extension_pct',
        'Pre 1919_pct',
        'all_none_age_pct',
        '1960-1979_pct',
        '1919-1944_pct',
        'Post 1999_pct',
        'HDD',
        'CDD',
        'HDD_summer',
        'HDD_winter',
        'postcode_area',
        'postcode_density',
        'log_pc_area',
        'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
        'central_heating_perc_Mains gas only',
        'household_siz_perc_perc_1 person in household'
    ],
    'bounds': [
        [740.0, 6830.0],    # all_types_total_fl_area_H_total
        [350.0, 3340.0],    # all_types_premise_area_total
        [2, 43],            # clean_res_total_buildings
        [340.0, 3170.0],    # clean_res_premise_area_total
        [730.0, 6650.0],    # clean_res_total_fl_area_H_total
        [0.0, 0.0],         # clean_res_base_floor_total
        [3.2, 40.9],        # Domestic outbuilding_pct
        [3.6, 100],         # Standard size detached_pct
        [6.9, 100],         # Standard size semi detached_pct
        [6.8, 100],         # Small low terraces_pct
        [10.2, 100],        # 2 storeys terraces with t rear extension_pct
        [5.6, 100],         # Pre 1919_pct
        [0, 0.0],           # all_none_age_pct
        [4.0, 100],         # 1960-1979_pct
        [5.3, 100],         # 1919-1944_pct
        [2.7, 100],         # Post 1999_pct
        [46.24, 65.34],     # HDD
        [0.0, 5.42],        # CDD
        [6.03, 13.23],      # HDD_summer
        [39.65, 52.34],     # HDD_winter
        [1830.0, 35460.0],  # postcode_area
        [0.046, 0.33],      # postcode_density
        [7.51, 10.47],       # log_pc_area
        [0.3, 1.0],         # ethnic_group_perc_White
        [0.6, 0.9],         # central_heating_perc_Mains gas only
        [0.1, 0.5]          # household_siz_perc_perc_1 person in household
    ],
    'groups': [
        'G1',     # all_types_total_fl_area_H_total (area measures group)
        'G1',     # all_types_premise_area_total
        'G2',     # clean_res_total_buildings (independent)
        'G1',     # clean_res_premise_area_total
        'G1',     # clean_res_total_fl_area_H_total
        'G1',     # clean_res_base_floor_total
        'G3',     # Domestic outbuilding_pct (building type group)
        'G4',     # Standard size detached_pct
        'G5',     # Standard size semi detached_pct
        'G6',     # Small low terraces_pct
        'G7',     # 2 storeys terraces with t rear extension_pct
        'G8',     # Pre 1919_pct (age group)
        'G9',     # all_none_age_pct
        'G10',    # 1960-1979_pct
        'G11',    # 1919-1944_pct
        'G12',    # Post 1999_pct
        'G13',    # HDD (temperature group)
        'G14',    # CDD
        'G13',    # HDD_summer
        'G13',    # HDD_winter
        'G15',    # postcode_area (postal metrics group)
        'G16',    # postcode_density
        'G15',    # log_pc_area
        'G17',    # ethnic_group_perc_White (independent)
        'G18',    # central_heating_perc_Mains gas only (independent)
        'G19'     # household_siz_perc_perc_1 person in household (independent)
    ]
}



problem_feat_cols_final_excl_region = {
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
            [0, 1],                # all_res_total_fl_area_H_total (empty)
            [5.6, 100.0],          # Pre 1919_pct
            [740.0, 6830.0],       # all_types_total_fl_area_H_total
            [340.0, 3170.0],       # clean_res_premise_area_total
            [740.0, 6600.0],       # all_types_total_fl_area_FC_total
            [5.3, 100.0],          # 1919-1944_pct
            [350.0, 3340.0],       # all_types_premise_area_total
            [2.0, 43.0],           # clean_res_total_buildings
            [0, 1],                # all_res_total_fl_area_FC_total (empty)
            [730.0, 6650.0],       # clean_res_total_fl_area_H_total
            [3.6, 100.0],          # Standard size detached_pct
            [730.0, 6410.0],       # clean_res_total_fl_area_FC_total
            [0.046, 0.33],         # postcode_density
            [2.7, 100.0]           # Post 1999_pct
        ],
        'groups': [
            'building_count',    # all_types_uprn_count_total
            'building_count',    # clean_res_uprn_count_total
            'floor_area',      # all_res_total_fl_area_H_total
            'age1',              # Pre 1919_pct
            'floor_area',      # all_types_total_fl_area_H_total
            'floor_area',       # clean_res_premise_area_total
            'floor_area',       # all_types_total_fl_area_FC_total
            'age2',              # 1919-1944_pct
            'floor_area',       # all_types_premise_area_total
            'building_count',   # clean_res_total_buildings
            'floor_area',       # all_res_total_fl_area_FC_total
            'floor_area',      # clean_res_total_fl_area_H_total
            'building_type',    # Standard size detached_pct
            'floor_area',       # clean_res_total_fl_area_FC_total
            'density',          # postcode_density
            'age3'              # Post 1999_pct
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

