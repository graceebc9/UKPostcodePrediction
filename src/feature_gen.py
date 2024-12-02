import pandas as pd 
import logging

cols =[
      'perc_large_houses',
      'perc_standard_houses',
      'perc_small_terraces',
       'perc_estates',
      'perc_all_flats',
      'perc_age_Pre-1919',
      'perc_age_1919-1999',
      'perc_age_Post-1999',
      'perc_white',
      'perc_asian',
      'Perc_econ_employed',
      'perc_econ_inactive', 
    'perc_econ_unemployed',
      'perc_hh_size_small',
        'perc_hh_size_medium',
        'perc_hh_size_large',
      ]


def econ_settings():
   econ_act = ['economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time',
 
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Full-time',
   ]
   econ_inac = ['economic_activity_perc_Economically inactive: Retired',
 'economic_activity_perc_Economically inactive: Student',
 'economic_activity_perc_Economically inactive: Looking after home or family',
 'economic_activity_perc_Economically inactive: Long-term sick or disabled',
 'economic_activity_perc_Economically inactive: Other' ] 
   
   unemp = ['economic_activity_perc_Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
   'economic_activity_perc_Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' 
   ]
   other = ['economic_activity_perc_Does not apply']

   list_cols = [econ_act, unemp, econ_inac, other] 
   names = ['Perc_econ_employed', 'perc_econ_unemployed', 'perc_econ_inactive', 'perc_econ_other' ] 
   return list_cols, names 


def eth_setting():
    white = [ 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British', 
    'ethnic_group_perc_White: Irish', 
    'ethnic_group_perc_White: Gypsy or Irish Traveller',
    'ethnic_group_perc_White: Roma', 'ethnic_group_perc_White: Other White',]
    black = [ 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: African',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Caribbean',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Other Black',] 
    asian = [ 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Bangladeshi',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Chinese',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Indian',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Pakistani',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Other Asian',] 
    other = ['ethnic_group_perc_Does not apply',
 'ethnic_group_perc_Other ethnic group: Arab',
 'ethnic_group_perc_Other ethnic group: Any other ethnic group',] 
    mixed = [ 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Asian',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black African',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black Caribbean',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups',]
    list_cols = [white, black, asian, other, mixed]
    names = ['perc_white', 'perc_black', 'perc_asian', 'perc_asian', 'perc_ethnic_other', 'perc_mixed']
    return list_cols , names  

def hh_size_setting():
    small = ['household_siz_perc_perc_0 people in household','household_siz_perc_perc_1 person in household']
    medium = ['household_siz_perc_perc_2 people in household', 'household_siz_perc_perc_3 people in household',
     'household_siz_perc_perc_4 people in household'] 
    large = ['household_siz_perc_perc_5 people in household',
    'household_siz_perc_perc_6 people in household',
    'household_siz_perc_perc_7 people in household']
    list_cols = [small, medium, large]
    names = ['perc_hh_size_small', 'perc_hh_size_medium', 'perc_hh_size_large']
    return list_cols, names

def type_setting1():
    large = ['Very large detached', 'Large detached','Large semi detached' ,  'Tall terraces 3-4 storeys']
    standard = ['Standard size semi detached',  'Standard size detached']
    large_flats = ['Very tall point block flats', 'Tall flats 6-15 storeys']
    med_flats = ['Medium height flats 5-6 storeys', '3-4 storey and smaller flats']
    small_terraces = ['Small low terraces', '2 storeys terraces with t rear extension', 'Semi type house in multiples']
    flats = large_flats + med_flats
    estates = ['Linked and step linked premises', 'Planned balanced mixed estates']
    unkn_typ =  ['all_unknown']
    outbuilds = ['Domestic outbuilding']
    list_cols = [large, standard,  small_terraces, estates,  flats, unkn_typ, outbuilds ]
    names = ['perc_large_houses', 'perc_standard_houses',  'perc_small_terraces', 'perc_estates', 'perc_all_flats', 'perc_unknown_typ', 'perc_outbuildings']
    return list_cols, names

def age_setting1():
    pre_1919 = ['Pre 1919']
    o1919_1999= ['1919-1944', '1945-1959', '1960-1979', '1980-1989', '1990-1999',]
    post_1999= ['Post 1999'] 
    unk = ['Unknown_age']
    age_cols = [pre_1919, o1919_1999, post_1999, unk]
    age_names = ['perc_age_Pre-1919', 'perc_age_1919-1999', 'perc_age_Post-1999', 'perc_age_Unknown']
    return  age_cols, age_names 


import pandas as pd
import numpy as np

def optimize_preprocessing(data):
    """
    Convert attributes into meta attributes by combinign certain subgroups 
    """
    # Pre-calculate all column mappings
    column_mappings = {
        'type': (type_setting1(), lambda x: x + '_pct'),
        'age': (age_setting1(), lambda x: x + '_pct'),
        'eth': (eth_setting(), lambda x: x),
        'econ': (econ_settings(), lambda x: x),
        'hh_size': (hh_size_setting(), lambda x: x)
    }
    
    # Vectorized operations instead of loops
    def process_columns(columns, names, transform_fn):
        # Create all column lists at once
        all_cols = [[transform_fn(x) for x in col_group] for col_group in columns]
        
        # Perform vectorized operations
        results = pd.DataFrame({
            name: data[cols].fillna(0).sum(axis=1) 
            for name, cols in zip(names, all_cols)
        })
        
        return results
    
    # Process all categories at once
    results = []
    for (columns, names), transform_fn in column_mappings.values():
        results.append(process_columns(columns, names, transform_fn))
    
    # Combine results efficiently
    return pd.concat([data] + results, axis=1)

# Modified function calls
def pre_process_pc(data):
    data =  optimize_preprocessing(data)
    return data 