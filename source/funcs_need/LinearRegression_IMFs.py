import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

from funcs.Global_settings.global_settings_plots import *
from paths import ROOT_DIR
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""Compute the seizure distance based on either the initial seizure timings or the shuffled ones using each IMF and Dimension for all patients
Save the results. Implement Mantel test and save the results as well. 
"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

'''Define the output path'''
# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

name_character = "eucldist"

# in_path = files[0]

def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # folder = "figures_shuffled"
    id_patient = parts[-1]

    # Ouput directory for data-results
    out_subfolder = os.path.join (ROOT_DIR, result_file, id_patient, folder)
    os.makedirs ( out_subfolder, exist_ok=True )
    print ( "Processing file:", in_path )

    '''Seizure information for each patient'''
    # Read the seizure information for the corresponding patient
    print('Reading Seizure Timings')
    seizures_file = sio.loadmat ( os.path.join (in_path,  "seizure_all_{}".format(id_patient) ) )
    #seizures_all = seizures_file["seizures"]
    n_seizures = seizures_file["n_seizures"][0][0]

    if (n_seizures > 5):

        inpath = os.path.join(ROOT_DIR, result_file, id_patient, "Constrained_LASSO_IMFs")
        data_stand = pd.read_csv(os.path.join ( inpath, 'data_stand_from_constrained_lasso_{}{}.csv'.format(name_character, id_patient) ) )

        y_df = data_stand['y'].copy ()
        y = np.array ( y_df )
        X_df = data_stand.drop ( columns='y', axis=1 ).copy ()
        X = np.array ( X_df )

        X1 = sm.add_constant(X)
        result = sm.OLS(y, X1).fit()
        print(result.rsquared, result.rsquared_adj)

        lmResults = pd.DataFrame({"r_squared": result.rsquared, "adj_r_squared": result.rsquared_adj}, index=[0])

        # save csv files
        lmResults.to_csv ( os.path.join ( out_subfolder, 'lm_results_{}_{}.csv'.format(name_character, id_patient) ), index=False )

    return True

def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]
    # files = [files[i] for i in [3,5,8,9,11,12, 13]]
    # test the code
    #files = files[5:6]

    start_time = time.time ()
    # Test to make sure concurrent map is working
    with ProcessPoolExecutor ( max_workers=4 ) as executor:
        futures = [executor.submit ( process_file, in_path ) for in_path in files]
        for future in as_completed ( futures ):
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process ()