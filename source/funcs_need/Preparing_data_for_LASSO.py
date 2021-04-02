from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# in_path = files[0]

def upper_triu_values(A):

    n = A.shape[0]
    r, c = np.triu_indices ( n=n, k=1 )
    upper_triu = A[r, c]
    return upper_triu

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

        '''Read Seizure Dissimilarity'''
        # Import the file with the Seizure Dissimilarity results
        print ( "{}{}".format ( "Reading Seizure Dissimilarity matrix mat file ", id_patient ) )
        filename_AllzDissMat = "AllSzDissMat.mat"
        DissMat_all = sio.loadmat ( os.path.join ( ROOT_DIR, input_path, id_patient, filename_AllzDissMat ) )['AllSzDissMat']
        DissMatFC_all = DissMat_all[3][0]

        '''Reading all seizure distances'''
        print('Reading Seizure Distances')
        seizures_dist_eucl_all = sio.loadmat ( os.path.join (ROOT_DIR, result_file, id_patient, "sz_dist_raw",  "seizure_dist_eucl_{}".format(id_patient) ) )
        seizures_time_dist = sio.loadmat ( os.path.join (ROOT_DIR, result_file, id_patient, "sz_dist_raw",  "seizure_time_dist_{}".format(id_patient) ) )

        # Temporal Distance Matrix
        seizures_upper_triu = upper_triu_values ( seizures_time_dist['time_dist'])

        # Dissimilarity matrix upper triangular elements
        upper_triu_FC = upper_triu_values(DissMatFC_all.copy())

        '''Distance Matrices for all IMFS'''
        df_dict_eucl = {"sz_diss_FC": upper_triu_FC, "time_dist": seizures_upper_triu}

        eucl_names = list(seizures_dist_eucl_all.keys())
        eucl_names_keep = [name for name in eucl_names if name not in ['__header__', '__globals__','__version__']]
        eucl_dist_upper_triu = {k: upper_triu_values( seizures_dist_eucl_all[k]) for k in eucl_names_keep}

        # Update the dictionary with seizure distances obtained from all IMFs and DIMs
        df_dict_eucl.update(eucl_dist_upper_triu)

        # Convert dictionary to a dataframe
        eucl_data_df = pd.DataFrame (df_dict_eucl)

        eucl_data_reg = eucl_data_df.copy()
        # from sklearn.preprocessing import StandardScaler
        # data_reg1 = StandardScaler().fit_transform ( data_reg )
        #
        # Standardise the data
        eucl_data_reg_stand = eucl_data_reg.apply(lambda x: (x-x.mean()) / x.std())

        # Save csv files
        eucl_data_reg.to_csv ( os.path.join ( out_subfolder, 'data_for_Modelling_eucldist.csv' ), index=False )
        eucl_data_reg_stand.to_csv ( os.path.join ( out_subfolder, 'data_standardise_for_Modelling_eucldist.csv' ), index=False )
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
        futures = {executor.submit ( process_file, in_path ) for in_path in files}
        for future in as_completed ( futures ):
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process ()