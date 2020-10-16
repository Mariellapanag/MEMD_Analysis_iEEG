from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from funcs.Global_settings.main import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""
Extract the relevant information for seizures
"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

'''Define the output path'''
output_path = os.path.join ( "final_results" )
folder = "figures"

'''Define the output subfolder name'''
subfolder = "Initial_data(No permutation)"

'''Choose run name from the following: 'initial', 'shuffle_total_random', 'shuffle_seizure_random', 'shuffle_seizure_slice' '''
RUN_NAME = 'shuffle_seizure_slice'

selected_number = 50

if RUN_NAME == 'initial':
    sub_name = 'Initial_data'
    n_permutations = 1
elif RUN_NAME == 'shuffle_total_random':
    sub_name = 'shuffle_randomly_{}'.format(selected_number)
    n_permutations = selected_number
elif RUN_NAME == 'shuffle_seizure_random':
    sub_name = 'shuffle_seizure_times_{}'.format(selected_number)
    n_permutations = selected_number
elif RUN_NAME == 'shuffle_seizure_slice':
    sub_name = 'shuffle_seizure_vector_{}'.format(selected_number)
    n_permutations = selected_number
else:
    print('Please choose one of the available options for the parameter RUN_NAME')

# in_path = files[0]


def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # folder = "figures_shuffled"
    id_patient = parts[-1]

    # Ouput directory for data-results
    out_subfolder = os.path.join (ROOT_DIR, output_path, folder, id_patient, subfolder, sub_name )
    os.makedirs ( out_subfolder, exist_ok=True )
    print ( "Processing file:", in_path )

    '''MEMD RESULTS'''
    # Import the file with the final MEMD and STEMD results
    print ( "{}{}".format ( "Reading MEMD mat file ", id_patient ) )
    filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
    #filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand_shuffled.mat"
    MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
    IMF_MEMD = MEMD_all["imf_memd"]
    # IMF_MEMD = MEMD_all["imf_perm_memd"]
    [n_comp, n_imfs, n_time] = IMF_MEMD.shape


    '''Seizure information for each patient'''
    # Read the seizure information for the corresponding patient
    print('Reading seizure information')
    filename_info = "{}_{}".format ( id_patient, "info.mat" )
    info_all = sio.loadmat ( os.path.join ( ROOT_DIR, info_path, filename_info ) )
    info_seizure = info_all["seizure_begin"]
    ## Investigating the seizure events
    info_seizure_new = np.floor ( info_seizure / 30 )
    info_seizure_new = [int ( x ) for x in info_seizure_new]

    size = len ( info_seizure_new )
    seizures_idx_shuffled = np.zeros([n_permutations, size])

    for n in range(0, n_permutations):
        if RUN_NAME == 'initial':
            seizures_idx_shuffled[n, :] = info_seizure_new.copy()

        elif RUN_NAME == 'shuffle_total_random':
            seizures_idx_shuffled[n, :] = np.random.choice ( np.arange ( n_time ), size=size, replace=False, p=None )

        elif RUN_NAME == 'shuffle_seizure_random':
            seizures_idx_shuffled[n, :] = random.sample( info_seizure_new, size )

        elif RUN_NAME == 'shuffle_seizure_slice':
            l_last = n_time - (info_seizure_new[-1] - info_seizure_new[0])
            random_start = np.random.choice ( np.arange ( l_last ), size=1, replace=False, p=None )
            random_idx = np.concatenate((random_start, np.diff(info_seizure_new)), axis = None)
            seizures_idx_shuffled[n, :] = np.cumsum(random_idx)
        else:
            print ( 'Please choose one of the available options for the parameter RUN_NAME' )

    all_seizures = {'seizures': seizures_idx_shuffled, "n_seizures": len(info_seizure_new), "n_permutations": n_permutations}
    sio.savemat(os.path.join(out_subfolder, "seizure_all_{}.mat".format(id_patient)), all_seizures)


def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]
    # files = [files[i] for i in [3,5,8,9,11,12, 13]]
    # test the code
    # files = files[7:8]

    start_time = time.time ()
    # Test to make sure concurrent map is working
    with ProcessPoolExecutor ( max_workers=4 ) as executor:
        futures = [executor.submit ( process_file, in_path ) for in_path in files]
        for future in as_completed ( futures ):
            # if future.result() == True:
            processed += 1
            print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process ()
