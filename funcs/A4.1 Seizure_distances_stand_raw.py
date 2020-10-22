from pathlib import Path
import glob
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""Compute the seizure distances - Standardised"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

# folder = "seizure_stand_raw"

def upper_triu_values(X):
    """
    Function for returning the upper triangular values of a matrix X
    :param X: the initial matrix that will be used as input
    :return: the upper triangular values of matrix X as a vector
    """

    n = X.shape[0]
    r, c = np.triu_indices ( n=n, k=1 )
    upper_triu = X[r, c]
    return upper_triu


# in_path = files[0]

def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # folder = "figures_shuffled"
    id_patient = parts[-1]

    # Output directory for data-results
    out_subfolder = os.path.join (ROOT_DIR, result_file, id_patient, folder)
    os.makedirs ( out_subfolder, exist_ok=True )
    print ( "Processing file:", in_path )

    '''Seizure information for each patient'''
    # Read the seizure information for the corresponding patient
    print('Reading seizure information')
    filename_info = "{}_{}".format ( id_patient, "info.mat" )
    info_all = sio.loadmat ( os.path.join ( ROOT_DIR, info_path, filename_info ) )
    info_seizure = info_all["seizure_begin"]
    ## Investigating the seizure events
    info_seizure = np.floor ( info_seizure / 30 )
    seizures_all = [int ( x ) for x in info_seizure]
    n_seizures = len(seizures_all)

    if (n_seizures > 5):

        '''MEMD RESULTS'''
        # Import the file with the final MEMD and STEMD results
        print ( "{}{}".format ( "Reading MEMD mat file ", id_patient ) )
        filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
        #filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand_shuffled.mat"
        MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
        IMF_MEMD = MEMD_all["imf_memd"]
        # IMF_MEMD = MEMD_all["imf_perm_memd"]
        [n_comp, n_imfs, n_time] = IMF_MEMD.shape

        '''Read Seizure Dissimilarity'''
        # Import the file with the Seizure Dissimilarity results
        print ( "{}{}".format ( "Reading Seizure Dissimilarity matrix mat file ", id_patient ) )
        filename_AllzDissMat = "AllSzDissMat.mat"
        DissMat_all = sio.loadmat ( os.path.join ( ROOT_DIR, "data", "longterm_preproc_sz", id_patient, filename_AllzDissMat ) )['AllSzDissMat']
        DissMatFC_all = DissMat_all[3][0]

        '''Reading all seizure time distances and seizure euclidean distances'''
        print('Reading Seizure Distances')
        seizure_dist_path = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*Seizure_distances_raw"))
        seizures_dist_eucl_all = sio.loadmat ( os.path.join (seizure_dist_path[0],  "seizure_dist_eucl_{}.mat".format(id_patient) ) )
        seizures_time_dist = sio.loadmat ( os.path.join (seizure_dist_path[0],  "seizure_time_dist_{}.mat".format(id_patient) ) )['time_dist']

        '''Standardise the Dissimilarity Matrix'''
        DissMat_upper_triu = upper_triu_values ( DissMatFC_all.copy () )
        DissMatFC_all_stand = (DissMatFC_all - DissMat_upper_triu.mean())/DissMat_upper_triu.std()

        # Save standardised dissimilarity matrix as a mat file
        DissMatFC_all_stand_mat = {"DissFC_stand": DissMatFC_all_stand}
        sio.savemat(os.path.join(out_subfolder, "DissMatFC_stand_{}.mat".format(id_patient)), DissMatFC_all_stand_mat)

        '''Standardise the seizure distances for each IMF'''
        seizure_dist_eucl_stand = {}
        for imf in range(0, n_imfs):
            temp = seizures_dist_eucl_all['IMF{}'.format(imf+1)]
            SeizureDist_upper_triu = upper_triu_values (temp)
            SeizureDist_stand = (temp - SeizureDist_upper_triu.mean())/SeizureDist_upper_triu.std()

            seizure_dist_eucl_stand.__setitem__('IMF{}'.format(imf+1), SeizureDist_stand.copy())
        # Save as mat file
        sio.savemat(os.path.join(out_subfolder, "seizure_dist_eucl_stand_{}.mat".format(id_patient)), seizure_dist_eucl_stand)

        '''Standardise the seizure time distance'''
        seizure_dist_time_stand = {}
        temp = seizures_time_dist
        SeizureDist_upper_triu = upper_triu_values (temp)
        SeizureDist_stand = (temp - SeizureDist_upper_triu.mean())/SeizureDist_upper_triu.std()

        seizure_dist_time_stand.__setitem__('time_dist', SeizureDist_stand.copy())

        # Save as mat file
        sio.savemat(os.path.join(out_subfolder, "seizure_time_dist_stand_{}.mat".format(id_patient)), seizure_dist_time_stand)

def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

    # test the code
    files = files[5:6]

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
