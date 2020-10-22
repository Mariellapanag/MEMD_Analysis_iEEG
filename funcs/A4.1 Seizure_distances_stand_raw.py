from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt

from MEMD_funcs.Global_settings.main import ROOT_DIR
from MEMD_funcs.Global_settings.global_settings_plots import *

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
output_path = os.path.join ( "final_results" )
folder = "figures"

'''Define the output subfolder name'''
subfolder = "Initial_data(No permutation)"


out_subfolder_name = 'seizure_dist_Raw_Analysis'

'''Choose run name from the following: 'initial', 'shuffle_total_random', 'shuffle_seizure_random', 'shuffle_seizure_slice' '''
RUN_NAME = 'initial'

selected_number = 50

if RUN_NAME == 'initial':
    sub_name = 'Initial_data'
elif RUN_NAME == 'shuffle_total_random':
    sub_name = 'shuffle_randomly_{}'.format(selected_number)
elif RUN_NAME == 'shuffle_seizure_random':
    sub_name = 'shuffle_seizure_times_{}'.format(selected_number)
elif RUN_NAME == 'shuffle_seizure_slice':
    sub_name = 'shuffle_seizure_vector_{}'.format(selected_number)
else:
    print('Please choose one of the available options for the parameter RUN_NAME')

def upper_triu_values(A):

    n = A.shape[0]
    r, c = np.triu_indices ( n=n, k=1 )
    upper_triu = A[r, c]
    return upper_triu

# in_path = files[0]


def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # folder = "figures_shuffled"
    id_patient = parts[-1]

    # Ouput directory for data-results
    out_subfolder = os.path.join (ROOT_DIR, output_path, folder, id_patient, subfolder, sub_name,  out_subfolder_name)
    os.makedirs ( out_subfolder, exist_ok=True )
    print ( "Processing file:", in_path )

    '''Seizure information for each patient'''
    # Read the seizure information for the corresponding patient
    print('Reading Seizure Timings')
    seizures_file = sio.loadmat ( os.path.join (ROOT_DIR, output_path, folder, id_patient, subfolder, sub_name,  "seizure_all_{}".format(id_patient) ) )
    # seizures_all = seizures_file["seizures"]
    n_seizures = seizures_file["n_seizures"][0][0]
    n_permutations = seizures_file["n_permutations"][0][0]

    if (n_seizures > 5):

        '''NMF RESULTS - PLOTS'''
        #################################################################
        filename_nmf = "NMF_BP_CA_normedBand.mat"
        NMF_all = sio.loadmat ( os.path.join ( in_path, filename_nmf ) )
        H = NMF_all["H"]
        # W = NMF_all["W"]
        row, col = H.shape

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
        # DissMatFreqB_all = DissMat_all[1][0]
        DissMatFC_all = DissMat_all[3][0]

        '''Reading all seizure distances'''
        print('Reading Seizure Distances')
        seizures_dist_eucl_all = sio.loadmat ( os.path.join (out_subfolder,  "seizure_dist_eucl_{}".format(id_patient) ) )
        seizures_time_dist = sio.loadmat ( os.path.join (out_subfolder,  "seizure_time_dist_{}".format(id_patient) ) )


        if n_permutations == 1:
            '''Standardise the Dissimilarity Matrix'''
            DissMat_upper_triu = upper_triu_values ( DissMatFC_all.copy () )
            DissMatFC_all_stand = (DissMatFC_all - DissMat_upper_triu.mean())/DissMat_upper_triu.std()

            # Save standardised dissimilarity matrix as a mat file
            DissMatFC_all_stand_mat = {"DissFC_stand": DissMatFC_all_stand}
            sio.savemat(os.path.join(out_subfolder, "DissMatFC_stand_{}.mat".format(id_patient)), DissMatFC_all_stand_mat)

        '''Standardise the seizure distances for each IMF'''
        seizure_dist_eucl_stand = {}
        for id_perm in range ( 0, n_permutations):
            seizure_dist_e_stand = {}
            for imf in range(0, n_imfs):
                temp = seizures_dist_eucl_all['perm{}'.format(id_perm)]['IMF{}'.format(imf+1)][0][0]
                SeizureDist_upper_triu = upper_triu_values (temp)
                SeizureDist_stand = (temp - SeizureDist_upper_triu.mean())/SeizureDist_upper_triu.std()

                seizure_dist_e_stand.__setitem__('IMF{}'.format(imf+1), SeizureDist_stand.copy())

            seizure_dist_eucl_stand['perm{}'.format(id_perm)] = seizure_dist_e_stand
        # Save as mat file
        sio.savemat(os.path.join(out_subfolder, "seizure_dist_eucl_stand_{}.mat".format(id_patient)), seizure_dist_eucl_stand)

        '''Standardise the seizure time distance'''
        seizure_dist_time_stand = {}
        for id_perm in range ( 0, n_permutations):
            temp = seizures_time_dist['perm{}'.format(id_perm)]
            SeizureDist_upper_triu = upper_triu_values (temp)
            SeizureDist_stand = (temp - SeizureDist_upper_triu.mean())/SeizureDist_upper_triu.std()

            seizure_dist_time_stand.__setitem__('perm{}'.format(id_perm), SeizureDist_stand.copy())

        # Save as mat file
        sio.savemat(os.path.join(out_subfolder, "seizure_time_dist_stand_{}.mat".format(id_patient)), seizure_dist_time_stand)

def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]
    # files = [files[i] for i in [3,5,8,9,11,12, 13]]
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
