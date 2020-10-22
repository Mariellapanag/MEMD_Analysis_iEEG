from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""
Computing the temporal distance and the Multivariate Euclidean distance for each IMF  
"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.slpit(".py")[0]
# in_path = files[0]

def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    id_patient = parts[-1]

    # Ouput directory for data-results
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

        '''NMF RESULTS'''
        #################################################################
        filename_nmf = "NMF_BP_CA_normedBand.mat"
        NMF_all = sio.loadmat ( os.path.join ( in_path, filename_nmf ) )
        H = NMF_all["H"]
        W = NMF_all["W"]
        row, col = H.shape

        '''MEMD RESULTS'''
        # Import the file with the final MEMD and STEMD results
        print ( "{}{}".format ( "Reading MEMD mat file ", id_patient ) )
        filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
        MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
        IMF_MEMD = MEMD_all["imf_memd"]
        [n_comp, n_imfs, n_time] = IMF_MEMD.shape

        ''' Computation of temporal Distance'''
        seizure_time_dist_perm = {}
        values_seizures = np.zeros((n_seizures, n_seizures))
        for i in range ( 0, n_seizures ):
            for j in range ( 0, n_seizures ):
                # Computation of seizure temporal distance
                values_seizures[i, j] = abs (seizures_all[j] - seizures_all[i])

        seizure_time_dist_perm.__setitem__('time_dist', values_seizures.copy())
        # save the seizure distance matrix as a mat file
        sio.savemat(os.path.join(out_subfolder, "seizure_time_dist_{}.mat".format(id_patient)), seizure_time_dist_perm)

        '''Euclidean distance for IMFs across Dimensions'''
        print('Beginning calculating seizure distance for each IMF across Dimensions')
        seizure_dist_mEucl_perm = {}

        seizure_dist_mEucl = {}
        for imf in range(0, n_imfs):
            Raw_Matrix = np.matmul(W, IMF_MEMD[:, imf, :])
            multi_euclidean_dist = np.zeros ( (n_seizures, n_seizures) )
            for i in range ( 0, n_seizures ):
                for j in range ( 0, n_seizures ):
                    xx = Raw_Matrix[:, int ( seizures_all[i] )]
                    yy = Raw_Matrix[:, int ( seizures_all[j] )]
                    multi_euclidean_dist[i,j] = distance.euclidean(xx,yy)
            # seizure_dist_all.update(dict_value)
            seizure_dist_mEucl.__setitem__ ( 'IMF{}'.format ( imf + 1 ), multi_euclidean_dist.copy () )

        seizure_dist_mEucl_perm['eucl_dist'] = seizure_dist_mEucl
        # Save the Seizure distance matrix as a mat file
        sio.savemat ( os.path.join ( out_subfolder, "seizure_dist_eucl_{}.mat".format ( id_patient ) ), seizure_dist_mEucl_perm )



def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

    # Uncomment this line if you want to run the analysis for one patient
    # files = files[5:6]

    start_time = time.time ()
    # Parallel processing
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
