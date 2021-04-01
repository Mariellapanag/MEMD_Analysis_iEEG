from pathlib import Path
import glob
import scipy.io as sio
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""FDR after Mantel test for all patients"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
files_list = [glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "Mantel_test_raw")) for id_patient in folders]
keywordFilter = ["FDR"]
files = [sent for sent in files_list if not any(word in sent for word in keywordFilter)]

#file = file[0]
def run_process():

    Manteltest_eucl_dist_p_list = list()
    eucl_dist_len = list([0])
    patient_id = list ()

    for file in files:

        # Extract path components
        parts = Path(file[0]).parts

        id_patient = parts[-2]

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

            '''Reading all mantel p_values'''
            print('Reading Seizure Distances')
            mantel_p_dist_eucl_all = sio.loadmat ( os.path.join (file[0],  "mantel_p_dist_eucl_{}".format(id_patient) ) )['mantel_eucl_dist']

            # Mantel test for euclidean distance
            mantel_eucl_dist_p = mantel_p_dist_eucl_all['pvalue'][0][0]
            Manteltest_eucl_dist_p_list.append(mantel_eucl_dist_p.ravel())
            eucl_dist_len.append(mantel_eucl_dist_p.shape[1])
            patient_id.append(id_patient)

            # Mantel test for euclidean distance
            eucl_dist_len_p_last = np.cumsum(eucl_dist_len)
            eucl_dist_Manteltest_p_all = np.hstack(Manteltest_eucl_dist_p_list)

    # Run the FDR to all patients
    Manteltest_q_eucl = multipletests ( eucl_dist_Manteltest_p_all.ravel (), method='fdr_bh' )[1]


    for i in patient_id:
        indx = patient_id.index(i)

        # Mantel test for euclidean distance
        Manteltest_eucl_dist_l = Manteltest_q_eucl[eucl_dist_len_p_last[indx]: eucl_dist_len_p_last[indx+1]]
        Mantel_eucl_dist_q = {'qvalue': Manteltest_eucl_dist_l}

        # Output directory for data-results
        out_subfolder = os.path.join (ROOT_DIR, result_file, i, folder)
        os.makedirs ( out_subfolder, exist_ok=True )
        print ( "Saving q-values:", out_subfolder )
        sio.savemat(os.path.join(out_subfolder, "mantel_q_dist_eucl_{}.mat".format(i)), Mantel_eucl_dist_q)


if __name__ == "__main__":
    run_process()
