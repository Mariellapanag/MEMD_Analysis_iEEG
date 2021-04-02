from pathlib import Path
import glob
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import scipy.stats
import seaborn as sns

from paths import ROOT_DIR
import funcs.Mantel.Mantel as Mantel
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""Mantel test"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

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
        MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
        IMF_MEMD = MEMD_all["imf_memd"]
        [n_comp, n_imfs, n_time] = IMF_MEMD.shape

        '''Read Seizure Dissimilarity'''
        # Import the file with the Seizure Dissimilarity results
        print ( "{}{}".format ( "Reading Seizure Dissimilarity matrix mat file ", id_patient ) )
        filename_AllzDissMat = "AllSzDissMat.mat"
        DissMat_all = sio.loadmat ( os.path.join ( ROOT_DIR, "data", "longterm_preproc", id_patient, filename_AllzDissMat ) )['AllSzDissMat']
        DissMatFC_all = DissMat_all[3][0]

        '''Reading all seizure distances'''
        print('Reading Seizure Distances')

        '''Reading all seizure time distances and seizure euclidean distances'''
        print('Reading Seizure Distances')
        seizure_dist_path = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "sz_dist_raw"))
        seizures_dist_eucl_all = sio.loadmat ( os.path.join (seizure_dist_path[0],  "seizure_dist_eucl_{}.mat".format(id_patient) ) )
        seizures_time_dist = sio.loadmat ( os.path.join (seizure_dist_path[0],  "seizure_time_dist_{}.mat".format(id_patient) ) )['time_dist']

        print('Beginning plotting seizure dissimilarity')
        print('seizures:{}'.format(n_seizures))

        '''Spearman Correlation and Mantel test'''
        dist1 = DissMatFC_all.copy()
        dist1_array = squareform(dist1).ravel()

        '''Seizure distances for all IMFs'''
        mantel_results_dist_eucl_perm = {}
        spearman_corr = np.zeros(n_imfs)
        mantel_pvalue = np.zeros(n_imfs)
        z_values = np.zeros(n_imfs)
        for imf in range(0, n_imfs):
            dist2 = seizures_dist_eucl_all['IMF{}'.format(imf+1)].copy()
            dist2_array = squareform(dist2).ravel().tolist()
            n_perm = 10000
            mantel_test = Mantel.test ( dist1_array, dist2_array, perms=n_perm, method='spearman', tail='upper' )
            spearman_corr[imf] = mantel_test[0]
            mantel_pvalue[imf] = mantel_test[1]
            z_values[imf] = mantel_test[2]

        mantel_results = {'pvalue': mantel_pvalue, 'spearman_corr': spearman_corr,'z_value': z_values}
        mantel_results_dist_eucl_perm['mantel_eucl_dist'] = mantel_results

        sio.savemat ( os.path.join ( out_subfolder, "mantel_p_dist_eucl_{}.mat".format(id_patient) ), mantel_results_dist_eucl_perm )

        '''Scatterplot'''
        fig_name = "scatterplot_dist_eucl_all_{}.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for imf in range(0, n_imfs):
                dist2 = seizures_dist_eucl_all['IMF{}'.format(imf+1)].copy()
                dist2_array = squareform(dist2).ravel().tolist()
                rho, pvalue = scipy.stats.spearmanr(dist1_array, dist2_array)
                fig, ax = plt.subplots(figsize=(10,8))

                g = sns.scatterplot(x = dist1_array, y = dist2_array)
                g.set_xlabel ( "Seizure Dissimilarity" )
                g.set_ylabel ( "Seizure IMF distance" )
                g.set_title ( 'Seizure Dissimilarity vs Seizure IMF{} Distance \n Spearman {}'.format(imf+1, round(rho, 3)))
                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

        '''Time seizure distance'''
        mantel_results_time_dist_perm = {}
        dist2 = seizures_time_dist.copy()
        dist2_array = squareform(dist2).ravel().tolist()
        n_perm = 10000
        mantel_test = Mantel.test ( dist1_array, dist2_array, perms=n_perm, method='spearman', tail='upper' )
        spearman_corr = mantel_test[0]
        mantel_pvalue = mantel_test[1]
        z_values = mantel_test[2]

        mantel_results = {'pvalue': mantel_pvalue, 'spearman_corr': spearman_corr,'z_value': z_values}
        mantel_results_time_dist_perm['mantel_time_dist'] = mantel_results

        sio.savemat ( os.path.join ( out_subfolder, "mantel_p_time_dist_{}.mat".format(id_patient) ), mantel_results_time_dist_perm )

        '''Scatterplot'''
        fig_name = "scatterplot_time_dist_all_{}.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            dist2 = seizures_time_dist.copy()
            dist2_array = squareform(dist2).ravel().tolist()
            rho, pvalue = scipy.stats.spearmanr(dist1_array, dist2_array)
            fig, ax = plt.subplots(figsize=(12,8))

            g = sns.scatterplot(x = dist1_array, y = dist2_array)
            ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
            g.set_xlabel ( "Seizure Diss" )
            g.set_ylabel ( "Seizure time distance" )
            g.set_title ( 'Seizure dissimilarity vs seizure time distance \n Spearman {}'.format(round(rho,3)))
            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")
    return True

def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

    # Run for a number of patients
    # files = [files[i] for i in [3,5,8,9,11,12, 13]]
    # Run for one patient
    # files = files[5:6]

    start_time = time.time ()
    with ProcessPoolExecutor ( max_workers=3 ) as executor:
        futures = [executor.submit ( process_file, in_path ) for in_path in files]
        for future in as_completed ( futures ):
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process ()
