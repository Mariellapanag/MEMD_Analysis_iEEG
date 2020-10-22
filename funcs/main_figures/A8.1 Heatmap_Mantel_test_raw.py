from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.ticker as tkr
import seaborn as sns

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

        '''Reading all Mantel p'''
        print('Reading all Mantel p')
        mantel_p_dist_eucl_all = sio.loadmat ( os.path.join (out_subfolder,  "mantel_p_dist_eucl_allperm_{}".format(id_patient) ) )

        '''Reading all Mantel q (after FDR)'''
        print('Reading all Mantel q')
        mantel_q_dist_eucl_all = sio.loadmat ( os.path.join (out_subfolder,  "mantel_q_dist_eucl_allperm_{}".format(id_patient) ) )

        print('Beginning plotting mantel results')
        print('seizures:{}'.format(n_seizures))

        if n_permutations == 1:

            '''Seizure distances for IMFs'''
            fig_name = "Mantel_pvalues_dist_eucl_{}.{}".format (id_patient, "pdf" )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                for id_perm in range(0, n_permutations):
                    '''Heatmap of Mantel test for H'''
                    manteltest_p_tr = mantel_p_dist_eucl_all['perm{}'.format(id_perm)]['pvalue'][0][0].copy ()
                    fig, ax = plt.subplots ()
                    x_axis_labels = np.arange ( 1, manteltest_p_tr.shape[1] + 1, 1 )
                    y_axis_labels = np.arange ( 1, manteltest_p_tr.shape[0] + 1, 1 )
                    formatter = tkr.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    g = sns.heatmap (manteltest_p_tr, xticklabels=x_axis_labels,
                                     yticklabels=y_axis_labels, cbar_kws={"format": formatter})
                    g.set_xlabel ( "IMFs" )
                    g.set_ylabel ( "Mantel" )
                    plt.title ( 'Mantel test p\n perm{}'.format(id_perm))

                    plt.tight_layout()
                    canvas = FigureCanvasPdf(fig)
                    canvas.print_figure(pages)
                    plt.close("all")

            fig_name = "Mantel_qvalues_dist_eucl_{}.{}".format (id_patient, "pdf" )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                for id_perm in range(0, n_permutations):
                    '''Heatmap of Mantel test for H'''
                    manteltest_q_tr = mantel_q_dist_eucl_all['qvalue'].copy ()
                    fig, ax = plt.subplots ()
                    formatter = tkr.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    x_axis_labels = np.arange ( 1, manteltest_q_tr.shape[1] + 1, 1 )
                    y_axis_labels = np.arange ( 1, manteltest_q_tr.shape[0] + 1, 1 )
                    g = sns.heatmap (manteltest_q_tr, xticklabels=x_axis_labels,
                                     yticklabels=y_axis_labels, cbar_kws={"format": formatter}, annot = True)
                    g.set_xlabel ( "IMFs" )
                    g.set_ylabel ( "Mantel" )
                    plt.title ( 'Mantel test q\n perm{}'.format(id_perm))

                    plt.tight_layout()
                    canvas = FigureCanvasPdf(fig)
                    canvas.print_figure(pages)
                    plt.close("all")

            p_values = [0.001, 0.05, 0.01]
            for id_perm in range(0, n_permutations):
                fig_name = "Mantel_spearman_qvalues_dist_eucl_{}_{}.{}".format (id_perm, id_patient, "pdf" )
                with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                    for pvalue in p_values:
                        '''Heatmap of Mantel test for IMFs'''
                        spearman_corr = mantel_p_dist_eucl_all['perm{}'.format(id_perm)]['spearman_corr'][0][0]
                        manteltest_q_tr = mantel_q_dist_eucl_all['qvalue']
                        masktest_sig = manteltest_q_tr > pvalue
                        fig, ax = plt.subplots ()
                        formatter = tkr.ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        x_axis_labels = np.arange ( 1, manteltest_q_tr.shape[1] + 1, 1 )
                        y_axis_labels = np.arange ( 1, manteltest_q_tr.shape[0] + 1, 1 )
                        g = sns.heatmap (spearman_corr, xticklabels=x_axis_labels, vmin = -1, vmax = 1,
                                         yticklabels=y_axis_labels, mask = masktest_sig, cbar_kws={"format": formatter})
                        g.set_xlabel ( "IMFs" )
                        g.set_ylabel ( "Spearman" )
                        plt.title ( 'Spearman q perm{} \n Grey squares depicts to non-significance pvalue: {}'.format(id_perm, pvalue ))

                        plt.tight_layout()
                        canvas = FigureCanvasPdf(fig)
                        canvas.print_figure(pages)
                        plt.close("all")


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
