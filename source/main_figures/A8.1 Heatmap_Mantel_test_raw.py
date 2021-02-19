from pathlib import Path
import glob
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

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""Heatmap """

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

        '''Reading all Mantel p'''
        print('Reading all Mantel p')
        mantel_p_path_list = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*Mantel_test_raw"))
        keywordFilter = ["FDR"]
        mantel_p_path = [sent for sent in mantel_p_path_list if not any(word in sent for word in keywordFilter)]
        mantel_p_dist_eucl_all = sio.loadmat ( os.path.join (mantel_p_path[0], "mantel_p_dist_eucl_{}".format(id_patient) ) )['mantel_eucl_dist']

        '''Reading all Mantel q (after FDR)'''
        print('Reading all Mantel q')
        mantel_q_path = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*FDR_Mantel_test_raw"))
        mantel_q_dist_eucl_all = sio.loadmat ( os.path.join (mantel_q_path[0],  "mantel_q_dist_eucl_{}".format(id_patient) ) )

        print('Beginning plotting mantel results')
        print('seizures:{}'.format(n_seizures))

        '''Seizure distances for IMFs'''
        fig_name = "Mantel_pvalues_dist_eucl_{}.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            '''Heatmap of Mantel test'''
            manteltest_pvalues = mantel_p_dist_eucl_all['pvalue'][0][0].copy()
            manteltest_p_tr = np.transpose(manteltest_pvalues)
            fig, ax = plt.subplots ()
            x_axis_labels = np.arange ( 1, manteltest_p_tr.shape[1] + 1, 1 )
            y_axis_labels = np.arange ( 1, manteltest_p_tr.shape[0] + 1, 1 )
            formatter = tkr.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            g = sns.heatmap (manteltest_p_tr, xticklabels=x_axis_labels,
                             yticklabels=y_axis_labels, annot = True, cbar_kws={"format": formatter})
            g.set_xlabel ( "p-value (Mantel test)" )
            g.set_ylabel ( "IMFs" )
            plt.title ( 'Mantel test p values')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

            fig_name = "Mantel_qvalues_dist_eucl_{}.{}".format (id_patient, "pdf" )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                '''Heatmap of Mantel test for H'''
                manteltest_q_tr = np.transpose(mantel_q_dist_eucl_all['qvalue'].copy())
                fig, ax = plt.subplots ()
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                x_axis_labels = np.arange ( 1, manteltest_q_tr.shape[1] + 1, 1 )
                y_axis_labels = np.arange ( 1, manteltest_q_tr.shape[0] + 1, 1 )
                g = sns.heatmap (manteltest_q_tr, xticklabels=x_axis_labels,
                                 yticklabels=y_axis_labels, cbar_kws={"format": formatter}, annot = True)
                g.set_xlabel ( "q-values (Mantel test after FDR)" )
                g.set_ylabel ( "IMFs" )
                plt.title ( 'Mantel test q values')

                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

            p_values = [0.05, 0.01, 0.001]

            fig_name = "Mantel_spearman_qvalues_dist_eucl_{}.{}".format ( id_patient, "pdf" )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                for pvalue in p_values:
                    '''Heatmap of Mantel test for IMFs'''
                    spearman_corr = np.transpose(mantel_p_dist_eucl_all['spearman_corr'][0][0])
                    manteltest_q_tr = np.transpose(mantel_q_dist_eucl_all['qvalue'])
                    masktest_sig = manteltest_q_tr > pvalue
                    fig, ax = plt.subplots ()
                    formatter = tkr.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    x_axis_labels = np.arange ( 1, manteltest_q_tr.shape[1] + 1, 1 )
                    y_axis_labels = np.arange ( 1, manteltest_q_tr.shape[0] + 1, 1 )
                    g = sns.heatmap (spearman_corr, xticklabels=x_axis_labels, vmin = -1, vmax = 1, center=0, cmap = "coolwarm",
                                     yticklabels=y_axis_labels, mask = masktest_sig, cbar_kws={"format": formatter})
                    g.set_xlabel ( "Spearman correlation" )
                    g.set_ylabel ( "Mantel test q values (after FDR)" )
                    plt.title ( 'Spearman q \n Grey squares depicts non-significance \n significance level: s{}'.format(pvalue ))

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
