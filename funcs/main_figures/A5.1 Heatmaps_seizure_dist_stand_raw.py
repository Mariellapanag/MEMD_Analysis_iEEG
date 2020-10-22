from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.ticker as tkr

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

    format = "pdf"

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

        '''Reading seizure dissimilarity matrix'''
        DissM_FC_stand = sio.loadmat ( os.path.join (out_subfolder,  "DissMatFC_stand_{}".format(id_patient) ) )['DissFC_stand']

        '''Reading all seizure distances'''
        print('Reading Seizure Distances')
        seizures_dist_eucl_all_stand = sio.loadmat ( os.path.join (out_subfolder,  "seizure_dist_eucl_stand_{}".format(id_patient) ) )
        seizures_time_dist_stand = sio.loadmat ( os.path.join (out_subfolder,  "seizure_time_dist_stand_{}".format(id_patient) ) )

        x_axis_labels = np.arange ( 1, n_seizures + 1, 1 )
        y_axis_labels = np.arange ( 1, n_seizures + 1, 1 )

        print('Beginning plotting seizure dissimilarity')
        print('seizures:{}'.format(n_seizures))

        '''Plot seizure dissimilarity matrix only in the "initial" folder'''
        if n_permutations == 1:
            # Heatmap of Seizure Dissimilarity matrix
            fig, ax = plt.subplots ( figsize=(12, 8) )
            formatter = tkr.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            g = sns.heatmap ( DissM_FC_stand, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                              rasterized = True,  cbar_kws={"format": formatter})
            ax.set_xticklabels ( ax.get_xticklabels (), rotation=360 )
            g.set_xlabel ( "Seizure" )
            g.set_ylabel ( "Seizure" )
            g.set_title ( "Dissimilarity matrix of seizures \n Functional Connectivity" )
            plt.tight_layout()
            fig_name = "V1.seizure_dissFC_matrix_stand_{}.{}".format (id_patient, "pdf" )
            plt.savefig ( os.path.join ( out_subfolder, fig_name ), format=format )
            plt.close("all")

            # Mask the upper triangular matrix
            mask = np.zeros_like ( DissM_FC_stand, dtype=np.bool )
            mask[np.triu_indices_from ( mask )] = True
            # Want diagonal elements as well
            mask[np.diag_indices_from ( mask )] = True

            # Heatmap of Seizure Dissimilarity matrix
            fig, ax = plt.subplots ( figsize=(12, 8) )
            formatter = tkr.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            g = sns.heatmap ( DissM_FC_stand, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                              rasterized = True, square = True,  cbar_kws={"format": formatter}, mask=mask)
            ax.set_xticklabels ( ax.get_xticklabels (), rotation=360 )
            g.set_xlabel ( "Seizure" )
            g.set_ylabel ( "Seizure" )
            g.set_title ( "Dissimilarity matrix of seizures \n Functional Connectivity" )
            plt.tight_layout()
            fig_name = "V2.seizure_dissFC_matrix_stand_{}.{}".format (id_patient, "pdf" )
            plt.savefig ( os.path.join ( out_subfolder, fig_name ), format=format )
            plt.close("all")

        print('Beginning plotting seizure distances')
        print('seizures:{}'.format(n_seizures))

        ''' Plotting of time seizure distances'''
        # Heatmap of Seizure Distance matrix

        fig_name = "V1.seizure_time_dist_stand_{}_Allperm.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for id_perm in range(0, n_permutations):

                fig, ax = plt.subplots(figsize=(12,8))
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_time_dist_stand['perm{}'.format(id_perm)],
                                xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                rasterized = True, square = True,cbar_kws={"format": formatter})
                ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                g.set_xlabel ( "Seizure" )
                g.set_ylabel ( "Seizure" )
                g.set_title ( 'Seizure time distance perm{}'.format(id_perm) )

                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

        fig_name = "V2.seizure_time_dist_stand_{}_Allperm.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for id_perm in range(0, n_permutations):

                # Mask the upper triangular matrix
                mask = np.zeros_like ( seizures_time_dist_stand['perm{}'.format(id_perm)], dtype=np.bool )
                mask[np.triu_indices_from ( mask )] = True
                # Want diagonal elements as well
                mask[np.diag_indices_from ( mask )] = True

                fig, ax = plt.subplots(figsize=(12,8))

                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_time_dist_stand['perm{}'.format(id_perm)],
                                xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                rasterized = True, square = True, cbar_kws={"format": formatter}, mask=mask)
                ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                g.set_xlabel ( "Seizure" )
                g.set_ylabel ( "Seizure" )
                g.set_title ( 'Seizure time distance perm{}'.format(id_perm) )

                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

        list_a = [seizures_dist_eucl_all_stand['perm{}'.format(id_perm)]['IMF{}'.format(imf+1)][0][0].ravel() for imf in range(0, n_imfs)]

        min = np.array(list_a).min()
        max = np.array(list_a).max()
        '''Plotting of Euclidean distance for IMFs across Dimensions'''
        # Heatmap of Seizure Distance matrix
        for id_perm in range(0, n_permutations):
            fig_name = "V1.seizure_dist_eucl_stand_{}_perm{}.{}".format (id_patient, id_perm, format )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                for imf in range(0, n_imfs):

                    fig, ax = plt.subplots(figsize=(12,8))
                    formatter = tkr.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    g = sns.heatmap(seizures_dist_eucl_all_stand['perm{}'.format(id_perm)]['IMF{}'.format(imf+1)][0][0]
                                    , xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                    rasterized = True,square = True, cbar_kws={"format": formatter}, vmin=min, vmax=max)
                    ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                    g.set_xlabel ( "Seizure" )
                    g.set_ylabel ( "Seizure" )
                    g.set_title ( 'Seizure distance IMF{}'.format(imf+1) )

                    plt.tight_layout()
                    canvas = FigureCanvasPdf(fig)
                    canvas.print_figure(pages)
                    plt.close("all")

        # Heatmap of Seizure Distance matrix
        for id_perm in range(0, n_permutations):
            fig_name = "V2.seizure_dist_eucl_stand_{}_perm{}.{}".format (id_patient, id_perm, format )
            with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                for imf in range(0, n_imfs):

                    # Mask the upper triangular matrix
                    mask = np.zeros_like ( seizures_dist_eucl_all_stand['perm{}'.format(id_perm)]['IMF{}'.format(imf+1)][0][0]
                                           , dtype=np.bool )
                    mask[np.triu_indices_from ( mask )] = True
                    # Want diagonal elements as well
                    mask[np.diag_indices_from ( mask )] = True

                    fig, ax = plt.subplots(figsize=(12,8))
                    formatter = tkr.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    g = sns.heatmap(seizures_dist_eucl_all_stand['perm{}'.format(id_perm)]['IMF{}'.format(imf+1)][0][0]
                                    , xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                    rasterized = True, square = True, cbar_kws={"format": formatter}, mask=mask, vmin=min, vmax=max)
                    ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                    g.set_xlabel ( "Seizure" )
                    g.set_ylabel ( "Seizure" )
                    g.set_title ( 'Seizure distance IMF{}'.format(imf+1) )

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
