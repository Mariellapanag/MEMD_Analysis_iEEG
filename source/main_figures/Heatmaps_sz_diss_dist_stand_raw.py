from pathlib import Path
import glob
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

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )


"""Heatmap of seizure distances standardised"""

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )


# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

# Test for one file
# in_path = files[0]

def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # Extract the patient's id
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
    n_seizures = len(info_seizure)

    format = "pdf"

    if (n_seizures > 5):

        '''MEMD RESULTS'''
        # Import the file with the final MEMD and STEMD results
        print ( "{}{}".format ( "Reading MEMD mat file ", id_patient ) )
        filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
        MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
        IMF_MEMD = MEMD_all["imf_memd"]
        [n_comp, n_imfs, n_time] = IMF_MEMD.shape

        '''Path of standardised seizure distances and dissimilarity matrix'''
        stand_path = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "sz_dist_stand_raw"))

        '''Reading standardised seizure dissimilarity matrix'''
        DissM_FC_stand = sio.loadmat ( os.path.join (stand_path[0], "DissMatFC_stand_{}.mat".format(id_patient) ) )['DissFC_stand']

        '''Reading all standardised seizure distances'''
        print('Reading Seizure Distances')
        seizures_dist_eucl_all_stand = sio.loadmat ( os.path.join (stand_path[0],  "seizure_dist_eucl_stand_{}".format(id_patient) ) )
        seizures_time_dist_stand = sio.loadmat ( os.path.join (stand_path[0],  "seizure_time_dist_stand_{}".format(id_patient) ) )['time_dist']

        x_axis_labels = np.arange ( 1, n_seizures + 1, 1 )
        y_axis_labels = np.arange ( 1, n_seizures + 1, 1 )

        print('Beginning plotting seizure dissimilarity')
        print('seizures:{}'.format(n_seizures))

        '''Plot standardised seizure dissimilarity matrix'''
        # Heatmap of Seizure Dissimilarity matrix
        fig, ax = plt.subplots ( figsize=(12, 8) )
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        g = sns.heatmap ( DissM_FC_stand, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                          rasterized = True, square  =True,  cbar_kws={"format": formatter})
        ax.set_xticklabels ( ax.get_xticklabels (), rotation=360 )
        g.set_xlabel ( "Seizure" )
        g.set_ylabel ( "Seizure" )
        g.set_title ( "Dissimilarity matrix of seizures \n Functional Connectivity" )
        plt.tight_layout()
        fig_name = "V1.seizure_dissFC_matrix_stand_{}.{}".format (id_patient, "pdf" )
        plt.savefig ( os.path.join ( out_subfolder, fig_name ), format=format )
        plt.close("all")

        # Heatmap of Seizure Dissimilarity matrix - revealing only the lower triangular values (grey squares to all others)
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

        ''' Plotting of time seizure distances'''
        print('Beginning plotting seizure distances')
        print('seizures:{}'.format(n_seizures))

        fig_name = "V1.seizure_time_dist_stand_{}.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:

            fig, ax = plt.subplots(figsize=(12,8))
            formatter = tkr.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            g = sns.heatmap(seizures_time_dist_stand,
                            xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                            rasterized = True, square = True,cbar_kws={"format": formatter})
            ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
            g.set_xlabel ( "Seizure" )
            g.set_ylabel ( "Seizure" )
            g.set_title ( 'Seizure time distance')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        # Heatmap of Seizure time distance matrix - revealing only the lower triangular values (grey squares to all others)
        fig_name = "V2.seizure_time_dist_stand_{}.{}".format (id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
                # Mask the upper triangular matrix
                mask = np.zeros_like ( seizures_time_dist_stand, dtype=np.bool )
                mask[np.triu_indices_from ( mask )] = True
                # Want diagonal elements as well
                mask[np.diag_indices_from ( mask )] = True

                fig, ax = plt.subplots(figsize=(12,8))

                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_time_dist_stand,
                                xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                rasterized = True, square = True, cbar_kws={"format": formatter}, mask=mask)
                ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                g.set_xlabel ( "Seizure" )
                g.set_ylabel ( "Seizure" )
                g.set_title ( 'Seizure time distance')

                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

        # Heatmap of Seizure IMF distances
        list_a = [seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)].ravel() for imf in range(0, n_imfs)]

        min = np.array(list_a).min()
        max = np.array(list_a).max()

        '''Plotting of Euclidean distance for IMFs across Dimensions'''
        # Heatmap of Seizure Distance matrix
        fig_name = "V1.FixedRange_seizure_dist_eucl_stand_{}.{}".format (id_patient, format )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for imf in range(0, n_imfs):

                fig, ax = plt.subplots(figsize=(12,8))
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
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
        fig_name = "V1.NonFixedRange_seizure_dist_eucl_stand_{}.{}".format (id_patient, format )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for imf in range(0, n_imfs):

                fig, ax = plt.subplots(figsize=(12,8))
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
                                , xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                rasterized = True,square = True, cbar_kws={"format": formatter}, vmin=seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)].min(),
                                vmax= seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)].max())
                ax.set_xticklabels ( ax.get_xticklabels (), rotation= 360)
                g.set_xlabel ( "Seizure" )
                g.set_ylabel ( "Seizure" )
                g.set_title ( 'Seizure distance IMF{}'.format(imf+1) )

                plt.tight_layout()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")

        # Heatmap of Seizure Distance matrix
        fig_name = "V2.FixedRange_seizure_dist_eucl_stand_{}.{}".format (id_patient, format )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for imf in range(0, n_imfs):

                # Mask the upper triangular matrix
                mask = np.zeros_like ( seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
                                       , dtype=np.bool )
                mask[np.triu_indices_from ( mask )] = True
                # Want diagonal elements as well
                mask[np.diag_indices_from ( mask )] = True

                fig, ax = plt.subplots(figsize=(12,8))
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
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

                # Heatmap of Seizure Distance matrix
        fig_name = "V2.NonFixedRange_seizure_dist_eucl_stand_{}.{}".format (id_patient, format )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            for imf in range(0, n_imfs):

                # Mask the upper triangular matrix
                mask = np.zeros_like ( seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
                                       , dtype=np.bool )
                mask[np.triu_indices_from ( mask )] = True
                # Want diagonal elements as well
                mask[np.diag_indices_from ( mask )] = True

                fig, ax = plt.subplots(figsize=(12,8))
                formatter = tkr.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                g = sns.heatmap(seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)]
                                , xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                                rasterized = True, square = True, cbar_kws={"format": formatter}, mask=mask,
                                vmin=seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)].min(),
                                vmax=seizures_dist_eucl_all_stand['IMF{}'.format(imf+1)].max())
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
    #files = files[5:6]

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
