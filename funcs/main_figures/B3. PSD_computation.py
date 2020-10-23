from pathlib import Path
import glob
import time
import os
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join("data", "longterm_preproc")
# Path contains the seizure information
info_path = os.path.join("data", "info")

# in_path = files[0]

def process_file(in_path):
    """Weighted power - frequency computation for all IMF*DIM
    """
    # Extract path components
    parts = Path(in_path).parts
    folder = "figures"
    subfolder = parts[-1]

    # Make output directory if needed
    out_subfolder = os.path.join(ROOT_DIR, folder_results, folder, subfolder, subsubfolder)
    os.makedirs(out_subfolder, exist_ok = True)
    print ( "Processing file:", in_path )

    path_results = os.path.join ( out_subfolder, "Hilbert_Huang" )
    os.makedirs ( path_results, exist_ok=True )


    #####################################################################
    '''MEMD RESULTS - PLOTS'''
    ####################################################################
    # Import the file with the final MEMD and STEMD results
    print ( "{}{}".format ( "Reading MEMD mat file ", subfolder ) )
    filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
    MEMD_all = sio.loadmat ( os.path.join ( in_path, filename_memd ) )
    IMF_MEMD = MEMD_all["imf_memd"]
    [n_comp, n_imfs, n_time] = IMF_MEMD.shape

    # The edges of the frequency bins computed using logscale
    freqbins = np.squeeze ( MEMD_all["freqbins"] )
    amplbins = np.squeeze(MEMD_all["amplbins"])
    # b = np.zeros(800)
    # plt.plot(freqbins, b)

    # Read Hilbert instantaneous frequency, amplitude and phase
    print ( "{}{}".format ( "Read the instantaneous frequency, amplitude and phase ", subfolder ) )
    hilbert_output = sio.loadmat(os.path.join(out_subfolder, 'hilbert_output.mat'))
    n_imfs = hilbert_output['n_imfs'][0][0]
    n_comp = hilbert_output['n_comp'][0][0]

    ####################################################################
    frequency = hilbert_output['frequency']
    amplitude = hilbert_output['amplitude']
    # phase = hilbert_output['phase']
    # phase_wrap = hilbert_output['phase_wrap']
    # phase_angle = hilbert_output['phase_angle']
    power = hilbert_output['power']
    time = hilbert_output['time'][0]

    '''Hilbert Huang Spectrum Computation'''
    binstime = time
    fs_limit = 2880 / 2
    binsfreq = np.logspace ( -3,  np.log10 ( fs_limit ), 800, endpoint=True )
    imf_bin_mean = np.zeros ( shape=(n_imfs, binstime[:-1].shape[0], binsfreq.shape[0]-1) )
    all_imf_comp = np.zeros(shape=(n_imfs, n_comp ,binstime[:-1].shape[0], binsfreq.shape[0]-1))
    for imf in range ( 0, n_imfs ):
        allComp_mean = np.zeros ( shape=(n_comp, binstime[:-1].shape[0], binsfreq.shape[0]-1) )
        for comp in range ( 0, n_comp ):
            values = amplitude[comp, imf, :]**2
            x = time
            y = frequency[comp, imf, :]
            statistic, x_edge, y_edge, binnumber = stats.binned_statistic_2d ( x, y, values, 'sum',
                                                                               bins=[binstime, binsfreq],
                                                                               expand_binnumbers=True )
            allComp_mean[comp, :, :] = statistic
            all_imf_comp[imf, comp, :, :] = statistic
        imf_bin_mean[imf, :, :] = np.nanmean( allComp_mean, axis=0 )/2

    '''Marginal Density of each Comp within IMF'''

    all_imfs_comps = (np.nansum(all_imf_comp, axis = 2))/(2*time[:-1].shape[0])
    # indx = np.where(all_imfs<10)
    # all_imfs_comps[indx] = 0

    bin_cntrs_y = (y_edge[1:] + y_edge[:-1])/2
    # ######Plot imfs
    format = "pdf"

    for imf in range(0, n_imfs):
        ############################################
        # Plot all IMFs
        fig_name = "Marginal_Hilbert_IMF{}_{}.{}".format (imf, subfolder, "pdf" )
        with PdfPages(os.path.join(path_results, fig_name)) as pages:
            for comp in range(0, n_comp):
                fig, ax = plt.subplots (figsize=(10, 8)  )
                plt.plot (bin_cntrs_y, all_imfs_comps[imf,comp, :], alpha=0.7, label="IMF{}".format(imf+1) , rasterized = True)
                plt.xlabel ( "Frequency (cycles/day)" )
                plt.title('Comp{}'.format(comp+1))
                plt.ylabel ( "Power" )
                plt.xscale ( "log" )
                plt.yscale ( "log" )
                # plt.xlim(0.1, 100)
                # plt.ylim(0.0001, 1)
                plt.ylim ( 1e-09, 1 )
                plt.legend ()
                plt.tight_layout ()
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)
                plt.close("all")


    """
    Hilbert Huang Transform as a 2D representation - PSD
    Plot multiple figures into a single PDF with matplotlib, using the
    object-oriented interface.
    """
    fig_name = "2D_HilbertPSD_{}.{}".format (subfolder, "pdf" )
    with PdfPages(os.path.join(path_results, fig_name)) as pages:
        for imf in range ( 0, n_imfs):
            X, Y = np.meshgrid ( x_edge[:-1], y_edge[:-1] )
            Z = imf_bin_mean[imf, :, :].T
            fig = plt.figure ()
            ax = fig.gca()
            #cmap = viridis
            #cmap  seismic
            plot = ax.pcolormesh ( X, Y, Z, cmap="seismic" , rasterized = True)
            fig.colorbar ( plot, ax=ax, format='%.1e', label = "Power")
            # Set label and title names
            ax.set_title ( "IMF{} \n Hilbert Spectrum".format ( imf + 1 ) )
            ax.set ( xlabel='Time (days)', ylabel='Frequency' )
            # ax.set_xscale ( 'log' )
            ax.set_yscale ( 'log' )
            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.contour3D(X, Y, Z, 50, cmap='viridis')
    """
    3D Hilbert Transform as a representation of instantaneous frequency, amplitude and time
    """
    # for imf in range ( 0, n_imfs ):
    #     fig_name = "3D_Hilbert_Inst_IMF{}_{}.{}".format (imf+1, subfolder, "pdf" )
    #     with PdfPages(os.path.join(path_results, fig_name)) as pages:
    #         for comp in range(0, n_comp):
    #             # Give a representation across Components
    #             x = time
    #             y = frequency[comp, imf, :]
    #             z = power[comp, imf, :]
    #
    #             # 3d Hilbert Spectrum
    #             fig = plt.figure (constrained_layout=True)
    #             ax = Axes3D ( fig )
    #             surf = ax.plot_trisurf ( x, y, z, cmap=cm.jet, linewidth=0.1 )
    #             fig.colorbar ( surf, shrink=0.5, aspect=5 )
    #             ax.set_xlabel ( 'Time (days)' )
    #             ax.set_ylabel ( 'Frequency (cycles/day)' )
    #             ax.set_zlabel ( 'Power' )
    #             ax.set_title('COMP{}'.format(comp+1))
    #             #plt.tight_layout()
    #             canvas = FigureCanvasPdf(fig)
    #             canvas.print_figure(pages)
    #             plt.close("all")


    """
    Investigate cases where the observations are low in bins
    and set a threshold
    """
    binstime = time
    fs_limit = 2880 / 2
    binsfreq = np.logspace ( -3,  np.log10 ( fs_limit ), 800, endpoint=True )
    allImf_count = np.zeros ( shape=(n_imfs, binsfreq.shape[0]-1) )
    fig = plt.figure()
    for imf in range ( 0, n_imfs ):
        allComp_count = np.zeros ( shape=(n_comp, binstime[:-1].shape[0], binsfreq.shape[0]-1) )
        for comp in range ( 0, n_comp ):
            values = amplitude[comp, imf, :]**2
            x = time
            y = frequency[comp, imf, :]
            statistic, x_edge, y_edge, binnumber = stats.binned_statistic_2d ( x, y, values, 'count',
                                                                               bins=[binstime, binsfreq],
                                                                               expand_binnumbers=True )

            allComp_count[comp, :, :] = statistic

        allImf_temp = np.nansum(allComp_count, axis = 1)
        allImf_count[imf, :] = np.nansum(allImf_temp, axis = 0)

        sns.kdeplot(allImf_count.ravel(), shade=True, label = "IMF{}".format(imf+1))
        plt.xlabel('Count')
        plt.ylabel("Density")

    plt.tight_layout ()
    format = "pdf"
    name = "Density_Count_bins_IMFs_{}".format (subfolder )
    fig_name = "{}.{}".format ( name, format )
    plt.savefig ( os.path.join ( path_results, fig_name ) )
    plt.close ( 'all' )


    ############# IMF power spectral density
    all_imfs = (np.nansum(imf_bin_mean, axis = 1))/time[:-1].shape
    indx = np.where(allImf_count<40)
    all_imfs[indx] = 0

    bin_cntrs_y = (y_edge[1:] + y_edge[:-1])/2
    # ######Plot imfs
    colors = []
    fig, ax = plt.subplots ( figsize=(10, 8) )

    for imf in range(1, n_imfs-1):

        plot = ax.plot(bin_cntrs_y, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1) )
        colors.append(plot[0].get_color ())
        ax.legend ( loc='center left', bbox_to_anchor=(1.1, 0.5) )
    total = np.nansum(all_imfs[1:-1, :], axis=0)
    plt.plot(bin_cntrs_y, total,  label="SUM", alpha=0.8, color="black" )
    plt.xlabel ( "Frequency (cycles/day)" )
    plt.ylabel ( "Power" )
    plt.xscale ( "log" )
    plt.yscale ( "log" )
    # plt.xlim(0.1, 100)
    # plt.ylim(0.0001, 1)
    plt.ylim ( 1e-09, 1 )
    plt.legend ()
    plt.tight_layout ()

    format = "pdf"
    name = "AllIMFs_psd{}".format (subfolder )
    fig_name = "{}.{}".format ( name, format )
    plt.savefig ( os.path.join ( path_results, fig_name ) )
    plt.close ( 'all' )

    ############################################
    # Plot all IMFs
    fig_name = "EachIMF_{}.{}".format (subfolder, "pdf" )
    with PdfPages(os.path.join(path_results, fig_name)) as pages:
        i=0
        for imf in range(1, n_imfs-1):
            fig, ax = plt.subplots (figsize=(10, 8)  )
            plt.plot (bin_cntrs_y, all_imfs[imf,:], alpha=0.7, color = colors[i], label="IMF{}".format(imf+1) , rasterized = True)
            plt.xlabel ( "Frequency (cycles/day)" )
            plt.ylabel ( "Power" )
            plt.xscale ( "log" )
            plt.yscale ( "log" )
            # plt.xlim(0.1, 100)
            # plt.ylim(0.0001, 1)
            plt.ylim ( 1e-09, 1 )
            plt.legend ()
            plt.tight_layout ()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")
            i = i+1


    ################################################################
    # Compute and save the dominant frequency based on the above calculation
    idfreq = np.nanargmax ( all_imfs, axis=1 )
    dominant_frequency = np.zeros(n_imfs)
    for imf in range(0, n_imfs):
        id = idfreq[imf]
        dominant_frequency[imf] = bin_cntrs_y[id]

    # Save the dominant frequency for all IMFs
    d_freq = {'dom_freq': dominant_frequency}
    sio.savemat ( os.path.join ( out_subfolder, "dominant_Psdfreq_allIMFs_{}.mat".format ( subfolder ) ), d_freq )

    #################################################################
    # Compute and save the total SUM for each patient
    total_psd = {'total_SUM': np.nansum(all_imfs[1:-1, :], axis=0), "freqseq": binsfreq, "freq_edge": y_edge, "bin_cntrs_freq": bin_cntrs_y}
    sio.savemat ( os.path.join ( out_subfolder, "Total_SUM_IMFs_wo_1andLast{}.mat".format ( subfolder ) ), total_psd )

    total_psd1 = {'total_SUM': np.nansum(all_imfs, axis=0), "freqseq": binsfreq, "freq_edge": y_edge, "bin_cntrs_freq": bin_cntrs_y}
    sio.savemat ( os.path.join ( out_subfolder, "Total_SUM_IMFs_{}.mat".format ( subfolder ) ), total_psd1 )


def parallel_process():
    processed = 0

    folders = os.listdir(os.path.join(ROOT_DIR,input_path))
    files = [os.path.join(ROOT_DIR, input_path, folder) for folder in folders]

    # test the code
    # files = files[5:6]

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
    parallel_process()





