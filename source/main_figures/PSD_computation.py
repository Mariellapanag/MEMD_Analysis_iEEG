from pathlib import Path
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

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]
#folder = "PSD_computation"
#in_path = files[0]

def process_file(in_path):
    """Weighted power - frequency computation for all IMF*DIM
    """
    '''Extract path components'''
    parts = Path(in_path).parts
    id_patient = parts[-1]

    # Make output directory if needed
    out_subfolder = os.path.join (ROOT_DIR, result_file, id_patient, folder)
    os.makedirs(out_subfolder, exist_ok = True)
    print ( "Processing file:", in_path )

    '''Read Hilbert instantaneous frequency and amplitude'''
    print ( "{}{}".format ( "Read the instantaneous frequency, amplitude and phase ", id_patient ) )
    hilbert_path = os.path.join(ROOT_DIR, result_file, id_patient, "Hilbert_output")
    hilbert_output = sio.loadmat(os.path.join(hilbert_path, 'hilbert_output.mat'))
    n_imfs = hilbert_output['n_imfs'][0][0]
    n_comp = hilbert_output['n_comp'][0][0]

    ####################################################################
    frequency = hilbert_output['frequency']
    amplitude = hilbert_output['amplitude']
    time = hilbert_output['time'][0]
    '''Hilbert Huang Spectrum Computation'''
    binstime = time
    fs_limit = 2880 / 2
    binsfreq = np.logspace ( -3,  np.log10 ( fs_limit ), 800, endpoint=True )
    imf_bin_mean = np.zeros ( shape=(n_imfs, binstime[:-1].shape[0], binsfreq.shape[0]-1) )
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
        imf_bin_mean[imf, :, :] = np.nanmean( allComp_mean, axis=0 )
    id_nan = np.where(np.isnan(imf_bin_mean))
    imf_bin_mean[id_nan] = 0
    '''Marginal Density of each Comp within IMF'''
    id_nan = np.where(np.isnan(imf_bin_mean))
    imf_bin_mean[id_nan] = 0
    bin_cntrs_y = (y_edge[1:] + y_edge[:-1])/2
    """Investigate cases where the observations are low in bins
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
        plt.title("Count of observations within each time-frequency bin \n for each IMF")
        plt.xlabel('Count')
        plt.ylabel("Density")
    plt.tight_layout ()
    format = "pdf"
    name = "Density_Count_bins_IMFs_{}".format (id_patient )
    fig_name = "{}.{}".format ( name, format )
    plt.savefig ( os.path.join ( out_subfolder, fig_name ) )
    plt.close ( 'all' )
    ############# IMF power spectral density
    all_imfs = (np.nansum(imf_bin_mean, axis = 1))/time[:-1].shape
    indx = np.where(allImf_count<40)
    all_imfs[indx] = 0
    '''Plot imfs'''
    # colors = ["#0a3955","#174c6f", "#1b5b85", "#3b526d", "#466484", "#4e7399",
    #           "#68738c", "#7484a0", "#7c92b0", "#969cae", "#a1aabd", "#a8b3c9", "#c9cbd5", "#cfd2de",
    #           "#d4d8e4","#bfbdbb"]
    colors = []
    fig, ax = plt.subplots ( figsize=(6,4) )
    ax.set_facecolor('xkcd:white')
    for imf in range(0, n_imfs):
        # plot = plt.plot(bin_cntrs_y, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1),
        #                color = colors[imf] , linewidth = 1)
        plot = plt.plot(bin_cntrs_y, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1), linewidth = 1)
        colors.append(plot[0].get_color ())
    total = np.sum(all_imfs, axis=0)
    # total = np.sum(all_imfs[:-1,], axis=0)
    plt.plot(np.squeeze(bin_cntrs_y), np.squeeze(total),  label="SUM", alpha=0.8, color="black" )
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    plt.xlabel ( "Frequency (cycles/day)" )
    plt.ylabel ( "Power" )
    plt.xscale ( "log" )
    plt.yscale ( "log" )
    ax.legend (loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.xlim(0.1, 100)
    plt.xlim(0.001, bin_cntrs_y.max())
    plt.ylim ( 1e-07, 1e-03 )
    plt.tight_layout ()
    format = "pdf"
    name = "AllIMFs_psd_last{}".format (id_patient )
    fig_name = "{}.{}".format ( name, format )
    #fig.set_size_inches(66/25.4, 54/25.4)
    plt.savefig ( os.path.join ( out_subfolder, fig_name ), dpi = 100)
    plt.close ( 'all' )
    '''Plot all IMFs'''
    fig_name = "EachIMF_{}.{}".format (id_patient, "pdf" )
    with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
        i=0
        for imf in range(0, n_imfs):
            fig, ax = plt.subplots (figsize=(10, 8)  )
            plt.plot (bin_cntrs_y, all_imfs[imf,:], alpha=0.7, color = colors[i], label="IMF{}".format(imf+1) , rasterized = True)
            plt.xlabel ( "Frequency (cycles/day)" )
            plt.ylabel ( "Power" )
            plt.xscale ( "log" )
            plt.yscale ( "log" )
            # plt.xlim(0.1, 100)
            plt.ylim(0.001, bin_cntrs_y.max())
            plt.ylim ( 1e-07, 1e-03 )
            plt.legend ()
            plt.tight_layout ()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")
            i = i+1
    '''Plot imfs without last IMF'''
    # colors = ["#0a3955","#174c6f", "#1b5b85", "#3b526d", "#466484", "#4e7399",
    #           "#68738c", "#7484a0", "#7c92b0", "#969cae", "#a1aabd", "#a8b3c9", "#c9cbd5", "#cfd2de",
    #           "#d4d8e4","#bfbdbb"]
    colors = []
    fig, ax = plt.subplots ( figsize=(6,4) )
    ax.set_facecolor('xkcd:white')
    for imf in range(0, n_imfs-1):
        # plot = plt.plot(bin_cntrs_y, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1),
        #                color = colors[imf] , linewidth = 1)
        plot = plt.plot(bin_cntrs_y, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1), linewidth = 1)
        colors.append(plot[0].get_color ())
    total = np.sum(all_imfs[:-1,:], axis=0)
    # total = np.sum(all_imfs[:-1,], axis=0)
    plt.plot(np.squeeze(bin_cntrs_y), np.squeeze(total),  label="SUM", alpha=0.8, color="black" )
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    plt.xlabel ( "Frequency (cycles/day)" )
    plt.ylabel ( "Power" )
    plt.xscale ( "log" )
    plt.yscale ( "log" )
    ax.legend (loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.xlim(0.1, 100)
    plt.xlim(0.001, bin_cntrs_y.max())
    plt.ylim ( 1e-07, 1e-03 )
    plt.tight_layout ()
    format = "pdf"
    name = "AllIMFs_psd_wo_lastIMF{}".format (id_patient )
    fig_name = "{}.{}".format ( name, format )
    #fig.set_size_inches(66/25.4, 54/25.4)
    plt.savefig ( os.path.join ( out_subfolder, fig_name ), dpi = 100)
    plt.close ( 'all' )

    '''Plot imfs without last IMF - using cycle length'''

    # colors = ["#0a3955","#174c6f", "#1b5b85", "#3b526d", "#466484", "#4e7399",
    #           "#68738c", "#7484a0", "#7c92b0", "#969cae", "#a1aabd", "#a8b3c9", "#c9cbd5", "#cfd2de",
    #           "#d4d8e4","#bfbdbb"]
    colors = []
    fig, ax = plt.subplots ( figsize=(6,4) )
    ax.set_facecolor('xkcd:white')
    x = 1/bin_cntrs_y
    for imf in range(0, n_imfs-1):
        # plot = plt.plot(x, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1),
        #                 color = colors[imf] , linewidth = 1)
        plot = plt.plot(x, all_imfs[imf,:], alpha=0.7, label="IMF{}".format(imf+1), linewidth = 1)
        colors.append(plot[0].get_color ())
    total = np.sum(all_imfs[:-1,:], axis=0)
    # total = np.sum(all_imfs[:-1,], axis=0)
    plt.plot(1/np.squeeze(bin_cntrs_y), np.squeeze(total),  label="SUM", alpha=0.8, color="black" )
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    plt.xlabel ( "Cycle length" )
    plt.ylabel ( "Power" )
    plt.xscale ( "log" )
    plt.yscale ( "log" )
    ax.legend (loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.xlim(0.1, 100)
    plt.xlim(x.min(), 1000)
    plt.ylim ( 1e-07, 1e-03 )
    plt.tight_layout ()
    format = "pdf"
    name = "AllIMFs_psd_wo_lastIMF_cycle_length{}".format (id_patient )
    fig_name = "{}.{}".format ( name, format )
    #fig.set_size_inches(66/25.4, 54/25.4)
    plt.savefig ( os.path.join ( out_subfolder, fig_name ), dpi = 100)
    plt.close ( 'all' )

    '''Compute and save the dominant frequency based on the above calculation'''
    idfreq = np.argmax ( all_imfs, axis=1 )
    dominant_frequency = np.empty(n_imfs)
    lhh = np.empty(n_imfs)
    for imf in range(0, n_imfs):
        id = idfreq[imf]
        dominant_frequency[imf] = bin_cntrs_y[id]
        lhh[imf] = all_imfs[imf, id]
    '''Save the dominant frequency for all IMFs'''
    d_freq = {'dom_freq': dominant_frequency, "MarginalHH": lhh}
    sio.savemat ( os.path.join ( out_subfolder, "dominant_Psdfreq_allIMFs_{}.mat".format ( id_patient ) ), d_freq )
    '''Compute and save the total SUM for each patient'''
    total_psd = {'total_SUM': np.sum(all_imfs[1:-1, :], axis=0), "freqseq": binsfreq, "freq_edge": y_edge, "bin_cntrs_freq": bin_cntrs_y}
    sio.savemat ( os.path.join ( out_subfolder, "Total_SUM_IMFs_wo_1andLast{}.mat".format ( id_patient ) ), total_psd )
    total_psd1 = {'total_SUM': np.sum(all_imfs, axis=0), "freqseq": binsfreq, "freq_edge": y_edge, "bin_cntrs_freq": bin_cntrs_y}
    sio.savemat ( os.path.join ( out_subfolder, "Total_SUM_IMFs_{}.mat".format ( id_patient ) ), total_psd1 )
    total_psd2 = {'total_SUM': np.sum(all_imfs[0:-1,:], axis=0), "freqseq": binsfreq, "freq_edge": y_edge, "bin_cntrs_freq": bin_cntrs_y}
    sio.savemat ( os.path.join ( out_subfolder, "Total_SUM_IMFs_wo_LastIMF{}.mat".format ( id_patient ) ), total_psd2 )

    return True

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
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process()





