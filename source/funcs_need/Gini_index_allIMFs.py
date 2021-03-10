from pathlib import Path
import time
import os
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from funcs.Global_settings.global_settings_plots import *
from paths import ROOT_DIR
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

def Gini_index(array):

    '''
    [24/07 13:25] Mariella Panagiotopoulou (PGR)
    Computes the Gini Index of the vector X.
    The Gini Index is a measure of sparsity/inequality (0 = least sparse, 1 =
    most sparse).

    (Note: may not be properly formulated to handle negative elements)

    Mariella Panagiotopoulou
    24/07/2020
    '''

    # check for negative values
    if(np.sum(array<0) > 0):
        print('Check computation of Gini Index for distributions with negative values')

    # number of elements
    N = len(array)

    # sort X so elements are arranged from smallest to largest
    x = np.sort(array)

    # compute the L1 norm
    x_l1 = np.sum(np.abs(x), dtype=float)
    # gini_index = 0

    if(x_l1 == 0):
        gini_index = np.nan
        print('The sum of x must be greater than zero')
    else:
        gini_part = 0

        for k in np.arange(0,N):

            gini_part = gini_part + (x[k]/x_l1)*((N - (k+1) + 0.5)/N)

        gini_index = 1-(2*gini_part)

    return gini_index



# in_path = files[0]

def process_file(in_path):
    """Weighted power - frequency computation for all IMF*DIM
    """
    # Extract path components
    parts = Path(in_path).parts
    id_patient = parts[-1]

    # Make output directory if needed
    out_subfolder = os.path.join(ROOT_DIR, result_file, id_patient, folder)
    os.makedirs(out_subfolder, exist_ok = True)
    print ( "Processing file:", in_path )

    '''Frequency bands'''
    print ( "{}{}".format ( "Reading BPall_CA mat file ", input_path ) )
    BPall_CA_filename = "BPall_CA.mat"
    BPall_CA = sio.loadmat ( os.path.join ( in_path, BPall_CA_filename ) )['PA_all']

    [time_points, features, n_channels] = BPall_CA.shape

    '''NMF RESULTS - PLOTS'''
    #################################################################
    filename_nmf = "NMF_BP_CA_normedBand.mat"
    NMF_all = sio.loadmat ( os.path.join ( in_path, filename_nmf ) )
    # H = NMF_all["H"]
    W = NMF_all["W"]

    #####################################################################
    '''MEMD RESULTS - PLOTS'''
    ####################################################################
    # Import the file with the final MEMD and STEMD results
    print ( "{}{}".format ( "Reading MEMD mat file ", input_path ) )
    filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
    MEMD_all = sio.loadmat ( os.path.join ( in_path, filename_memd ) )
    IMF_MEMD = MEMD_all["imf_memd"]
    [n_comp, n_imfs, n_time] = IMF_MEMD.shape

    # Read Hilbert instantaneous frequency, amplitude and phase
    hilbert_path = os.path.join(ROOT_DIR, result_file, id_patient, "Hilbert_output")
    print ( "{}{}".format ( "Read the instantaneous frequency, amplitude and phase ", hilbert_path ) )

    hilbert_output = sio.loadmat(os.path.join(hilbert_path, 'hilbert_output.mat'))
    n_imfs = hilbert_output['n_imfs'][0][0]

    ####################################################################
    amplitude = hilbert_output['amplitude']

    '''READ DOMINANT FREQUENCY FOR PATIENT'''
    domFreq_path = os.path.join(ROOT_DIR, result_file, id_patient, "PSD_computation")
    print ( "{}{}".format ( "Read the Dominant frequency for all IMFs", domFreq_path ) )
    dom_freq_output = sio.loadmat ( os.path.join ( domFreq_path,
                                                   "dominant_Psdfreq_allIMFs_{}.mat".format ( id_patient )  ))
    dom_freq = dom_freq_output['dom_freq'].squeeze()
    dom_freq_df = pd.DataFrame({'dom_freq': dom_freq, "IMFs": ["IMF{}".format(i+1) for i in range(0, n_imfs)]})

    '''Gini index in W matrix'''
    # Comp1 = W[:,0]
    [rows, columns] = W.shape
    features_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    channels = ['chan' + str ( i ) for i in range ( 1, n_channels + 1 )]

    ''' Compute a measure of weight for all the contribution of all dimensions in each IMF'''
    weighted_power = np.zeros ( [n_imfs, n_comp] )
    for imf in range ( 0, n_imfs ):
        for comp in range ( 0, n_comp ):
            a = amplitude[comp, imf]
            n = a.shape[0]
            weighted_power[imf, comp] = ((a**2).sum()) / n

    weights_IMF = weighted_power/weighted_power.sum(axis=1)[:,None]
    weights_IMF_df = pd.DataFrame(weights_IMF, columns = ["DIM{}".format(i+1) for i in range(0, n_comp)])
    weights_IMF_df['IMF'] = ["{}".format(i+1) for i in range(0, n_imfs)]

    # Save the weights of all DIMs for all IMFs
    weights = {"IMF{}".format ( imf + 1 ): weights_IMF[imf,:] for imf in range ( 0, n_imfs )}
    sio.savemat ( os.path.join ( out_subfolder, "ALLWeights_allIMFs_weighted_Power{}.mat".format ( id_patient ) ), weights )

    '''Computation of W*weights'''
    W_weighted = pd.DataFrame(np.matmul(W, weights_IMF.T), columns = ["IMF" + str(i + 1) for i in range(0, n_imfs)])

    combined = []

    for pair in product(features_names, channels ):
        combined.append ( '_'.join ( pair ) )

    W_weighted["combined"] = combined
    new_col_list = ['features', 'channels']
    for n, col in enumerate ( new_col_list ):
        W_weighted[col] = W_weighted['combined'].apply ( lambda x: x.split("_")[n] )

    '''W*Weights Gini index of all channels in each frequency band for all IMFs'''

    W_weighted1 = W_weighted.drop(['combined', 'channels'], axis = 1)

    W_weighted_melt = pd.melt(W_weighted1, id_vars=['features'],
                              value_vars= ["IMF" + str(i + 1) for i in range(0, n_imfs)],
                              var_name = 'IMFs', value_name ='IMF_values')
    #a = W_weighted_melt.groupby(['IMFs', 'features']).apply(lambda x: gini(np.array(x))).reset_index()

    a = W_weighted_melt.groupby(['IMFs', 'features'])['IMF_values'].apply(lambda x: Gini_index(np.array(x))).reset_index()

    a['Patients'] = np.repeat(id_patient, a.shape[0])
    df_features = pd.merge(a, dom_freq_df, on = "IMFs")
    df_features.to_csv(os.path.join(out_subfolder, "df_features_combined_dom_freq_{}.csv".format(id_patient)))

def parallel_process():
    processed = 0

    folders = os.listdir(os.path.join(ROOT_DIR,input_path))
    files = [os.path.join(ROOT_DIR, input_path, folder) for folder in folders]

    # test the code
    # files = files[0:1]

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





