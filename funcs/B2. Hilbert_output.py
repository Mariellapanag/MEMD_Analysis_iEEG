from pathlib import Path
import time
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import pandas as pd

from MEMD_funcs.FINAL_CODE.Hilbert_funcs import inst_features
from MEMD_funcs.Global_settings.main import ROOT_DIR


# Path specifying the input data of all patients
input_path = os.path.join(ROOT_DIR, "data", "longterm_preproc")

# Path for storing the results
folder_results = "final_results"

# subsubfolder
subsubfolder = "Initial_data(No permutation)"

"""Code for run and save Hilbert features 

This script allows the user to run the Hilbert function in order to obtain instantaneous frequency, 
amplitude and phase for a subject.

The code is made use of parallel processing in order to provide the aforementioned results for all patients.

    * The results are written as .mat file named hilbert_output.mat 
    * and it is stored in results_dir/ID**/data_proc
"""

# in_path = files[0]

def process_file(in_path):
    """
    Apply Hilbert transform is applied to the signal to extract the analytical form of the signal and then
    obtain the instantaneous frequency, amplitude and phase of the signal as a function of time.

    Parameters
    ----------
    in_path : os path
       Path of the input data.

    Returns
    -------
    hilbert_output : mat file
       Mat file that contains all the results for one subject.

    References
    ----------
    """

    #Define the output path
    folder = "figures"
    # folder = "figures_shuffled"

    # Extract path components
    parts = Path(in_path).parts
    id_patient = parts[-1]

    # Make output directory if needed
    out_subfolder = os.path.join(ROOT_DIR, folder_results, folder, id_patient,  subsubfolder)
    os.makedirs(out_subfolder, exist_ok = True)
    print ( "Processing file:", in_path )

    '''NMF RESULTS - PLOTS'''

    filename_nmf = "NMF_BP_CA_normedBand.mat"
    print ( "Processing file NMF:", filename_nmf)
    NMF_all = sio.loadmat(os.path.join(in_path, filename_nmf))
    H = NMF_all["H"]
    W = NMF_all["W"]
    n_comp, n_time = H.shape
    # Convert the H matrix into dataframe
    df_H = pd.DataFrame ( H.T, columns=['Comp' + str ( i ) for i in range ( 1, n_comp + 1)] )

    '''MEMD RESULTS - PLOTS'''
    # Import the file with the final MEMD and STEMD results
    print ( "{}{}".format ( "Reading MEMD mat file ", id_patient ) )
    filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand.mat"
    #filename_memd = "MEMDNSTEMD_NMF_BP_CA_normedBand_shuffled.mat"
    MEMD_all = sio.loadmat ( os.path.join(in_path, filename_memd ))
    IMF_MEMD = MEMD_all["imf_memd"]
    # IMF_MEMD = MEMD_all["imf_perm_memd"]

    df_melt = df_H.melt ()
    [n_comp, n_imfs, n_time] = IMF_MEMD.shape
    # Make a DataFrame with all the information about the IMFs and Components
    # This is for visualization purposes - It is convenient to use the function FacetGrid()
    df_imf = pd.DataFrame ()
    for imf in range ( 0, n_imfs ):
        for c in range ( 0, n_comp ):
            df_display = pd.DataFrame (
                {'imf_value': IMF_MEMD[c, imf, :], 'time': (np.arange ( 1, n_time + 1 ) * (30 / 3600))/24} )
            rows = df_display.shape[0]
            df_display["Imf"] = np.repeat ( "IMF" + str ( imf + 1 ), rows )
            df_display["Components"] = np.repeat ( "Comp" + str ( c + 1 ), rows )
            df_imf = df_imf.append ( df_display )
    df_imf.columns = ["value", "time", "Type", "Components"]

    '''HUANG - HILBERT SPECTRUM'''
    print ( "{}{}".format ( "Compute the instantaneous frequency, amplitude and phase ", id_patient ) )
    # Create the 3D matrices of instantaneous frequency, amplitude and phase in order to save the corresponding results
    frequency = np.zeros ( (n_comp, n_imfs, n_time - 1), dtype=float )
    amplitude = np.zeros ( (n_comp, n_imfs, n_time - 1), dtype=float )
    phase = np.zeros ( (n_comp, n_imfs, n_time - 1), dtype=float )
    phase_wrap = np.zeros ( (n_comp, n_imfs, n_time - 1), dtype=float )
    phase_angle = np.zeros ( (n_comp, n_imfs, n_time - 1), dtype=float )

    # Define the daily frequency in order to compute the instantaneous frequency, amplitude and phase
    fs = 2 * 60 * 24
    for i in range ( 0, n_imfs ):
        frequency[:, i, :] = np.array ( [inst_features ( IMF_MEMD[c, i, :], fs )[0] for c in range ( 0, n_comp )] )
        amplitude[:, i, :] = np.array ( [inst_features ( IMF_MEMD[c, i, :], fs )[1] for c in range ( 0, n_comp )] )
        phase[:, i, :] = np.array ( [inst_features ( IMF_MEMD[c, i, :], fs )[2] for c in range ( 0, n_comp )] )
        phase_wrap[:, i, :] = np.array ( [inst_features ( IMF_MEMD[c, i, :], fs )[3] for c in range ( 0, n_comp )] )
        phase_angle[:, i, :] = np.array ( [inst_features ( IMF_MEMD[c, i, :], fs )[4] for c in range ( 0, n_comp )] )
    # Compute the power of the signals
    power = np.power ( amplitude, 2 )
    # time in days
    time = (np.arange ( 1, n_time ) * 30 / 3600) / 24

    hilb_res = {'frequency': frequency, 'amplitude': amplitude, 'phase': phase, 'phase_wrap': phase_wrap,
                'phase_angle': phase_angle, 'power': power, 'time': time, 'n_comp': n_comp, 'n_imfs': n_imfs, 'n_time': n_time}
    sio.savemat(os.path.join(out_subfolder, "hilbert_output.mat"), hilb_res)

    # a = sio.loadmat(os.path.join(path_results, "hilbert_output.mat"))

def parallel_process(input_path):

    """
    The code is making use of parallel processing in order to provide the aforementioned results for all patients.

    Parameters
    ----------
    input_path : os path
        Path of the input data.
    """
    processed = 0

    folders = os.listdir(os.path.join(ROOT_DIR, input_path))
    files = [os.path.join(ROOT_DIR, input_path, folder) for folder in folders]

    # run the code for one or a selection of patients; just uncomment the following command and specify the index
    # that corresponds to the exact patient willing to run the code
    # files = files[15:16]

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
    '''Define the input paths'''
    # Path contains all the results from the analysis
    input_path = os.path.join ( "data", "longterm_preproc" )
    parallel_process(input_path)
