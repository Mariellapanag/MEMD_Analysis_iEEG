from pathlib import Path
import time
import os
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
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

# in_path = files[0]

def process_file(in_path):
    """Weighted power - frequency computation for all IMF*DIM
    """
    # Extract path components
    # Extract path components
    parts = Path(in_path).parts
    id_patient = parts[-1]

    # Make output directory if needed
    out_subfolder = os.path.join (ROOT_DIR, result_file, id_patient, folder)
    os.makedirs(out_subfolder, exist_ok = True)
    print ( "Processing file:", in_path )

    '''Frequency bands'''
    print ( "{}{}".format ( "Reading BPall_CA mat file ", in_path ) )
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
    print ( "{}{}".format ( "Reading MEMD mat file ", in_path ) )
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
    dom_freq = dom_freq_output['dom_freq'].ravel()
    Mhh = dom_freq_output["MarginalHH"].ravel()

    ''' Compute a measure of weight for the distance of all dominant frequencies from 1'''
    dist24 = abs(dom_freq.squeeze() - 1)
    indices = np.where(dist24 <= dist24.min())
    if len(indices[0]) == 1:
        index_24c_IMF = indices[0][0]
    else:
        MHH = Mhh[indices]
        indx = np.argmax(MHH)
        index_24c_IMF = indices[0][indx]

    weighted_power = np.zeros ( [n_imfs, n_comp] )
    for imf in range ( 0, n_imfs ):
        for comp in range ( 0, n_comp ):
            a = amplitude[comp, imf]
            n = a.shape[0]
            weighted_power[imf, comp] = ((a**2).sum()) / n

    format = "pdf"

    weights_IMF = weighted_power/weighted_power.sum(axis=1)[:,None]
    weights_IMF_df = pd.DataFrame(weights_IMF*100, columns = ["DIM{}".format(i+1) for i in range(0, n_comp)])
    weights_IMF_df['IMF'] = ["{}".format(i+1) for i in range(0, n_imfs)]
    fig = plt.figure()
    weights_IMF_df.plot(x = "IMF", kind = "bar", stacked = True)
    plt.legend ( loc='center left', bbox_to_anchor=(1.0, 0.5) )
    plt.ylabel("Weighted Power (%)")
    plt.xlabel("IMF")
    plt.title("Relative power for all components \n within each IMF (%)")
    plt.tight_layout ()
    name_fig = "Relative_Power_allIMF_barplot_{}.{}".format ( id_patient, format )
    plt.savefig ( os.path.join ( out_subfolder, name_fig ) )
    plt.close ( 'all' )

    '''The same plot as above but using a heatmap'''
    fig = plt.figure ()
    y_axis_labels = ["IMF{}".format(i) for i in np.arange(1, n_imfs+1,1)]
    x_axis_labels = ["DIM{}".format ( i ) for i in np.arange ( 1, n_comp + 1, 1 )]
    sns.heatmap(weights_IMF*100, yticklabels=y_axis_labels, xticklabels=x_axis_labels, cmap='Blues', annot = True)
    plt.title("Relative power for all components \n within each IMF (%)")

    plt.tight_layout ()
    name_fig = "Relative_Power_allIMF_heatmap_{}.{}".format ( id_patient, format )
    plt.savefig ( os.path.join ( out_subfolder, name_fig ) )
    plt.close ( 'all' )

    ##################################################################################
    [rows, columns] = W.shape
    features_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    channels = ['chan' + str ( i ) for i in range ( 1, n_channels + 1 )]
    W_matrix = pd.DataFrame ( W, columns= ["Comp" + str(i + 1) for i in range(0, columns)] )

    combined = []

    for pair in product(features_names, channels ):
        combined.append ( '_'.join ( pair ) )

    W_matrix["combined"] = combined
    new_col_list = ['features', 'channels']
    for n, col in enumerate ( new_col_list ):
        W_matrix[col] = W_matrix['combined'].apply ( lambda x: x.split("_")[n] )
    # Plot - sum of W weights per frequency within each component
    d1_group = W_matrix.groupby(['features']).sum().reset_index()
    d1 = pd.DataFrame({'features': features_names, 'index':np.arange(1,len(features_names)+1)})
    d1_df = pd.merge(d1, d1_group)
    y_names = d1_df["features"]
    d1_df = d1_df.drop(['features'], axis=1)
    d1_df = d1_df.set_index("index")
    d1_df = np.asmatrix(d1_df)
    x_names = range(1, d1_df.shape[1]+1)
    # Plot of column vectors of the sum of weights within each frequency band in each component
    fig, ax = plt.subplots (figsize=[10, 8] )
    sns.heatmap( d1_df, yticklabels=y_names, xticklabels=x_names)
    plt.xlabel ( "Components", fontsize=16 )
    plt.ylabel ( 'Frequency bands', fontsize=16 )
    plt.xticks ( fontsize=14 )
    plt.title ( 'Sum of component weights', fontsize=19 )
    plt.tight_layout ()
    name_fig = "Sum_of_components_weights_IMF{}.{}".format ( imf+1, format )
    plt.savefig ( os.path.join ( out_subfolder, name_fig ) )
    plt.close ( 'all' )

    for imf in range ( 0, n_imfs ):
        if imf == index_24c_IMF:
            idx_name = "24c"

            # Multiply every column-component with the weights corresponding to each component
            W_weighted = pd.DataFrame( W*weights_IMF[imf, :], columns= ["Comp" + str(i + 1) for i in range(0, columns)] )

            combined = []

            for pair in product(features_names, channels ):
                combined.append ( '_'.join ( pair ) )

            W_weighted["combined"] = combined
            new_col_list = ['features', 'channels']
            for n, col in enumerate ( new_col_list ):
                W_weighted[col] = W_weighted['combined'].apply ( lambda x: x.split("_")[n] )

            # Total sum of each row
            # W_weighted['sum'] = W_weighted.sum(axis=1)
            d1_group = W_weighted.groupby(['features']).sum().reset_index()
            d1 = pd.DataFrame({'features': features_names, 'index':np.arange(1,len(features_names)+1)})
            d1_df = pd.merge(d1, d1_group)
            y_names = d1_df["features"]
            d1_df = d1_df.drop(['features'], axis=1)
            d1_df = d1_df.set_index("index")
            d1_df = np.asmatrix(d1_df)
            x_names = range(1, d1_df.shape[1]+1)

            d1_group.to_csv(os.path.join(out_subfolder, "Frequency_bands_weights_Weighted_power_IMF{}_{}_{}.csv".format(imf+1,idx_name,id_patient)))

            df_disp = pd.DataFrame(d1_group.sum(axis = 1))
            df_disp["features"] = d1_group["features"]
            d = pd.DataFrame({'features': features_names, 'index':np.arange(1,len(features_names)+1)})
            df_disp2 = pd.merge(df_disp, d)
            df_disp3 = df_disp2.sort_values(by = ["index"])
            df_disp3.rename(columns={0: "values"}, inplace=True)

            df_save = df_disp3[['values','features', 'index']]
            df_save['Patients'] = np.repeat(id_patient, df_save.shape[0])
            df_save.to_csv(os.path.join(out_subfolder, "Final_weights_per_freqBand_IMF{}_{}_{}.csv".format(imf+1,idx_name,id_patient)))

            df_disp3 = df_disp3[['features','values']]

            df_disp4 = df_disp3.copy()
            df_disp4['values%'] = df_disp4['values'].div(df_disp4['values'].sum(), axis=0).multiply(100)

            fig, ax = plt.subplots (figsize=[8, 10] )
            df_disp4.plot ( kind='bar', x='features', y='values%', legend=None, rot=360 )
            locs, labels = plt.yticks()
            yticks_new = [str(int(i)) +"%" for i in locs]
            plt.yticks ( locs, yticks_new,  fontsize=14 )
            # for c in fig.axes[-1].texts: print(c.get_text())
            #     # c.set_text(c.get_text() + " %" )
            plt.xlabel ( "Frequency bands", fontsize=16 )
            plt.ylabel ( 'W * weighted power', fontsize=16 )
            plt.xticks ( fontsize=14 )
            plt.title ( 'Across all components', fontsize=19 )
            # plt.axis("auto")

            plt.tight_layout ()
            name_fig = "Frequency_bands_weights_Weighted_power_perc_IMF{}_{}.{}".format ( imf+1, idx_name, format )
            plt.savefig ( os.path.join ( out_subfolder, name_fig ) )
            plt.close ( 'all' )
            break

def parallel_process():
    processed = 0

    folders = os.listdir(os.path.join(ROOT_DIR,input_path))
    files = [os.path.join(ROOT_DIR, input_path, folder) for folder in folders]
    # test the code
    # files = files[5:6]
    start_time = time.time ()
    # Test to make sure concurrent map is working
    with ProcessPoolExecutor ( max_workers=2 ) as executor:
        futures = [executor.submit ( process_file, in_path ) for in_path in files]
        for future in as_completed ( futures ):
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process()





