from pathlib import Path
import glob
import scipy.io as sio
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.ticker as tkr
import seaborn as sns

from paths import ROOT_DIR
from funcs.Global_settings.global_settings_plots import *
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

"""   """

'''Define the input paths'''
# Path contains all the results from the analysis
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# '''Define the output path'''
# output_path = os.path.join ( "final_results" )
# folder = "figures"
#
# '''Define the output subfolder name'''
# subfolder = "Initial_data(No permutation)"
#
# out_subfolder_name = 'seizure_dist_NMF_Analysis'
#
# '''Choose run name from the following: 'initial', 'shuffle_total_random', 'shuffle_seizure_random', 'shuffle_seizure_slice' '''
# RUN_NAME = 'initial'
#
# selected_number = 50
#
# if RUN_NAME == 'initial':
#     sub_name = 'Initial_data'
#     n_permutations = 1
# elif RUN_NAME == 'shuffle_total_random':
#     sub_name = 'shuffle_randomly_{}'.format(selected_number)
#     n_permutations = selected_number
# elif RUN_NAME == 'shuffle_seizure_random':
#     sub_name = 'shuffle_seizure_times_{}'.format(selected_number)
#     n_permutations = selected_number
# elif RUN_NAME == 'shuffle_seizure_slice':
#     sub_name = 'shuffle_seizure_vector_{}'.format(selected_number)
#     n_permutations = selected_number
# else:
#     print('Please choose one of the available options for the parameter RUN_NAME')


# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
files = [glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*Mantel_test_raw")) for id_patient in folders]

def process_file ():

    pvalues = [0.001, 0.05, 0.01]

    pvalues_dict = {}
    eucl_pvalues_dict = {}
    for pvalue in pvalues:

        merge_list = []
        eucl_merge_list = []
        names = list()
        for file in files:
            # Extract path components
            parts = Path ( file ).parts

            id_patient = parts[-4]

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
                names.append(id_patient)
                '''Reading all Mantel q (after FDR)'''
                print('Reading all Mantel q')
                mantel_q_path = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*FDR_Mantel_test_raw"))
                mantel_q_dist_eucl_all = sio.loadmat ( os.path.join (mantel_q_path[0],  "mantel_q_dist_eucl_{}".format(id_patient) ) )['qvalue']

                '''Reading all Spearman Correlation'''
                print('Reading all Mantel p')
                mantel_p_path_list = glob.glob(os.path.join(ROOT_DIR, result_file, id_patient, "*Mantel_test_raw"))
                keywordFilter = ["FDR"]
                mantel_p_path = [sent for sent in mantel_p_path_list if not any(word in sent for word in keywordFilter)]
                spearman_eucl_dist = sio.loadmat ( os.path.join (mantel_p_path[0],  "mantel_p_dist_eucl_{}".format(id_patient) ) )['mantel_eucl_dist']['spearman_corr'][0][0]

                '''Dominant Frequency'''
                print("Reading dominant Frequency")
                dom_freq = sio.loadmat(os.path.join(ROOT_DIR, output_path,folder, id_patient, subfolder, "dominant_Psdfreq_allIMFs_{}.mat".format(id_patient)))['dom_freq'].squeeze()

                '''Seizure distance all IMFs and DIMs'''
                rows_imfs = np.where(mantel_q_dist_all <= pvalue)[0]

                imfs = ["IMF{}".format(i+1) for i in np.unique(rows_imfs)]

                ''' Read the Spearman correlation'''
                # Import the file with the Beta coefficients
                print ( "{} {}".format ( "Reading Spearman mat file", id_patient ) )
                spearman_subj = spearman_dist_all.copy()
                spearman_max = spearman_subj[np.unique(rows_imfs), :].max(axis=1)
                spearman_df = pd.DataFrame({"IMF": imfs, "spearman": spearman_max})

                '''Using Mantel test obtained from Distance of all IMFs and DIMs'''
                df_display = pd.DataFrame({"IMF": imfs})
                df_display['Patients'] = np.repeat(id_patient, df_display.shape[0])
                df_display["id"] = ["{}_{}".format ( a, b ) for a, b in
                                    zip ( df_display["Patients"], df_display["IMF"] )]

                df_display1 = df_display.merge ( spearman_df, on=[ "IMF"], how='left' )

                Patients = np.repeat(id_patient, dom_freq.shape[0])
                IMF = ["IMF{}".format(x) for x in range(1, dom_freq.shape[0]+1)]
                dom_freq_df = pd.DataFrame({"dom_freq": dom_freq, "Patients": Patients, "IMF": IMF})
                dom_freq_df["id"] = ["{}_{}".format ( a, b ) for a, b in
                                     zip ( dom_freq_df["Patients"], dom_freq_df["IMF"] )]

                merge_list.append(df_display1.merge ( dom_freq_df, on=["id", "Patients", "IMF"], how='left' ))

                '''Using Mantel test obtained from euclidean distance of IMFs across all Dimensions'''
                eucl_rows_imfs = np.where(mantel_q_dist_eucl_all <= pvalue)[1]
                eucl_spearman = spearman_eucl_dist[0][eucl_rows_imfs]
                eucl_imfs = ["IMF{}".format(i+1) for i in np.unique(eucl_rows_imfs)]
                eucl_spearman_df = pd.DataFrame({"IMF": eucl_imfs, "spearman": eucl_spearman})

                eucl_df_display = pd.DataFrame({"IMF": eucl_imfs})
                eucl_df_display['Patients'] = np.repeat(id_patient, eucl_df_display.shape[0])
                eucl_df_display["id"] = ["{}_{}".format ( a, b ) for a, b in
                                    zip ( eucl_df_display["Patients"], eucl_df_display["IMF"] )]

                eucl_df_display1 = eucl_df_display.merge ( eucl_spearman_df, on=[ "IMF"], how='left' )
                eucl_merge_list.append(eucl_df_display1.merge ( dom_freq_df, on=["id", "Patients", "IMF"], how='left' ))

                dfs = [df.set_index ( 'id' ) for df in merge_list]
                display_all = pd.concat ( dfs, axis=0 , sort = True)
                patient_names = np.unique(display_all['Patients'])
                add_names = [x for x in names if x not in patient_names]
                df_add = pd.DataFrame({"IMF": np.repeat("IMF", len(add_names)),
                                       "Patients": add_names, "dom_freq": np.repeat(np.nan, len(add_names)), "spearman": np.repeat(np.nan, len(add_names))})

                display_all = display_all.append(df_add)
                display_all["idx"] = [x.split("ID")[1] for x in display_all['Patients']]
                display_all.sort_values(by = ['idx'], inplace = True)

                pvalues_dict.__setitem__('pvalue{}'.format(pvalue), display_all)

                '''Euclidean distance '''
                eucl_dfs = [df.set_index ( 'id' ) for df in eucl_merge_list]
                eucl_display_all = pd.concat ( eucl_dfs, axis=0 , sort = True)
                eucl_patient_names = np.unique(eucl_display_all['Patients'])
                eucl_add_names = [x for x in names if x not in eucl_patient_names]
                eucl_df_add = pd.DataFrame({"IMF": np.repeat("IMF", len(eucl_add_names)),
                                       "Patients": eucl_add_names, "dom_freq": np.repeat(np.nan, len(eucl_add_names)), "spearman": np.repeat(np.nan, len(eucl_add_names))})

                eucl_display_all = eucl_display_all.append(eucl_df_add)
                eucl_display_all["idx"] = [x.split("ID")[1] for x in eucl_display_all['Patients']]
                eucl_display_all.sort_values(by = ['idx'], inplace = True)

                eucl_pvalues_dict.__setitem__('pvalue{}'.format(pvalue), eucl_display_all)

            fs_limit = 2880 / 2
            binsfreq = np.logspace ( -3,  np.log10 ( fs_limit ), 800, endpoint=True )
            fig_name = "Summary_Mantel_domFreq_dist_all.{}".format ("pdf" )
            with PdfPages(os.path.join(out_figure, fig_name)) as pages:
                for pvalue in pvalues:

                    fig, ax = plt.subplots ()
                    plt.scatter( x="Patients", y="dom_freq", c="spearman", cmap="RdBu",
                                  data = pvalues_dict['pvalue{}'.format(pvalue)], vmin = -1, vmax = 1)
                                 #norm=mpl.colors.LogNorm())
                    plt.xlabel('Patients')
                    plt.ylabel('Dominant Frequency')
                    plt.colorbar(label = "Spearman")
                    plt.yscale ( "log" )
                    plt.ylim(binsfreq.min(), binsfreq.max()+200)
                    plt.title("pvalue: {}".format(pvalue))
                    plt.tight_layout()
                    canvas = FigureCanvasPdf(fig)
                    canvas.print_figure(pages)
                    plt.close("all")


            fig_name = "Summary_Mantel_domFreq_dist_eucl.{}".format ( "pdf" )
            with PdfPages(os.path.join(out_figure, fig_name)) as pages:
                for pvalue in pvalues:

                    fig, ax = plt.subplots ()
                    plt.scatter( x="Patients", y="dom_freq", c="spearman", cmap="RdBu",
                                 data = eucl_pvalues_dict['pvalue{}'.format(pvalue)], vmin = -1, vmax = 1)
                    plt.xlabel('Patients')
                    plt.ylabel('Dominant Frequency')
                    plt.colorbar(label = "Spearman")
                    plt.title("pvalue: {}".format(pvalue))
                    plt.yscale ( "log" )
                    plt.ylim(binsfreq.min(), binsfreq.max()+200)

                    plt.tight_layout()
                    canvas = FigureCanvasPdf(fig)
                    canvas.print_figure(pages)
                    plt.close("all")


if __name__ == "__main__":
    process_file()