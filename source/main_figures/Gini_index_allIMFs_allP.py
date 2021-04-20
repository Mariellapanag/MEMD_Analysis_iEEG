from pathlib import Path
import scipy.io as sio
from matplotlib.colors import LogNorm
import math
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from funcs.Global_settings.global_settings_plots import *
from paths import ROOT_DIR
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

# Define the input paths
input_path = os.path.join ( "data", "longterm_preproc" )
# Path contains the seizure information
info_path = os.path.join ( "data", "info" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

id_patients = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
files = [os.path.join ( ROOT_DIR, result_file, id, "Gini_index_allIMFs") for id in id_patients]

out_figure = os.path.join ( ROOT_DIR, result_file, folder)
os.makedirs ( out_figure, exist_ok=True )

def summarise_allP():
    '''
    fgfgfgggf

    :return:
    '''

    # Store Gini index across IMFs for every frequency band
    features_list = []

    for i in files:
        parts = Path (i).parts
        id_patient = parts[-2]

        # Final data after permuted - count
        df_features = pd.read_csv(os.path.join ( i, "df_features_combined_dom_freq_{}.csv".format(id_patient) ), index_col = 0)
        features_list.append(df_features)

    features_df_allP = pd.concat(features_list)
    features_df_allP.reset_index(inplace=True)

    features_df_allP['ind'] = pd.to_numeric(features_df_allP.Patients.str.split('ID', n=1, expand = True)[1])
    features_df_allP.sort_values(by = ["ind"], inplace = True)
    features = features_df_allP.features.unique()

    '''Violin plots'''
    '''Gini Index for each Frequency band - different bin selection for dominant frequency'''
    sns.set_style("whitegrid")
    bins_new = np.logspace(-3,3.11,12)
    bin_labels = []
    for i in range(0, len(bins_new)-1):
        bin_labels.append("($10^{{{:.1f}}}$, $10^{{{:.1f}}}$]".format(np.log10(bins_new[i]), np.log10(bins_new[i + 1])))

    features_df_allP['dom_freq_bins'] = pd.cut(features_df_allP['dom_freq'], bins=bins_new)

    # Create an array with the colors you want to use
    colors = ["#eb8a65", "#388bbd", "#988ec4", "#f6b3b9", "#fbc15e", "#c9984b", "#78504b",
              "#99a246", "#49af72", "#4aaca5", "#edcc9e", "#54accc", "#b7a8d2", "#d691bf", "#e24a33", "#97b6d8",
              "#72688e", "#777777"]

    fig_name = "2.VIOLIN_GiniIndex_dom_freq_frequencies_allP.{}".format ( "pdf" )
    with PdfPages(os.path.join(out_figure, fig_name)) as pages:
        for feature in features:
            plt.rcParams.update({'font.size': 8})
            # Set your custom color display as palette
            sns.set_palette(sns.color_palette(colors))
            data_display = features_df_allP[features_df_allP['features'] == feature]
            fig, ax = plt.subplots (figsize=(10, 8)  )
            ax = sns.violinplot(x="dom_freq_bins", y="IMF_values", data=data_display, cut = 0, color = "lightgray",
                                linewidth = 0.5, alpha = 0.1)
            #sns.stripplot(x="dom_freq_bins", y="IMF_values", data=data_display, hue ="Patients" , alpha = 1.0, size = 7)
            sns.swarmplot(x="dom_freq_bins", y="IMF_values", data=data_display, hue ="Patients" , alpha = 0.8, size = 8)
            if feature == "Delta":
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
            else:
                ax.legend(ncol=9,
                          loc='upper center',
                          bbox_to_anchor=(0.5, 1.0),
                          bbox_transform=plt.gcf().transFigure)
            # Set the x labels
            ax.set_xticklabels(bin_labels)
            ax.set(ylim = (0,0.3), yticks = [0.0, 0.05,0.10,0.15,0.20,0.25,0.30])
            plt.xticks(rotation=20)
            plt.title(feature)
            plt.ylabel('Gini index')
            plt.xlabel('IMF Dominant Frequency (cycles/day)', fontsize = 8)
            plt.tight_layout ()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")



if __name__ == "__main__":
    summarise_allP()

