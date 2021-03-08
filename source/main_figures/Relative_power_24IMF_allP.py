from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import glob

from funcs.Global_settings.global_settings_plots import *
from paths import ROOT_DIR
from funcs.Global_settings.results import *

plt.style.use ( selected_style )
mpl.rcParams.update ( rc )

# Define the input paths
input_path = os.path.join ( "data", "longterm_preproc" )

# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
files = [os.path.join ( ROOT_DIR, result_file, subject, "Relative_power_24IMF") for subject in folders]

out_figure = os.path.join ( ROOT_DIR, result_file, folder)
os.makedirs ( out_figure, exist_ok=True )

def summarise_imfs_allP():
    '''
    fgfgfgggf

    :return:
    '''
    files_l24 = []
    for i in files:
        id_patient = Path(i).parts[-2]
        os.chdir(i)
        df = pd.read_csv(glob.glob("Final_weights_per_freqBand_*_24c_{}.csv".format(id_patient))[0])
        files_l24.append(df)

    files_all = {"files_l24": files_l24}
    names = ["files_l24"]
    for i in range(0,1):
        print(i)
        files_l = files_all[names[i]]
        name_fig = names[i].split("files_l")[1]
        df_display = pd.concat (files_l, axis=0 )
        df_display = df_display[["features", "values", "Patients"]]
        order = pd.DataFrame({"id": [1,2,3,4,5], "features": ["Delta", "Theta", "Alpha", "Beta", "Gamma"]})
        df_wide = df_display.pivot(index='features', columns='Patients', values='values')
        df_wide.reset_index(inplace = True)
        df_merged = pd.merge(df_wide, order, on = "features")
        df_merged.sort_values ( by=['id'], inplace = True )
        df_merged.drop(['id'], axis = 1, inplace = True)
        df_merged.set_index('features', inplace = True)
        #Make percentages per patient
        df_merged_perc = df_merged.apply ( lambda x: (x / (x.sum ()))*100 )

        format = "pdf"


        fig, ax = plt.subplots ()
        g = sns.heatmap ( df_merged_perc , annot = True)
        for t in ax.texts: t.set_text ( t.get_text () + " %" )
        g.set_xlabel ( "Patients" )
        g.set_ylabel ( "Frequency Bands" )
        plt.title ( 'All Patients - % per patient' )

        # Save figure
        plt.tight_layout ()
        fig_name = "Allperc_annot_Summary_W_weights_per_freqBand_{}.{}".format (name_fig, format )
        plt.savefig ( os.path.join ( out_figure, fig_name ), format=format )
        plt.close ( "all" )

        fig, ax = plt.subplots ()
        g = sns.heatmap ( df_merged_perc)
        for t in ax.texts: t.set_text ( t.get_text () + " %" )
        g.set_xlabel ( "Patients" )
        g.set_ylabel ( "Frequency Bands" )
        plt.title ( 'All Patients - % per patient' )

        # Save figure
        plt.tight_layout ()
        fig_name = "Allperc_Summary_W_weights_per_freqBand_{}.{}".format (name_fig, format )
        plt.savefig ( os.path.join ( out_figure, fig_name ), format=format )
        plt.close ( "all" )
        print("The code was executed")

if __name__ == "__main__":
    summarise_imfs_allP ()