import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
import scipy.io as sio

from funcs.Global_settings.global_settings_plots import *
from paths import ROOT_DIR
from funcs.Global_settings.results import *

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
# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]

suffix_name = "eucldist"

folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
files = [os.path.join ( ROOT_DIR, input_path, f ) for f in folders]

out_figure = os.path.join ( ROOT_DIR, result_file, folder)
os.makedirs ( out_figure, exist_ok=True )

def process_file ():

    R_results_lm = pd.DataFrame ( columns=('patients', 'R_results', 'values') )

    for file in files:
        # Extract path components
        parts = Path ( file ).parts

        id_patient = parts[-1]

        '''Seizure information for each patient'''
        # Read the seizure information for the corresponding patient
        print('Reading Seizure Timings')
        seizures_file = sio.loadmat ( os.path.join (file,  "seizure_all_{}".format(id_patient) ) )
        #seizures_all = seizures_file["seizures"]
        n_seizures = seizures_file["n_seizures"][0][0]

        if (n_seizures > 5):

            in_results = os.path.join(ROOT_DIR, result_file, id_patient, "LinearRegression_IMFs")

            Rsquared_csv = pd.read_csv(os.path.join(in_results, "lm_results_{}_{}.csv".format(suffix_name, id_patient)))
            Rsquared_melt = Rsquared_csv.melt()
            Rsquared_melt['Patient'] = np.repeat(id_patient, 2)
            Rsquared_melt = Rsquared_melt[['Patient', 'variable', 'value' ]]
            Rsquared_melt.columns = ['patients', 'R_results', 'values']
            R_results_lm = pd.concat([R_results_lm, Rsquared_melt], axis = 0)

    R2_adj_df = R_results_lm.loc[R_results_lm.R_results == "adj_r_squared"]
    R2_adj_df.reset_index ( inplace=True )

    plt.figure()
    sns.barplot ( x="patients", y="values",
              data=R2_adj_df)
    plt.ylim([0,1])
    plt.ylabel("Adjusted R^2 values")

    format = 'pdf'
    fig_name = "{}.{}".format ( "Barplot_R_results_all_patients",  format )
    plt.tight_layout ()
    plt.savefig ( os.path.join ( out_figure, fig_name ) )
    plt.close ( 'all' )

if __name__ == "__main__":
    process_file()