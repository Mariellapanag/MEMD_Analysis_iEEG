import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn import linear_model
import itertools
import statsmodels.api as sm
import math
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.ticker as tkr
import matplotlib as mpl
from pathlib import Path
import sklearn.metrics as metrics

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
output_path = os.path.join ( "final_results" )
folder = "figures"

'''Define the output path'''
# Get the name of the current script
folder = os.path.basename(__file__) # This will be used to specify the name of the file that the output will be stored in the file results
folder = folder.split(".py")[0]


def calc_train_error (X_train, y_train, model):
    '''returns in-sample error for already fit model'''
    predictions = model.predict ( X_train )
    mse = mean_squared_error ( y_train, predictions )
    #rmse = np.sqrt ( mse )
    return mse

def calc_validation_error (X_test, y_test, model):
    '''returns out-of-sample error for already fit model'''
    predictions = model.predict ( X_test )
    mse = mean_squared_error ( y_test, predictions )
    #rmse = np.sqrt ( mse )
    return mse


def calc_metrics (X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit ( X_train, y_train )
    train_error = calc_train_error ( X_train, y_train, model )
    validation_error = calc_validation_error ( X_test, y_test, model )
    return train_error, validation_error

# def fit_linear_reg_sklearn(X,Y, n, n_features):
#     #Fit linear regression model and return RSS and R squared values
#     model_k = linear_model.LinearRegression(fit_intercept = True)
#     model_k.fit(X,Y)
#     RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
#     MSE = mean_squared_error(Y, model_k.predict(X))
#     BIC = n * math.log ( MSE ) + n_features * math.log ( n )
#     R_squared = model_k.score(X,Y)
#     return RSS, R_squared, MSE, BIC

def regression_results (y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score ( y_true, y_pred )
    mean_absolute_error = metrics.mean_absolute_error ( y_true, y_pred )
    mse = metrics.mean_squared_error ( y_true, y_pred )
    #mean_squared_log_error = metrics.mean_squared_log_error ( y_true, y_pred )
    median_absolute_error = metrics.median_absolute_error ( y_true, y_pred )
    r2 = metrics.r2_score ( y_true, y_pred )

    print ( 'explained_variance: ', round ( explained_variance, 4 ) )
    #print ( 'mean_squared_log_error: ', round ( mean_squared_log_error, 4 ) )
    print ( 'r2: ', round ( r2, 4 ) )
    print ( 'MAE: ', round ( mean_absolute_error, 4 ) )
    print ( 'MSE: ', round ( mse, 4 ) )
    print ( 'RMSE: ', round ( np.sqrt ( mse ), 4 ) )
    return r2, mean_absolute_error, mse, explained_variance

def fit_linear_reg_statsmodel(X, Y):

    #Fit linear regression model and return RSS and R squared values
    X = sm.add_constant ( X )  # adding a constant
    # Fit regression model
    reg_k = sm.OLS ( Y, X, hasconst=True ).fit ()
    # Inspect the results
    print ( reg_k.summary () )
    # Residuals sum of squares
    RSS = reg_k.mse_resid * len(Y)
    # Mean squared error of residuals
    MSE = reg_k.mse_resid
    # Bic Criterion
    BIC = reg_k.bic
    # R squared of the model
    R_squared = reg_k.rsquared
    return RSS, R_squared, MSE, BIC


name_character = "eucldist"

# in_path = files[0]

def process_file (in_path):

    # Extract path components
    parts = Path ( in_path ).parts

    # folder = "figures_shuffled"
    id_patient = parts[-1]

    # Ouput directory for data-results
    out_subfolder = os.path.join (ROOT_DIR, result_file, id_patient, folder)
    os.makedirs ( out_subfolder, exist_ok=True )
    print ( "Processing file:", in_path )

    '''Seizure information for each patient'''
    # Read the seizure information for the corresponding patient
    print('Reading Seizure Timings')
    seizures_file = sio.loadmat ( os.path.join (in_path,  "seizure_all_{}".format(id_patient) ) )
    #seizures_all = seizures_file["seizures"]
    n_seizures = seizures_file["n_seizures"][0][0]

    if (n_seizures > 5):

        inpath = os.path.join(ROOT_DIR, result_file, id_patient, "Preparing_data_for_LASSO")

        CV_dict = {}
        coefs_dict = {}
        lasso_coefs_dict = {}
        data_all_dict = {}

        '''Read DATA'''
        # Import the file with the Seizure Dissimilarity results
        print ( "{} {}".format ( "Reading CSV Dataset", id_patient ) )
        data = pd.read_csv ( os.path.join ( inpath, 'data_for_Modelling_{}.csv'.format(name_character) ) )
        data_stand = pd.read_csv(os.path.join ( inpath, 'data_standardise_for_Modelling_{}.csv'.format(name_character) ) )
        data.rename ( columns={'time_dist': 'time.dist'}, inplace = True)
        data_stand.rename ( columns={'time_dist': 'time.dist'}, inplace=True )

        X_df = data_stand.drop ( columns='sz_diss_FC', axis=1 ).copy ()
        X = np.array ( X_df )
        y_df = data_stand['sz_diss_FC'].copy ()
        y = np.array ( y_df )

        tuned_parameters = np.logspace(-3, 2, 100)
        k_folds = 10

        kf = KFold ( n_splits=k_folds, shuffle=True, random_state=42 )

        train_error_per_tuning = []
        validation_error_per_tuning = []
        std_train_errors = []
        std_validation_errors = []

        for alpha in tuned_parameters:
            train_errors = []
            validation_errors = []
            for train_index, val_index in kf.split ( X, y ):
                # split data
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Perform Lasso Regularisation to the data
                lasso = Lasso( alpha=alpha, fit_intercept=True, normalize=False, random_state=77, max_iter=50000, positive = True)

                # calculate errors
                train_error, val_error = calc_metrics ( X_train, y_train, X_val, y_val, lasso )

                # calculate std of errors
                # append to appropriate list
                train_errors.append ( train_error )
                validation_errors.append ( val_error )

            std_train_errors.append (np.sqrt(np.var(train_errors)/len(train_errors)))
            std_validation_errors.append(np.sqrt(np.var(validation_errors)/len(validation_errors)))
            # generate report
            print ( 'alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
                    format ( alpha,
                             round ( np.mean ( train_errors ), 4 ),
                             round ( np.mean ( validation_errors ), 4 ) ) )

            train_error_per_tuning.append(round ( np.mean ( train_errors ), 4 ))
            validation_error_per_tuning.append(round ( np.mean ( validation_errors ), 4 ) )

        best_alpha = tuned_parameters[validation_error_per_tuning.index ( min ( validation_error_per_tuning ) )]

        CV_dict.__setitem__('noperm', {'tuned_parameters': tuned_parameters, 'train_error_per_tuning':train_error_per_tuning, 'validation_error_per_tuning':validation_error_per_tuning,
                                                       'std_train_errors':std_train_errors, 'std_validation_errors':std_validation_errors,
                                                       'best_alpha': best_alpha})
        coefs = []
        for alpha in tuned_parameters:
            # Perform Lasso Regularisation to the data
            lasso = Lasso ( alpha=alpha, fit_intercept=True, normalize=False, random_state=77, max_iter=50000, positive = True )
            lasso.fit ( X, y )
            coefs.append ( lasso.coef_ )

        coefs_dict.__setitem__('noperm', {'coefs': coefs})

        # Extract the optimal value of tuning parameter based on MSE
        id_opt = validation_error_per_tuning.index(min(validation_error_per_tuning))
        lasso.set_params(alpha = tuned_parameters[id_opt])
        model = lasso.fit(X,y)
        lasso_coefs = lasso.coef_
        y_pred = model.predict(X)
        y_real = y.copy()

        is_all_zero = np.all((lasso_coefs == 0))
        k = 0
        while (is_all_zero==True) & (k<=id_opt):
            id_opt_new = id_opt - k
            lasso.set_params(alpha = tuned_parameters[id_opt_new])
            model = lasso.fit(X,y)
            lasso_coefs = lasso.coef_
            is_all_zero = np.all((lasso_coefs == 0))
            k = k + 1
            if is_all_zero!=True:
                id_opt = id_opt_new
                break

        CV_dict['noperm'].update({'accepted_alpha':tuned_parameters[id_opt]})
        y_pred = model.predict(X)
        y_real = y.copy()

        coefs_df = pd.DataFrame({"variables": X_df.columns.values, "coefs": lasso_coefs})
        coefs_df = coefs_df[coefs_df.coefs != 0]
        coefs_df.to_csv(os.path.join(out_subfolder, "Constrained_lasso_coefficients_{}{}.csv".format(name_character, id_patient)))

        colours = np.zeros(lasso_coefs.shape[0])
        for ind in range(0, lasso_coefs.shape[0] ):
            if lasso_coefs[ind] !=0:
                colours[ind] = 3
            else:
                colours[ind] = 4

        len_labels = len(X_df.columns.values)

        lasso_coefs_dict.__setitem__('noperm', {'len_labels': len_labels, 'X_df.columns.values': X_df.columns.values,
                                                                'lasso_coefs': lasso_coefs, 'colours': colours, 'X_df': X_df})

        # Collect all remaining variables from Lasso
        names_variables_remain = coefs_df[coefs_df['coefs']!=0]["variables"]
        X_selection = X_df[names_variables_remain]
        y_selection = y_df.copy()

        final_data_stand = X_selection.copy()
        final_data_stand['y'] = y_selection
        final_data_stand.to_csv(os.path.join(out_subfolder, 'data_stand_from_constrained_lasso_{}{}.csv'.format(name_character, id_patient)), index=False)

        X_sel  = data[names_variables_remain]
        y_sel = data["sz_diss_FC"]
        data_fin = X_sel.copy()
        data_fin["y"] = y_sel
        data_fin.to_csv(os.path.join(out_subfolder, 'data_from_constrained_lasso_{}{}.csv'.format(name_character, id_patient)), index=False)

        data_all_dict.__setitem__('noperm',  {'final_data_stand': final_data_stand, 'data_fin': data_fin})

        fig_name = "Cross_Validation_Scores_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            # Plot of CV-MSE validation error against the tuning parameters
            fig, ax = plt.subplots(1,2)
            axes = ax.flatten()
            axes[0].plot(CV_dict['noperm']['tuned_parameters']
                         , CV_dict['noperm']['validation_error_per_tuning'], c = 'b')
            axes[0].set_xscale('log')
            axes[0].set_xlabel('Tuning parameter')
            axes[0].set_ylabel('MSE Validation CVerror')
            axes[0].axvline(CV_dict['noperm']['best_alpha'], c = "black")
            axes[0].set_title("noperm")

            axes[1].plot ( CV_dict['noperm']['tuned_parameters'],
                           CV_dict['noperm']['train_error_per_tuning'] , c= 'r', alpha = 0.5)
            axes[1].set_xscale ( 'log' )
            axes[1].set_xlabel ( 'Tuning parameter' )
            axes[1].set_ylabel ( 'MSE Training CVerror' )
            axes[1].axvline(CV_dict['noperm']['best_alpha'], c = "black")
            axes[1].set_title('noperm')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        # Plot of CV-MSE validation error along with error bars against the tuning parameters
        fig_name = "1Cross_Validation_Scores_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            fig, ax = plt.subplots ( 1, 2 )
            axes = ax.flatten ()
            axes[0].plot(CV_dict['noperm']['tuned_parameters'],
                         CV_dict['noperm']['validation_error_per_tuning'],
                         'or', c = 'b', markersize = 2)
            axes[0].plot ( CV_dict['noperm']['tuned_parameters']
                           , CV_dict['noperm']['validation_error_per_tuning'],
                           '-', color='gray', alpha = 1.2, linewidth=1.5 )
            axes[0].fill_between ( CV_dict['noperm']['tuned_parameters'],
                                   np.array(CV_dict['noperm']['validation_error_per_tuning']) -
                                   np.array(CV_dict['noperm']['std_validation_errors']),
                                   np.array(CV_dict['noperm']['validation_error_per_tuning']) +
                                   np.array(CV_dict['noperm']['std_validation_errors']),
                                   color='darkgray', alpha=0.5 )
            # axes[0].errorbar ( tuned_parameters, validation_error_per_tuning , std_validation_errors, ecolor = 'r', elinewidth = 2, capsize = 2)
            axes[0].axvline(CV_dict['noperm']['best_alpha'], dash_joinstyle = 'round', linewidth = 1,
                            color = 'r').set_linestyle('--')
            axes[0].text ( CV_dict['noperm']['best_alpha'] + 0.4*CV_dict['noperm']['best_alpha'],
                           np.median(CV_dict['noperm']['validation_error_per_tuning']) +
                           1.5*np.std(CV_dict['noperm']['validation_error_per_tuning']), "alpha: %0.4f" % CV_dict['noperm']['best_alpha'],
                           rotation=90, verticalalignment='center' , color = 'r')
            axes[0].set_xscale ( 'log' )
            axes[0].set_xlabel ( 'Tuning parameter' )
            axes[0].set_ylabel ( 'MSE Validation CVerror' )
            axes[0].set_title('noperm')

            axes[1].plot ( CV_dict['noperm']['tuned_parameters'],
                           CV_dict['noperm']['train_error_per_tuning'],
                           'or', c='r', markersize=2 , alpha = 0.5)
            axes[1].plot ( CV_dict['noperm']['tuned_parameters'],
                           CV_dict['noperm']['train_error_per_tuning'],
                           '-', color='gray', alpha=1.2, linewidth=1.5 )
            axes[1].fill_between ( CV_dict['noperm']['tuned_parameters'],
                                   np.array (  CV_dict['noperm']['train_error_per_tuning'] )
                                   - np.array ( CV_dict['noperm']['std_train_errors'] ),
                                   np.array (  CV_dict['noperm']['train_error_per_tuning'] )
                                   + np.array ( CV_dict['noperm']['std_train_errors'] ),
                                   color='darkgray', alpha=0.5 )
            # axes[1].errorbar ( tuned_parameters, train_error_per_tuning , std_train_errors, ecolor = 'r', elinewidth = 2, capsize = 2)
            axes[1].set_xscale ( 'log' )
            axes[1].set_xlabel ( 'Tuning parameter' )
            axes[1].set_ylabel ( 'MSE Training CVerror' )
            axes[1].set_title('noperm')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        '''Cross-validation scores plots combined together'''
        fig_name = "Combined_Cross_Validation_Scores_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            fig, ax = plt.subplots ()
            plt.plot(CV_dict['noperm']['tuned_parameters'],
                     CV_dict['noperm']['validation_error_per_tuning'],
                     'or', c = 'b', markersize = 3, label = "MSE Validation CVerror")
            plt.plot ( CV_dict['noperm']['tuned_parameters']
                       , CV_dict['noperm']['validation_error_per_tuning'],
                       '-', color='gray', alpha = 1.2, linewidth=1.5)
            plt.fill_between ( CV_dict['noperm']['tuned_parameters'],
                               np.array(CV_dict['noperm']['validation_error_per_tuning']) -
                               np.array(CV_dict['noperm']['std_validation_errors']),
                               np.array(CV_dict['noperm']['validation_error_per_tuning']) +
                               np.array(CV_dict['noperm']['std_validation_errors']),
                               color='silver', alpha=0.5 )
            # axes[0].errorbar ( tuned_parameters, validation_error_per_tuning , std_validation_errors, ecolor = 'r', elinewidth = 2, capsize = 2)
            plt.axvline(CV_dict['noperm']['best_alpha'], dash_joinstyle = 'round', linewidth = 1,
                        color = 'r').set_linestyle('--')
            plt.text ( CV_dict['noperm']['best_alpha'] + 0.4*CV_dict['noperm']['best_alpha'],
                       np.median(CV_dict['noperm']['validation_error_per_tuning']) +
                       1.5*np.std(CV_dict['noperm']['validation_error_per_tuning']), "alpha: %0.4f" % CV_dict['noperm']['best_alpha'],
                       rotation=90, verticalalignment='center' , color = 'r')

            plt.plot ( CV_dict['noperm']['tuned_parameters'],
                       CV_dict['noperm']['train_error_per_tuning'],
                       'or', c='goldenrod', markersize=3 , label = 'MSE Training CVerror')
            plt.plot ( CV_dict['noperm']['tuned_parameters'],
                       CV_dict['noperm']['train_error_per_tuning'],
                       '-', color='gray', alpha=1.2, linewidth=1.5 )
            plt.fill_between ( CV_dict['noperm']['tuned_parameters'],
                               np.array (  CV_dict['noperm']['train_error_per_tuning'] )
                               - np.array ( CV_dict['noperm']['std_train_errors'] ),
                               np.array (  CV_dict['noperm']['train_error_per_tuning'] )
                               + np.array ( CV_dict['noperm']['std_train_errors'] ),
                               color='silver', alpha=0.5 )
            # axes[1].errorbar ( tuned_parameters, train_error_per_tuning , std_train_errors, ecolor = 'r', elinewidth = 2, capsize = 2)
            plt.legend()
            plt.xscale ( 'log' )
            plt.xlabel ( 'Tuning parameter' )
            plt.ylabel ( 'MSE CVerror' )
            plt.title('noperm')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        fig_name = "Effect_tuning_parameter_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            fig, ax = plt.subplots()
            plt.plot(CV_dict['noperm']['tuned_parameters'], coefs_dict['noperm']['coefs'])
            plt.axvline ( CV_dict['noperm']['best_alpha'], dash_joinstyle='round', linewidth=1, color='r' ).set_linestyle ( '--' )
            plt.text ( CV_dict['noperm']['best_alpha'] + 0.1 * CV_dict['noperm']['best_alpha'],
                       np.mean (  coefs_dict['noperm']['coefs'] )
                       + 2.5 * np.std (  coefs_dict['noperm']['coefs'] ),
                       "alpha: %0.4f" % CV_dict['noperm']['best_alpha'], rotation=90, verticalalignment='center', color='r' )

            plt.xscale('log')
            plt.xlabel('Tuning parameter')
            plt.ylabel('Coefficients')
            plt.title('noperm')

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        fig_name = "Scatterplot_Lasso_Coefficients_values_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            fig = plt.figure(figsize=(lasso_coefs_dict['noperm']['len_labels'] * 2, 10))
            plt.scatter(lasso_coefs_dict['noperm']['X_df.columns.values'],
                        lasso_coefs_dict['noperm']['lasso_coefs'], c = lasso_coefs_dict['noperm']['colours'],
                        s=100)
            # set parameters for tick labels
            plt.tick_params ( axis='x', which='major', labelsize=9,
                              labelrotation=90, zorder=2, labelleft=True )

            plt.xticks ( lasso_coefs_dict['noperm']['X_df.columns.values'], linespacing=5 )
            plt.xlabel('Explanatory Variables')
            plt.ylabel('Coefficients')
            plt.title('samples: {}, features: {}\n remaining features: {}'.format(lasso_coefs_dict['noperm']['X_df'].shape[0],
                                                                                         lasso_coefs_dict['noperm']['X_df'].shape[1],
                                                                                         np.count_nonzero(lasso_coefs_dict['noperm']['lasso_coefs'])))

            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")

        fig_name = "Pairwise_Correlation_{}_{}_Allperm.{}".format (name_character, id_patient, "pdf" )
        with PdfPages(os.path.join(out_subfolder, fig_name)) as pages:
            fig, ax = plt.subplots ( 2,1)
            axes = ax.flatten ()
            # plot the heatmap
            corr1 = data_all_dict['noperm']['final_data_stand'].corr()
            sns.heatmap ( corr1,
                          xticklabels=corr1.columns,
                          yticklabels=corr1.columns, ax = axes[0] )
            axes[0].set_title("Stand data Pairwise correlation")

            # plot the heatmap
            corr2 = data_all_dict['noperm']['data_fin'].corr()
            sns.heatmap ( corr2,
                          xticklabels=corr2.columns,
                          yticklabels=corr2.columns )
            axes[1].set_title ( "data Pairwise correlation")
            plt.tight_layout()
            canvas = FigureCanvasPdf(fig)
            canvas.print_figure(pages)
            plt.close("all")
    return True

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
            if future.result() == True:
                processed += 1
                print ( "Processed {}files.".format ( processed, len ( files ) ), end="\r" )

    end_time = time.time ()
    print ( "Processed {} files in {:.2f} seconds.".format ( processed, end_time - start_time ) )


if __name__ == "__main__":
    parallel_process ()