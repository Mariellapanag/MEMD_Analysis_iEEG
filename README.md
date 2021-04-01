MEMD_Analysis_iEEG

# Table of contents
* [General info](#general-info)
* [Setup](#setup)
    * [Choose the name of the folder _results_](#results-folder)
    * [Run for one patient or all at once](#choice-of-run)
* [Main Figures](#main-figures)
    * [Seizure Dissimilarity and Distance Heatmaps](#dist-heatmaps)
    * [Seizure Dissimilarity and Distance Heatmaps (standardised)](#dist-stand-heatmaps)
    * [Scatterplots of seizure dissimilarity with seizure distances](#scatter-szdiss-szdist)
    * [Marginal Hilbert Spectrum of IMFs](#psd-computation)
    * [Contribution of iEEG main frequency bands to the 24h IMF](#rel-power)
    * [Gini index of IMFs for every frequency band across all subjects](#gini-index)
* [Supporting figures](#other-figures)
    * [Mantel test figures](#mantel-test-fig)
    * [Summary plot combining Mantel test and Dominant Frequency](#mantel-dom-freq)


# <a name="general-info"></a> General info
This project is ........

# <a name="setup"></a> Setup
## <a name="results-folder"></a> Choose the name of the folder _results_
After downloading the project from Github, you can start producing results and figures.
Results will be stored in a folder results within the main directory folder of the project by default.
However, the user can change the name of the folder by going to the python file `results.py`
which is located in the following path:
`MEMD_Analysis_iEEG\funcs\Global_settings\results.py`
The output of each python script by default will be saved in a folder named as 
the python script file running within the _**results**_ folder.

## <a name="choice-of-run"></a> Run for one patient or all at once
In most of the python files used by the user, there is the choice of generating the results for 1 or more patients or
for all patients using parallel programming.
The exact lines of code appear in each python file the user can run are the following:

```python
def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

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

```
Using the above code chunk the user can generate results for all patients using parallel programming.

For running the chosen python scripts for **one patient**, for example **patient ID06**, user needs
 to add the following line of code:
 
```python
files = files[5:6]
```
The final code chunk would look like this:

```python
def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

    files = files[5:6]

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

```
For running the chosen python scripts for **more than one subject**, for example **subjects ID02, ID03, ID04, ID05**, user needs
 to add the following line of code:
 
```python
files = [files[i] for i in [1,2,3,4]]
```
The final code chunk would look like this:

```python
def parallel_process ():
    processed = 0

    folders = os.listdir ( os.path.join ( ROOT_DIR, input_path ) )
    files = [os.path.join ( ROOT_DIR, input_path, folder ) for folder in folders]

    files = [files[i] for i in [1,2,3,4]]

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

```
The above code chunk is not included in the following python scripts:
 - `FDR_Mantel_test_raw.py`
 
# <a name="main-figures"></a> Main Figures

Main Figures .......

## <a name="dist-heatmaps"></a> Seizure Dissimilarity and Distance Heatmaps

In order to generate the heatmap plots of seizure dissimilarity and seizure IMF distances,
 you need to run the following python scripts:
 - `sz_diss_dist_raw.py`
 - `Heatmaps_sz_diss_dist_raw.py`

## <a name="dist-stand-heatmaps"></a> Seizure Dissimilarity and Distance Heatmaps (standardised)

In order to generate the heatmap plots of standardised seizure dissimilarity and standardised seizure IMF distances,
 you need to run the following python scripts:
 - `sz_dist_raw.py`
 - `sz_diss_dist_stand_raw.py`
 - `Heatmaps_sz_diss_dist_stand_raw.py`

## <a name="scatter-szdiss-szdist"></a> Scatterplots of seizure dissimilarity and seizure distances

In order to obtain the scatterplots between seizure dissimilarity and seizure IMF distance, 
as well as the ones of seizure dissimilarity and seizure time distance, you will have to run the following python scripts:
 - `sz_dist_raw.py`
 - `Mantel_test_raw.py`

The output of `Mantel_test_raw.py` would be the aforementioned scatterplots (one for the time distance and multiple ones for all IMFs), along with
the spearman correlation values displayed in the title of the plots.
Mantel test results are also generated, but not displayed in the scatterplots, as they will be used later on, in order to perform FDR to all patients.

## <a name="psd-computation"></a> Marginal Hilbert Spectrum of IMFs
For obtaining the graphical representations, the following python scripts need to be run:
 
 - `Hilbert_output.py`
 - `PSD_computation.py`

## <a name="rel-power"></a>  Contribution of iEEG main frequency bands to the 24h IMF
For obtaining the graphical representations, the following python scripts need to be run:

 - `Hilbert_output.py`
 - `PSD_computation.py`
 - `Relative_power_24IMF.py`
 - `Relative_power_24IMF_allP.py`

## <a name="gini-index"></a> Gini index of IMFs for every frequency band across all subjects
For obtaining the graphical representation, the following python scripts need to be run:

 - `Hilbert_output.py`
 - `PSD_computation.py`
 - `Gini_index_allIMFs.py`
 - `Gini_index_allIMFs_allP.py`

# <a name="other-figures"></a> Supporting figures
Figures .....
add brief description of the figures included

## <a name="mantel-test-fig"></a> Mantel test figures
For generating figures for the Mantel test p-values, as well as the resulting q-values after applying the FDR correction, 
the user should run the following python scripts:
 - `sz_dist_raw.py`
 - `Mantel_test_raw.py`
 - `FDR_Mantel_test_raw.py`
 - `Heatmap_Mantel_test_raw.py`
Note that the first 3 python scripts (`sz_dist_raw.py`, `Mantel_test_raw.py`, `FDR_Mantel_test_raw.py`) should executed for all subjects; the code automatically generate the results 
for subjects with #seizures > 5.

 - ``
 
