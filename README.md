MEMD_Analysis_iEEG

## Table of contents
* [General info](#general-info)
* [Main Figures](#main-figures)
    + [Seizure Dissimilarity and Distance Heatmaps](##dist-heatmaps)
    + [Seizure Dissimilarity and Distance Heatmaps (standardised)](##dist-stand-heatmaps)
* [Setup](#setup)

## General info
This project is ........

## Setup
After downloading the project from Github, you can start producing results and figures.
Results will be stored in a folder results within the main directory folder of the project by default.
However, the user can change the name of the folder by going to the python file `results.py`
which is located in the following path:
`MEMD_Analysis_iEEG\funcs\Global_settings\results.py`
## Main Figures
Main Figures .......

### Seizure Dissimilarity and Distance Heatmaps
In order to generate the heatmap plots of seizure dissimilarity and seizure IMF distances,
 you need to run the following python scripts:
 - `sz_dist_raw.py`
 - `Heatmaps_sz_dist_raw.py`

### Seizure Dissimilarity and Distance Heatmaps (standardised)
In order to generate the heatmap plots of standardised seizure dissimilarity and standardised seizure IMF distances,
 you need to run the following python scripts:
 - `sz_dist_raw.py`
 - `sz_dist_stand_raw.py`
 - `Heatmaps_sz_dist_stand_raw.py`

## Setup
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```