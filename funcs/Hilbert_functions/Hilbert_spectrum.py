import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl

from MEMD_funcs.Global_settings.global_settings_plots import *


def hilbert_spectrum(frequency, power, time, comp, imf, fig_name, out_path, format):
    """
    Hilbert spectrum plot

    Parameters
    ----------
    frequency : numpy array
       Instantaneous frequency
    power : numpy array
       Power of the analytical signal
    time : numpy array
        Time period that the signal occurs
    comp : int
        Index of the dimension of the IMF (MEMD)
    imf : int
        Index of the IMF (MEMD)
    fig_name : string
        The name of the figure
    out_path : os path
        The path that the figure will be saved
    format : format figure
        The format that the figure will be saved (pdf)

    Returns
    -------
    pdf file containing the plot of hilbert spectrum for the specified
    IMF and Dimension

    References
    ----------
    1.

    2.
    """

    plt.style.use ( selected_style )
    mpl.rcParams.update ( rc )
    fig, ax = plt.subplots ( ncols=3, figsize=(18, 8) )

    fig.subplots_adjust ( bottom=0.1, top=0.9, left=0.1, right=0.8,
                          wspace=0.1, hspace=0.04 )

    ax[0].scatter(time, frequency, c = power)
    ax[0].set_title ( "IMF{} Comp{}".format (imf + 1, comp + 1 ) )
    ax[0].set ( xlabel='Time', ylabel='Frequency' )

    plot1 = ax[1].tricontourf ( time, frequency, power )
    fig.colorbar ( plot1, ax=ax[1] )
    ax[1].set_title ( "IMF{} Comp{}".format ( imf + 1, comp + 1 ) )
    ax[1].set ( xlabel='Time', ylabel='Frequency' )

    plot2 = ax[2].tricontourf ( time, frequency, 10.0 * np.log10 ( power) )
    fig.colorbar(plot2, ax = ax[2])
    ax[2].set_title( "IMF{} Comp{} Power = 10*log(power)".format ( imf + 1, comp + 1 ) )
    ax[2].set ( xlabel='Time', ylabel='Frequency' )

    # Save figure
    plt.tight_layout ()
    fig_name = "{}.{}".format ( fig_name, format )
    plt.savefig ( os.path.join ( out_path, fig_name ), format=format )
    plt.close ( "all" )