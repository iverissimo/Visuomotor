
# useful functions to use in other scripts

import os
import hedfpy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def edf2h5(edf_files, hdf_file, pupil_hp = 0.01, pupil_lp = 6.0):
    
    """ convert edf files (can be several)
    into one hdf5 file, for later analysis
    
    Parameters
    ----------
    edf_files : List/arr
        list of absolute filenames for edf files
    hdf_file : str
        absolute filename of output hdf5 file

    Outputs
    -------
    all_alias: List
        list of strings with alias for each run
    
    """
    
    # first check if hdf5 already exists
    if os.path.isfile(hdf_file):
        print('The file %s already exists, skipping'%hdf_file)
        
    else:
        ho = hedfpy.HDFEyeOperator(hdf_file)

        all_alias = []

        for ef in edf_files:
            alias = os.path.splitext(os.path.split(ef)[1])[0] #name of data for that run
            ho.add_edf_file(ef)
            ho.edf_message_data_to_hdf(alias = alias) #write messages ex_events to hdf5
            ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = pupil_hp, pupil_lp = pupil_lp) #add raw and preprocessed data to hdf5   

            all_alias.append(alias)
    
    return all_alias


def plot_sacc_hist(df_sacc, outpath):
    
    
    """ plot saccade histogram
    
    Parameters
    ----------
    df_sacc : pd dataframe
        with saccade data
    outpath : str
        absolute path to save plot
    
    """
    # plot gaze density

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(30,15))
    #fig.subplots_adjust(hspace = .25, wspace=.001)

    plt_counter = 0

    for i in range(3):

        for w in range(2):
            
            if (i==2) and (w==1):
                
                print('no more runs, done!')
            
            else:
                
                run = df_sacc['run'].iloc[plt_counter]

                amp = df_sacc.loc[(df_sacc['run'] == run)]['expanded_amplitude'].values[0]

                if amp == [0]: # if 0, then no saccade

                    amp = [np.nan]

                a = sns.histplot(ax = axs[i,w], 
                                x = amp,
                                color = 'red')
                a.tick_params(labelsize=15)
                a.set_xlabel('Amplitude (degrees)',fontsize=15, labelpad = 15)

                axs[i][w].set_title(run,fontsize=18)
                axs[i][w].axvline(0.5, lw=0.5, color='k',alpha=0.5,linestyle='--')

                # count number of saccades with amplitude bigger than 0.5 deg
                sac_count = len(np.where(np.array(amp) >= 0.5)[0])
                axs[i][w].text(0.7, 0.9,'%i saccades > 0.5deg'%(sac_count), 
                               ha='center', va='center', transform=axs[i][w].transAxes,
                              fontsize = 15)

                plt_counter += 1

            
    fig.savefig(os.path.join(outpath,'sacc_histogram.png'))
    
    # combine all runs
    
    amp_all_runs = []
    for _,a in enumerate(df_sacc['expanded_amplitude'].values):
        amp_all_runs = amp_all_runs+a
        
    # plot hist of all combined!
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30,15)) #plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

    a = sns.histplot(ax = axs,
                     x = amp_all_runs,
                    color = 'red')
    a.tick_params(labelsize=40)
    a.set_xlabel('Amplitude (degrees)',fontsize=40, labelpad = 15)
    a.set_ylabel('')


    axs.set_title('Saccade amplitude across pRF runs',fontsize=40)
    axs.axvline(0.5, lw=0.5, color='k',alpha=0.75,linestyle='--')

    # count number of saccades with amplitude bigger than 0.5 deg
    sac_count = len(np.where(np.array(amp_all_runs) >= 0.5)[0])
    axs.text(0.7, 0.9,'%i saccades > 0.5deg'%(sac_count), 
                   ha='center', va='center', transform=axs.transAxes,
                  fontsize = 30)   
    fig.savefig(os.path.join(outpath,'sacc_all_runs_histogram.png'))


def get_saccade_angle(arr, angle_unit='radians'):
    
    """
    convert vector position of saccade to angle
    given a list of vector locations (N x 2)
    """
    
    # compute complex location
    complex_list = [sac[0] + sac[1]*1j for _,sac in enumerate(arr)]
    
    if angle_unit == 'degrees':
        deg_unit = True
    else:
        deg_unit = False
    
    # actually calculate angle
    angles = np.angle(complex_list, deg = deg_unit)
    
    return list(angles)


def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False,color='g',alpha=0.5, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """

    if not np.isnan(angles[0]):    
        # Wrap angles to [-pi, pi)
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        # Set bins symetrically around zero
        if start_zero:
            # To have a bin edge at zero use an even number of bins
            if bins % 2:
                bins += 1
            bins = np.linspace(-np.pi, np.pi, num=bins+1)

        # Bin data and record counts
        count, bin = np.histogram(angles, bins=bins)

        # Compute width of each bin
        widths = np.diff(bin)

        # By default plot density (frequency potentially misleading)
        if density is None or density is True:
            # Area to assign each bin
            area = count / angles.size
            # Calculate corresponding bin radius
            radius = (area / np.pi)**.5
        else:
            radius = count

        # Plot data on ax
        ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
               edgecolor='0.5', fill=True, linewidth=1,color=color,alpha=alpha)

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        if lab_unit == "radians":
            label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
            ax.set_xticklabels(label)
    