
################################################
#      Plot soma fit on hands vs face, 
#  over timeseries data, for design figure
################################################


import re, os
import glob, yaml
import sys

import numpy as np

from nilearn import surface

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

import matplotlib.pyplot as plt

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01) '	
                    'as 1st argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	


# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# file extension
file_extension = params['fitting']['soma']['extension']

# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','soma','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files
median_pth = os.path.join(postfmriprep_pth,'median') # path to median run files

fits_pth = os.path.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj=sj)) # path to soma fits
# should add here an "if doesn't exist, fit single voxel", at a later stage.

figures_pth = os.path.join(deriv_pth,'plots','single_vertex','somafit','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# set threshold for plotting
z_threshold = params['plotting']['soma']['z_threshold']

# load functional files
# for median run (so we plot over data)

# load data for median run, one hemisphere 
hemi = ['hemi-L','hemi-R']

med_filenames = [os.path.join(median_pth,x) for _,x in enumerate(os.listdir(median_pth)) if 'soma' in x and params['processing']['space'] in x and x.endswith(file_extension)]
data = []
for _,h in enumerate(hemi):
    gii_file = [x for _,x in enumerate(med_filenames) if h in x][0]
    print('loading %s' %gii_file)
    data.append(np.array(surface.load_surf_data(gii_file)))

data = np.vstack(data) # will be (vertex, TR)

TR = params['general']['TR']
# array with 141 timepoints, in seconds
time_sec = np.linspace(0,data.shape[-1]*TR,num=data.shape[-1]) 

# list of key names (of different body regions)
ROIs = ['face','upper_limb']

# make/load average event file
events_avg_file = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth))if x.endswith('run-median_events.tsv')][0]
events_avg = pd.read_csv(events_avg_file,sep = '\t').drop('Unnamed: 0', axis=1)

# load estimates array (to get model timecourse)
estimates_filename = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if 'hemi-both' in x and x.endswith(file_extension.replace('.func.gii','')+'_estimates.npz')][0]
estimates = np.load(estimates_filename,allow_pickle=True)
print('loading estimates array from %s'%estimates_filename)

blue_color = ['#004759','#00a7d1']
data_color = ['#262626','#8a8a8a']

fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)

for idx,rois_ks in enumerate(ROIs):
    
    zmask_filename = os.path.join(fits_pth,'zscore_thresh-%.2f_%s_vs_all_contrast.npy' %(z_threshold,rois_ks))
    zmask = np.load(zmask_filename, allow_pickle = True)
    
    # find max z-score
    if sj == '11' and rois_ks == 'face': 
        index = 325606 # because it's a nice one to plot, to use in figure (rsq = 93% so also good fit)
    elif sj == '11' and rois_ks == 'upper_limb':
        index = 309071 # because it's a nice one to plot, to use in figure (rsq = 93% so also good fit)
    else:
        index =  np.where(zmask == np.nanmax(zmask))[0][0] 
        
    # legend labels for data
    dlabel = 'face' if rois_ks == 'face' else 'hand'

    # plot data with model
    axis.plot(time_sec,estimates['prediction'][index], c = blue_color[idx], lw = 3,label = dlabel+', R$^2$ = %.2f'%estimates['r2'][index],zorder=1)
    axis.scatter(time_sec,data[index], marker='v',s=15,c=blue_color[idx])#,label=dlabel)
    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.tick_params(axis='both', labelsize=18)
    axis.set_xlim(0,len(data[index])*TR)
    if sj=='11': axis.set_ylim(top=4)


    # plot axis vertical bar on background to indicate stimulus display time
    stim_onset = []
    for w in range(len(events_avg)):
        if events_avg['trial_type'][w] in params['fitting']['soma']['all_contrasts'][rois_ks]:
            stim_onset.append(events_avg['onset'][w])
    stim_onset = list(set(stim_onset)); stim_onset.sort()  # to remove duplicate values (both hands situation)

    handles,labels = axis.axes.get_legend_handles_labels()

    ax_count = 0
    for h in range(6):
        incr = 3 if rois_ks=='face' else 4 # increment for vertical bar (to plot it from index 0 to index 4)
        plt.axvspan(stim_onset[ax_count], stim_onset[ax_count+3]+2.25, facecolor=blue_color[idx], alpha=0.1)
        ax_count += 4 if rois_ks=='face' else 5


axis.legend(handles,labels,loc='upper left',fontsize=15)   # doing this to guarantee that legend is how I want it   

fig.savefig(os.path.join(figures_pth,'soma_singvoxfit_timeseries_%s.svg'%(str(ROIs))), dpi=100,bbox_inches = 'tight')









