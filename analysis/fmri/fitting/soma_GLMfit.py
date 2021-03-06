import re, os
import glob, yaml
import sys

import pandas as pd
import numpy as np

from popeye import utilities

from nistats.design_matrix import make_first_level_design_matrix
from nistats.contrasts import compute_contrast

from nistats.reporting import plot_design_matrix

from joblib import Parallel, delayed

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions


# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number and which chunk of data to fit

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) or "median" '
                    'as 1st argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets   
    sj = 'sub-'+str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets


# set general params, for model
TR = params['general']['TR']
hrf = utilities.spm_hrf(0,TR)
z_threshold = params['plotting']['soma']['z_threshold']

# use smoothed data?
fit_smooth = params['fitting']['soma']['fit_smooth']

# file extension
file_extension = params['fitting']['soma']['extension']
if fit_smooth:
    new_ext = '_smooth%d'%params['processing']['smooth_fwhm']+params['processing']['extension']
    file_extension = file_extension.replace(params['processing']['extension'],new_ext)


# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

   
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','soma',sj) # path to post_fmriprep files

out_pth = os.path.join(deriv_pth,'soma_fit',sj) # path to save estimates

if not os.path.exists(out_pth): # check if path to save processed files exist
    os.makedirs(out_pth)   
    
# send message to user
print('fitting functional files from %s'%postfmriprep_pth)

# list of functional files
filename = [os.path.join(postfmriprep_pth,run) for run in os.listdir(postfmriprep_pth) if 'soma' in run and params['processing']['space'] in run and run.endswith(file_extension)]
filename.sort()

# check if median run is computed, if not make it 
median_path = os.path.join(postfmriprep_pth,'median')

if not os.path.exists(median_path): 
    os.makedirs(median_path) 

med_gii = []
for field in ['hemi-L', 'hemi-R']:
    hemi = [h for h in filename if field in h and 'run-median' not in h]  #we don't want to average median run if already in original dir

    # set name for median run (now numpy array)
    med_file = os.path.join(median_path, re.sub('run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
    # if file doesn't exist
    if not os.path.exists(med_file):
        med_gii.append(median_gii(hemi, median_path))  # create it
        print('computed %s' % (med_gii))
    else:
        med_gii.append(med_file)
        print('median file %s already exists, skipping' % (med_gii))
        

# get events file
sourcedata_pth = params['general']['paths']['data']['sourcedata']
events_dir = [run for run in glob.glob(os.path.join(sourcedata_pth,sj,'*','func/*')) if 'soma' in run and 'median' not in run and run.endswith('events.tsv')]
print('event files from %s' % os.path.split(events_dir[0])[0])

# make/load average event file
events_avg_file = median_soma_events(events_dir,out_pth)
events_avg = pd.read_csv(events_avg_file,sep = '\t').drop('Unnamed: 0', axis=1)

# load data from both hemispheres
data = []

for ind,gii_file in enumerate(med_gii):

    print('loading data from %s' % gii_file)    
    data.append(np.array(surface.load_surf_data(gii_file)))

data = np.vstack(data) # will be (vertex, TR)
print('data array with shape %s'%str(data.shape))

# specifying the timing of fMRI frames
frame_times = TR * (np.arange(data.shape[-1]))

# Create the design matrix, hrf model containing Glover model 
design_matrix = make_first_level_design_matrix(frame_times,
                                               events = events_avg,
                                               hrf_model = 'glover'
                                               )

# plot design matrix and save just to check if everything fine
plot = plot_design_matrix(design_matrix)
fig = plot.get_figure()
fig.savefig(os.path.join(out_pth,'design_matrix.svg'), dpi=100,bbox_inches = 'tight')

# set estimates filename
estimates_filename = os.path.join(out_pth,os.path.split(med_gii[0])[-1].replace('.func.gii','_estimates.npz').replace('hemi-L_','hemi-both_'))

if not os.path.isfile(estimates_filename): # if doesn't exist already
        print('fitting GLM to %d vertices'%data.shape[0])
        soma_params = Parallel(n_jobs=16)(delayed(fit_glm)(vert, design_matrix.values) for _,vert in enumerate(data))
        soma_params = np.vstack(soma_params)

        np.savez(estimates_filename,
                  prediction = soma_params[..., 0],
                  betas = soma_params[..., 1],
                  r2 = soma_params[..., 2],
                  mse = soma_params[...,3])
else:
    print('loading %s'%estimates_filename)
    
# load betas
soma_estimates = np.load(estimates_filename,allow_pickle=True)
betas = soma_estimates['betas']

# now make simple contrasts
print('Computing simple contrasts')
print('Using z-score of %0.2f as threshold for localizer' %z_threshold)

stats_all = {} # save all computed stats, don't need to load again
data_masked_all = {} # to save masked data array for each region

reg_keys = list(params['fitting']['soma']['all_contrasts'].keys()); reg_keys.sort() # list of key names (of different body regions)
loo_keys = leave_one_out(reg_keys) # loo for keys 

for index,region in enumerate(reg_keys): # one broader region vs all the others

    print('contrast for %s ' %region)
    # list of other contrasts
    other_contr = np.append(params['fitting']['soma']['all_contrasts'][loo_keys[index][0]],
                            params['fitting']['soma']['all_contrasts'][loo_keys[index][1]])

    contrast = set_contrast(design_matrix.columns,[params['fitting']['soma']['all_contrasts'][str(region)],other_contr],
                        [1,-len(params['fitting']['soma']['all_contrasts'][str(region)])/len(other_contr)],
                        num_cond=2)

    # save estimates in dir 
    stats_filename = os.path.join(out_pth,'glm_stats_%s_vs_all_contrast.npz' %(region))

    if not os.path.isfile(stats_filename): # if doesn't exist already
        # compute contrast-related statistics
        soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values,contrast,betas[w]) for w,vert in enumerate(data))
        soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore

        np.savez(stats_filename,
                  t_val = soma_stats[..., 0],
                  p_val = soma_stats[..., 1],
                  zscore = soma_stats[..., 2])

    else:
        print('loading %s'%stats_filename)
    
    stats_all[str(region)] = np.load(stats_filename,allow_pickle=True)
    
    # save thresholded z-scores for each region 
    z_masked_file = os.path.join(out_pth,'zscore_thresh-%.2f_%s_vs_all_contrast' %(z_threshold,region))
    
    if not os.path.isfile(z_masked_file): # if doesn't exist already
        z_masked = mask_arr(stats_all[region]['zscore'],
                            threshold = z_threshold, side = 'above')
        np.save(z_masked_file,z_masked)
    else:
        print('%s already exists'%z_masked_file)
        z_masked = np.load(z_masked_file,allow_pickle=True)

    # mask data - only significant voxels for region
    data_masked = data.copy()
    data_masked[np.isnan(z_masked)] = np.nan
    
    data_masked_all[str(region)] = data_masked


## now do rest of the contrasts within region ###

# compare left and right
print('Right vs Left contrasts')

limbs = [['hand',params['fitting']['soma']['all_contrasts']['upper_limb']],
         ['leg',params['fitting']['soma']['all_contrasts']['lower_limb']]]

for _,key in enumerate(limbs):
    print('For %s' %key[0])

    rtask = [s for s in key[1] if 'r'+key[0] in s]
    ltask = [s for s in key[1] if 'l'+key[0] in s]
    tasks = [rtask,ltask] # list with right and left elements

    contrast = set_contrast(design_matrix.columns,tasks,[1,-1],num_cond=2)

    # save estimates in dir 
    stats_filename = os.path.join(out_pth,'glm_stats_RvsL-%s_contrast_zscore_thresh-%.2f.npz' %(key[0],z_threshold))

    if not os.path.isfile(stats_filename): # if doesn't exist already

        # compute contrast-related statistics
        data4stat = data_masked_all['lower_limb'] if key[0] == 'leg' else data_masked_all['upper_limb']

        soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values, contrast, betas[w]) for w,vert in enumerate(data4stat))
        soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore

        np.savez(stats_filename,
                  t_val = soma_stats[..., 0],
                  p_val = soma_stats[..., 1],
                  zscore = soma_stats[..., 2])

    else:
        print('loading %s'%stats_filename)

    stats_all['RL_%s'%key[0]] = np.load(stats_filename, allow_pickle = True)  
    









