
################################################
#      Make and save hand region 
#       related surface plots 
################################################

import re, os
import glob, yaml
import sys

import numpy as np

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

from nistats.design_matrix import make_first_level_design_matrix

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)
    

# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01)  or "median" '  
                    'as 1st argument in the command line!') 

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets 


# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set general params of model/plotting
TR = params['general']['TR']
z_threshold = params['plotting']['soma']['z_threshold']


# file extension
file_extension = params['fitting']['soma']['extension']

# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

# path to figures
figures_pth = os.path.join(deriv_pth,'plots','hand_plots','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 
    
# path to save computed estimates (ex smoothed COM)
out_estimates_dir = os.path.join(deriv_pth,'estimates','soma','sub-{sj}'.format(sj = sj))
if not os.path.exists(out_estimates_dir):
    os.makedirs(out_estimates_dir) 


# change this to simplify appending all subs and making median plot
if sj == 'median':
    all_sj = params['general']['subs']
    sj = [str(x).zfill(2) for _,x in enumerate(all_sj) if x not in params['general']['exclude_subs']]
else:
    sj = [sj]


for i,s in enumerate(sj): # for each subject (if all)
    
    all_RL_zmasked = {'lhand': [],'rhand': []}
    all_COM = {'lhand': [],'rhand': []}
    
    fits_pth = os.path.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj=s)) # path to soma fits of each sub

    # load z-score localizer area, for both hand movements
    z_masked = np.load(os.path.join(fits_pth,'zscore_thresh-%.2f_upper_limb_vs_all_contrast.npy' %(z_threshold)), allow_pickle=True)

    # load z-score localizer area, for Right vs Left movements
    z_RL_masked = np.load(os.path.join(fits_pth,'glm_stats_RvsL-hand_contrast_zscore_thresh-%.2f.npz' %(z_threshold)), allow_pickle=True)
    z_RL_masked = z_RL_masked['zscore']

    # sub specific plots should be save in sub folder (for sanity checking)
    sub_out_estimates_dir = out_estimates_dir.replace('sub-{sj}'.format(sj = str(sys.argv[1]).zfill(2)),'sub-{sj}'.format(sj = s))
    if not os.path.exists(sub_out_estimates_dir):
        os.makedirs(sub_out_estimates_dir) 

    # plot RL just for sanity check
    images = {}

    # vertex for right vs left hand 
    images['RL'] = cortex.Vertex(z_RL_masked, params['processing']['space'],
                               vmin=-5, vmax=5,
                               cmap='BuWtRd')

    cortex.quickshow(images['RL'],with_curvature=True,with_sulci=True,with_colorbar=True)
    filename = os.path.join(figures_pth.replace('sub-{sj}'.format(sj = str(sys.argv[1]).zfill(2)),'sub-{sj}'.format(sj = s)),
                            'flatmap_space-fsaverage_zscore-%.2f_type-RLhands.svg' %(z_threshold))
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images['RL'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


    # load beta estimates
    estimates_filename = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if x.endswith(file_extension.replace('.func.gii','_estimates.npz'))][0]
    soma_estimates = np.load(estimates_filename,allow_pickle=True)
    betas = soma_estimates['betas']

    if i==0: # only do it once because we are interested in the DM column names
        # get events
        events_avg_file = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if x.endswith('_run-median_events.tsv')][0]
        events_avg = pd.read_csv(events_avg_file,sep = '\t').drop('Unnamed: 0', axis=1)

        frame_times = TR * (np.arange(soma_estimates['prediction'][0].shape[0])) # specifying the timing of fMRI frames

        # Create the design matrix, hrf model containing Glover model 
        design_matrix = make_first_level_design_matrix(frame_times,
                                                       events = events_avg,
                                                       hrf_model = 'glover'
                                                       )
    # get design matrix column of regressors keys
    all_regressors = design_matrix.columns # all regressor names
    regressors = params['fitting']['soma']['all_contrasts']['upper_limb'] # relevant regressors within face region
    
    # reshape betas into 2D array
    new_betas = np.array([betas[k] if not np.isnan(betas[k]).any() else np.tile(np.nan,len(all_regressors)) for k in range(betas.shape[0])])

    # loop for each hand
    hands = ['lhand','rhand']

    for _,hnd in enumerate(hands):

        # choose side to thresh depending on hand (duw to contrast used)
        side_str = 'above' if hnd == 'rhand' else 'below'
        z_hand_masked = mask_arr(z_RL_masked.copy(), threshold = z_threshold, side = side_str)

        # set regressor names for left and right hand
        regressors_H = [x for _,x in enumerate(regressors) if hnd in x]

        # load and append each face part beta value in list
        betas_reg = [] 

        for k,part in enumerate(regressors_H):

            # get regressor index
            reg_ind = np.where(all_regressors==part)[0][0]

            # check if smoothed file exists, if not smooth it for plotting
            betas_smooth_filename = [os.path.join(sub_out_estimates_dir,x) for _,x in enumerate(os.listdir(sub_out_estimates_dir)) 
                                 if x.endswith('betas-%s_smooth%d.npy'%(part, params['processing']['smooth_fwhm']))]

            # smooth betas for that regressor (for plotting)
            if not betas_smooth_filename: # if no smooth file exists

                # get path to post_fmriprep files, for header info 
                post_proc_gii_pth = os.path.join(deriv_pth,'post_fmriprep', 'soma','sub-{sj}'.format(sj = s), 'median')
                post_proc_gii = [os.path.join(post_proc_gii_pth,x) for _,x in enumerate(os.listdir(post_proc_gii_pth)) if params['processing']['space'] in x and x.endswith(file_extension)]
                post_proc_gii.sort()

                betas_smooth = smooth_nparray(new_betas[...,reg_ind].copy(), 
                                               post_proc_gii, 
                                               sub_out_estimates_dir, 
                                               '_betas-%s'%(part), 
                                               sub_space = params['processing']['space'], 
                                               n_TR = params['plotting']['soma']['n_TR'], 
                                               smooth_fwhm = params['processing']['smooth_fwhm'],
                                               sub_ID = s)

            else:
                print('loading %s'%betas_smooth_filename[0])
                betas_smooth = np.load(betas_smooth_filename[0])

            if k == 0:
                # do same for z-score  
                # check if smoothed file exists, if not smooth it for plotting
                z_smooth_filename = [os.path.join(sub_out_estimates_dir,x) for _,x in enumerate(os.listdir(sub_out_estimates_dir)) 
                                     if x.endswith('zscore_thresh-%.2f_RL-%s_contrast_smooth%d.npy'%(z_threshold,hnd,params['processing']['smooth_fwhm']))]

                if not z_smooth_filename: # if no smooth file exists

                    # get path to post_fmriprep files, for header info 
                    post_proc_gii_pth = os.path.join(deriv_pth,'post_fmriprep', 'soma','sub-{sj}'.format(sj = s), 'median')
                    post_proc_gii = [os.path.join(post_proc_gii_pth,x) for _,x in enumerate(os.listdir(post_proc_gii_pth)) if params['processing']['space'] in x and x.endswith(file_extension)]
                    post_proc_gii.sort()

                    z_smooth = smooth_nparray(z_hand_masked, 
                                               post_proc_gii, 
                                               sub_out_estimates_dir, 
                                               '_zscore_thresh-%.2f_RL-%s_contrast'%(z_threshold,hnd), 
                                               sub_space = params['processing']['space'], 
                                               n_TR = params['plotting']['soma']['n_TR'], 
                                               smooth_fwhm = params['processing']['smooth_fwhm'],
                                               sub_ID = s)

                else:
                    print('loading %s'%z_smooth_filename[0])
                    z_smooth = np.load(z_smooth_filename[0])


                # append for later plotting
                all_RL_zmasked[hnd].append(z_smooth) # save for later


            betas_reg.append(betas_smooth)

        betas_reg = np.array(betas_reg)

        # compute COM from betas values
        print('Computing center of mass for face elements %s' %(str(regressors_H)))
        allparts_COM = COM(betas_reg)

        all_COM[hnd].append(allparts_COM)


# now do median (for when we are looking at all subs)
# and plot

# RIGHT hand
z_RH_median = np.nanmedian(np.array(all_RL_zmasked['rhand']), axis = 0)
COM_RH_median = np.nanmedian(np.array(all_COM['rhand']), axis = 0)
    
# need to mask again because smoothing removes nans
z_RH_median = mask_arr(z_RH_median, threshold = z_threshold, side = 'above')
# ignore smoothed nan voxels
COM_RH_median[np.isnan(z_RH_median)] = np.nan    
    

# vertex for face vs all others
images['v_RHand'] = cortex.Vertex(z_RH_median, params['processing']['space'],
                           vmin=0, vmax=7,
                           cmap='Reds')

cortex.quickshow(images['v_RHand'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-Rhand.svg'%z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_RHand'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


images['v_Rfingers'] = cortex.Vertex(COM_RH_median, params['processing']['space'],
                           vmin=0, vmax=4,
                           cmap='rainbow_r')#costum colormap added to database

cortex.quickshow(images['v_Rfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-RHfingers.svg' %(z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# LEFT hand
z_LH_median = np.nanmedian(np.array(all_RL_zmasked['lhand']), axis = 0)
COM_LH_median = np.nanmedian(np.array(all_COM['lhand']), axis = 0)

# need to mask again because smoothing removes nans
z_LH_median = mask_arr(z_LH_median, threshold = z_threshold, side = 'below')
# ignore smoothed nan voxels
COM_LH_median[np.isnan(z_LH_median)] = np.nan


# vertex for face vs all others
images['v_LHand'] = cortex.Vertex(z_LH_median, params['processing']['space'],
                           vmin=-7, vmax=0,
                           cmap='Blues_r')

cortex.quickshow(images['v_LHand'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-Lhand.svg'%z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_LHand'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


images['v_Lfingers'] = cortex.Vertex(COM_LH_median, params['processing']['space'],
                           vmin=0, vmax=4,
                           cmap = 'rainbow_r')#costum colormap added to database

cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-LHfingers.svg' %(z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)














