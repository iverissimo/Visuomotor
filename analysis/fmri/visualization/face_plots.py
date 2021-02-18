
################################################
#      Make and save face region 
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
figures_pth = os.path.join(deriv_pth,'plots','face_plots','sub-{sj}'.format(sj = sj)) # path to save plots
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


# variables to store all values, and compute median (if median subject)
all_zmasked = []
all_COM = []

for i,s in enumerate(sj): # for each subject (if all)
    
    fits_pth = os.path.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj = s)) # path to soma fits of each sub

    # load z-score localizer area, for face movements
    z_masked = np.load(os.path.join(fits_pth,'zscore_thresh-%.2f_face_vs_all_contrast.npy' %(z_threshold)), allow_pickle=True)
    
    # append for later plotting
    all_zmasked.append(z_masked)
    
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
    regressors = params['fitting']['soma']['all_contrasts']['face'] # relevant regressors within face region

    # reshape betas into 2D array
    new_betas = np.array([betas[k] if not np.isnan(betas[k]).any() else np.tile(np.nan,len(all_regressors)) for k in range(betas.shape[0])])

    # load and append each face part beta value in list
    betas_reg = [] 

    for _,part in enumerate(regressors):

        # get regressor index
        reg_ind = np.where(all_regressors==part)[0][0]
        
        # smooth betas for that regressor (for plotting)
        
        # check if smoothed file exists, if not smooth it for plotting
        sub_out_estimates_dir = out_estimates_dir.replace('sub-{sj}'.format(sj = str(sys.argv[1]).zfill(2)),'sub-{sj}'.format(sj = s))
        betas_smooth_filename = [os.path.join(sub_out_estimates_dir,x) for _,x in enumerate(os.listdir(sub_out_estimates_dir)) 
                             if x.endswith('betas-%s_smooth%d.npy'%(part, params['processing']['smooth_fwhm']))]

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
                                           sub_ID = str(sys.argv[1]).zfill(2))

        else:
            print('loading %s'%betas_smooth_filename[0])
            betas_smooth = np.load(betas_smooth_filename[0])


        # mask betas for vertices within region
        #betas_masked = new_betas[...,reg_ind].copy()
        betas_smooth[np.isnan(z_masked)] = np.nan

        betas_reg.append(betas_smooth)

    betas_reg = np.array(betas_reg)
    
    # compute COM from betas values
    print('Computing center of mass for face elements %s' %(params['fitting']['soma']['all_contrasts']['face']))
    allparts_COM = COM(betas_reg)

    all_COM.append(allparts_COM)


# now do median (for when we are looking at all subs)
all_zmasked_median = np.nanmedian(np.array(all_zmasked), axis = 0)
all_COM_median = np.nanmedian(np.array(all_COM), axis = 0)

# check if smoothed file exists, if not smooth it for plotting
z_smooth_filename = [os.path.join(out_estimates_dir,x) for _,x in enumerate(os.listdir(out_estimates_dir)) 
                     if x.endswith('zscore_thresh-%.2f_face_vs_all_contrast.npy'.replace('.npy','_smooth%d.npy'%params['processing']['smooth_fwhm'])%(z_threshold))]

if not z_smooth_filename: # if no smooth file exists

    # get path to post_fmriprep files, for header info 
    post_proc_gii_pth = os.path.join(deriv_pth,'post_fmriprep', 'soma','sub-{sj}'.format(sj = sj[0]), 'median')
    post_proc_gii = [os.path.join(post_proc_gii_pth,x) for _,x in enumerate(os.listdir(post_proc_gii_pth)) if params['processing']['space'] in x and x.endswith(file_extension)]
    post_proc_gii.sort()
    
    z_smooth = smooth_nparray(all_zmasked_median, 
                               post_proc_gii, 
                               out_estimates_dir, 
                               '_zscore_thresh-%.2f_face_vs_all_contrast'%(z_threshold), 
                               sub_space = params['processing']['space'], 
                               n_TR = params['plotting']['soma']['n_TR'], 
                               smooth_fwhm = params['processing']['smooth_fwhm'],
                               sub_ID = str(sys.argv[1]).zfill(2))

else:
    print('loading %s'%z_smooth_filename[0])
    z_smooth = np.load(z_smooth_filename[0])
    

# now plot in surface and save figure
images = {}

# need to mask again because smoothing removes nans
z_smooth = mask_arr(z_smooth, threshold = z_threshold, side = 'above')

# vertex for face vs all others
images['v_face'] = cortex.Vertex(z_smooth, params['processing']['space'],
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_face'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-face-vs-all.svg'%z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_face'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# ignore smoothed nan voxels
all_COM_median[np.isnan(z_smooth)] = np.nan

# 'eyebrows', 'eyes', 'mouth','tongue', , combined
images['v_facecombined'] = cortex.Vertex(all_COM_median, 'fsaverage',
                           vmin=0, vmax=3,
                           cmap='J4') #costum colormap added to database

#cortex.quickshow(images['v_facecombined'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_zscore-%.2f_type-eyebrows-eyes-mouth-tongue.svg' %(z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)







