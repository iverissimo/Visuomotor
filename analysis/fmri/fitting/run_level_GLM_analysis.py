# get first level results (a.k.a do run-level analysis)

import os
import yaml
import sys
import os.path as op

import pandas as pd
import numpy as np

from popeye import utilities

from nistats.design_matrix import make_first_level_design_matrix
from nistats.contrasts import compute_contrast

from nistats.reporting import plot_design_matrix

from joblib import Parallel, delayed

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

import cortex

import datetime

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

# define participant number 

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1)'
                    'as 1st argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)

# print start time, for bookeeping
start_time = datetime.datetime.now()

# set general params, for model
TR = params['general']['TR']
hrf = utilities.spm_hrf(0,TR)

# use smoothed data?
fit_smooth = False

# define tasks
tasks = params['general']['tasks']

## for each task

#cond = 'prf'
for _, cond in enumerate(tasks):

    # file extension
    file_extension = params['fitting'][cond]['extension']
    if fit_smooth:
        new_ext = '_smooth%d'%params['processing']['smooth_fwhm']+params['processing']['extension']
        file_extension = file_extension.replace(params['processing']['extension'],new_ext)

    # define input and output file dirs
    # depending on machine used to fit
    fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

    deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
    postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep',cond,'sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

    out_pth = op.join(deriv_pth, cond+'_fit','sub-{sj}'.format(sj=sj), 'run_level') # path to save estimates

    if not op.exists(out_pth): # check if path to save processed files exist
        os.makedirs(out_pth)   

    # define list of runs    
    run_conds = {'prf': ['01','02','03','04','05'], 
                 'soma': ['01','02','03','04']}
    run_list = run_conds[cond]

    # colormaps to plot stuff
    cmaps_cond = {'prf': 'Reds', 
                 'soma': 'Greens'}

    # iterate over all
    #count = 0
    #r = run_list[count]
    
    for count,r in enumerate(run_list):
        
        print('Fitting GLM on run-%s'%r)

        # list with left out run path (iff gii, then 2 hemispheres)
        gii_lo_run = [op.join(postfmriprep_pth, h) for h in os.listdir(postfmriprep_pth) if 'run-'+r in h and 
                      h.endswith(params['fitting']['soma']['extension'])]

        # load data from both hemispheres
        data_loo = []
        for _,loo_file in enumerate(gii_lo_run):
            print('loading data from left out run %s' % loo_file)    
            data_loo.append(np.array(surface.load_surf_data(loo_file)))

        data_loo = np.vstack(data_loo) # will be (vertex, TR)
        print('data array with shape %s'%str(data_loo.shape))

        if cond == 'prf':
            # path to estimates, obtained from the fits on the average of the other runs
            estimates_pth = op.join(deriv_pth,'prf_fit','sub-{sj}'.format(sj=sj), 'CV', 'rsq', 'leave_%s_out'%r)
        elif cond == 'soma':
            # path to estimates, obtained from the fits on the average of the other runs
            estimates_pth = op.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj=sj), 'loo_run', 'leave_%s_out'%r)

        estimates_pth = [op.join(estimates_pth, h) for h in os.listdir(estimates_pth) if 'CV_estimates.npz' in h][0]
        print('predictions from %s'%str(estimates_pth))

        CV_estimates = np.load(estimates_pth,allow_pickle=True)
        prediction = CV_estimates['prediction']

        # set DM per vertex

        DM_all = np.zeros((prediction.shape[0], prediction.shape[1], 2))

        for i in range(prediction.shape[0]):

            DM_all[i,:,0] = np.repeat(1,prediction.shape[1]) # intercept
            DM_all[i,:,1] = prediction[i] # "regressor" (which will be the model from the other fitted runs)

        run_level_estimates_filename = op.join(out_pth,op.split(estimates_pth)[-1].replace('CV_estimates','run-%s_level_estimates'%r))

        if not op.isfile(run_level_estimates_filename): # if doesn't exist already
            print('fitting GLM to %d vertices'%data_loo.shape[0])
            run_level_params = Parallel(n_jobs=16)(delayed(fit_glm_get_t_stat)(vert, DM_all[indx,...], np.array([0,1])) for indx,vert in enumerate(data_loo))

            np.savez(run_level_estimates_filename,
                        betas = np.array([run_level_params[i][0] for i in range(data_loo.shape[0])]),
                        r2 = np.array([run_level_params[i][1] for i in range(data_loo.shape[0])]),
                        t_val = np.array([run_level_params[i][2] for i in range(data_loo.shape[0])]),
                        cb = np.array([run_level_params[i][3] for i in range(data_loo.shape[0])]),
                        effect_var = np.array([run_level_params[i][4] for i in range(data_loo.shape[0])]),)
        else:
            print('loading %s'%run_level_estimates_filename)

        # load estimates for further calculations
        run_level_params = np.load(run_level_estimates_filename,allow_pickle=True)

        if count==0:
            df_run_all = pd.DataFrame({'run': r,'betas': [run_level_params['betas']],
                                        'r2': [run_level_params['r2']],'t_val': [run_level_params['t_val']],
                                        'cb': [run_level_params['cb']],'effect_var': [run_level_params['effect_var']]})
        else:
            df_run_all = df_run_all.append(pd.DataFrame({'run': r,'betas': [run_level_params['betas']],
                                        'r2': [run_level_params['r2']],'t_val': [run_level_params['t_val']],
                                        'cb': [run_level_params['cb']],'effect_var': [run_level_params['effect_var']]}))


    # save summary df - need to check, is saving arrays as strings
    df_run_all.to_csv(op.join(out_pth,'sub-%s_all_run_stats.csv'%sj), index=False)

    # and numpy array 
    np.savez(op.join(out_pth,'sub-%s_all_average_stats.npz'%sj),
              r2 = np.mean(df_run_all['r2'].values),
              t_val = np.mean(df_run_all['t_val'].values),
              cb = np.mean(df_run_all['cb'].values))

    # plot for now, for sanity check
    # after deciding, need to see which one we roll with

    vmax = 0.7 if cond=='soma' else 0.2

    # average r2 from fits
    images = cortex.Vertex(np.mean(df_run_all['r2'].values), 
                              params['processing']['space'],
                               vmin = 0, vmax = vmax, #vmin = 0.2, vmax = .6,
                               cmap = cmaps_cond[cond])
    #cortex.quickshow(images,with_curvature=True,with_sulci=True)
    filename = op.join(out_pth,'flatmap_space-fsaverage_type-average_rsquared-median_loo_runs_glmfit_%s.svg'%cond)
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

    # average t-val from contrast

    mean_tval = np.mean(df_run_all['t_val'].values)
    mean_tval[np.isnan(mean_tval)==True] = 0 # for nicer plots

    images = cortex.Vertex(mean_tval, 
                          params['processing']['space'],
                           vmin = 0, vmax = 5,
                           cmap = cmaps_cond[cond])
    #cortex.quickshow(images,with_curvature=True,with_sulci=True)
    filename = os.path.join(out_pth,'flatmap_space-fsaverage_type-average_t_val_across_runs_%s.svg'%cond)
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

    # fixed effects t statistic? (as nilearn does it)
    fixed_effects_T = np.mean(df_run_all['cb'].values)/np.sqrt(np.mean(df_run_all['effect_var'].values)/len(df_run_all['effect_var'].values))

    images = cortex.Vertex(fixed_effects_T, 
                          params['processing']['space'],
                           vmin = -5, vmax = 5,
                           cmap='coolwarm_r')
    #cortex.quickshow(images,with_curvature=True,with_sulci=True)
    filename = os.path.join(out_pth,'flatmap_space-fsaverage_type-fixedeffects_t_val_across_runs_%s.svg'%cond)
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)





