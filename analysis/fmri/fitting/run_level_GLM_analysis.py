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
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','soma','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

out_pth = op.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj=sj), 'run_level') # path to save estimates

if not op.exists(out_pth): # check if path to save processed files exist
    os.makedirs(out_pth)   

# define list of runs
run_list = ['01','02','03','04']

# iterate over all

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
        
    ## plot design matrix and save just to check if everything fine
    #vert = 26565
    #plot = plot_design_matrix(pd.DataFrame(DM_all[vert,...], columns=['intercept','model']))
    #fig = plot.get_figure()
    #fig.savefig(op.join(out_pth,'design_matrix_vertex-%i.svg'%vert), dpi=100,bbox_inches = 'tight')

    # plot data and model to show as example
    #fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    #plt.plot(data_loo[vert,...], c='black', marker = 'o')
    #plt.plot(prediction[vert], c='red')
    #fig.savefig(op.join(out_pth,'example_model_ondata_vertex-%i.svg'%vert), dpi=100,bbox_inches = 'tight')

    # fit GLM and compute t-stat
    # from contrast of only regressor against implicit baseline
    # to test for the main effect of the model from the average of the other runs
    # on the left out run

    run_level_estimates_filename = op.join(out_pth,op.split(estimates_pth)[-1].replace('CV_estimates','run_level_estimates'))

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

# average r2 from fits
images = cortex.Vertex(np.mean(df_run_all['r2'].values), 
                          params['processing']['space'],
                           vmin = 0, vmax = .7, #vmin = 0.2, vmax = .6,
                           cmap='Blues')
#cortex.quickshow(images,with_curvature=True,with_sulci=True)
filename = op.join(out_pth,'flatmap_space-fsaverage_type-average_rsquared-median_loo_runs_glmfit_soma.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# average t-val from contrast
images = cortex.Vertex(np.mean(df_run_all['t_val'].values), 
                      params['processing']['space'],
                       vmin = 0, vmax = 5,
                       cmap='Blues')
#cortex.quickshow(images,with_curvature=True,with_sulci=True)
filename = os.path.join(out_pth,'flatmap_space-fsaverage_type-average_t_val_across_runs_soma.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
    
# fixed effects t statistic? (as nilearn does it)
fixed_effects_T = np.mean(df_run_all['cb'].values)/np.sqrt(np.mean(df_run_all['effect_var'].values)/len(df_run_all['effect_var'].values))

images = cortex.Vertex(fixed_effects_T, 
                      params['processing']['space'],
                       vmin = -5, vmax = 5,
                       cmap='coolwarm_r')
#cortex.quickshow(images,with_curvature=True,with_sulci=True)
filename = os.path.join(out_pth,'flatmap_space-fsaverage_type-fixedeffects_t_val_across_runs_soma.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
  




