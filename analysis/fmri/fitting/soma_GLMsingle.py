################################################
#   Fit GLM single on all runs
# and obtain single trial beta values
################################################

import re, os
import glob, yaml
import sys
import os.path as op

import pandas as pd
import numpy as np

import cortex

from nilearn import surface

import datetime

from glmsingle.glmsingle import GLM_single

# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

# define participant number and which chunk of data to fit
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

# file extension
file_extension = params['processing']['extension']

# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','soma','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

## output path for estimates
out_pth = op.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj=sj), 
                    'glm_single', params['fitting']['soma']['glm_single_hrf']) # path to save estimates
os.makedirs(out_pth, exist_ok=True)   
    
# get list of files to fit
input_file_pth = op.join(postfmriprep_pth)

# send message to user
print('fitting functional files from %s'%input_file_pth)

# get list with gii files
gii_filenames = [op.join(input_file_pth, name) for name in os.listdir(input_file_pth) if name.endswith('hemi-R{ext}'.format(ext=file_extension)) or \
                name.endswith('hemi-L{ext}'.format(ext=file_extension))]

# get list with run number
run_num_list = np.unique([int(re.findall(r'run-\d{1,3}', op.split(input_name)[-1])[0][4:]) for input_name in gii_filenames])

## load data of all runs
# will be [runs, vertex, TR]

all_data = []
for run_id in run_num_list:
    
    run_data = []
    for hemi in ['hemi-L', 'hemi-R']:
        
        hemi_file = [file for file in gii_filenames if 'run-0%i'%run_id in file and hemi in file][0]
        print('loading %s' %hemi_file)    
        run_data.append(np.array(surface.load_surf_data(hemi_file)))
        
    all_data.append(np.vstack(run_data)) # will be (vertex, TR)


## make design matrix

# get trial condition labels
stim_labels = [op.splitext(val)[0] for val in params['fitting']['soma']['soma_stimulus'] ]
# get unique labels of conditions
_, idx = np.unique(stim_labels, return_index=True) #np.unique(stim_labels)
cond_unique = np.array(stim_labels)[np.sort(idx)]

# get trial timings, in TR
# initial baseline period
start_baseline_dur = int(np.round(params['fitting']['soma']['empty_dur_in_sec']/TR)) # rounding up, will compensate by shifting hrf onset
# trial duration (including ITIs)
trial_dur = (params['fitting']['soma']['iti_in_sec'] * 2 + params['fitting']['soma']['stim_dur_in_sec'])/TR

## define DM [TR, conditions]
# initialize at 0
design_array = np.zeros((np.array(all_data).shape[-1],len(cond_unique)))

# fill it with ones on stim onset
for t, trl_cond in enumerate(stim_labels):
    
    # index for condition column
    cond_ind = np.where(cond_unique == trl_cond)[0][0]
    
    # fill it 
    design_array[int(start_baseline_dur + t*trial_dur),cond_ind] = 1
    print(int(start_baseline_dur + t*trial_dur))
    
# and stack it for each run
all_dm = []
for run_id in run_num_list:
    all_dm.append(design_array)

# create a directory for saving GLMsingle outputs
opt = dict()

# set important fields for completeness (but these would be enabled by default)
if params['fitting']['soma']['glm_single_hrf'] == 'hrf_canonical': # if we want canonical hrf, turn hrf fitting off
    opt['wantlibrary'] = 0
else:
    opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# shift onset by remainder, to make start point accurate
opt['hrfonset'] = -(params['fitting']['soma']['empty_dur_in_sec'] % TR) 

#opt['hrftoassume'] = hrf_final
#opt['brainexclude'] = final_mask.astype(int) #prf_mask.astype(int)
#opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold
#opt['brainR2'] = 100

# trying out defining polinomials to use
#opt['maxpolydeg'] = [[0, 1] for _ in range(data.shape[0])]

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
print(glmsingle_obj.params)

print(f'running GLMsingle...')

# run GLMsingle
results_glmsingle = glmsingle_obj.fit(
   all_dm,
   all_data,
   params['fitting']['soma']['stim_dur_in_sec'],
   TR,
   outputdir=out_pth)
                        

# Print duration
end_time = datetime.datetime.now()
print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                start_time = start_time,
                end_time = end_time,
                dur  = end_time - start_time))

## save some plots for sanity check

flatmap = cortex.Vertex(results_glmsingle['typed']['R2'], 
                  params['processing']['space'],
                   vmin = 0, vmax = 80, #.7,
                   cmap='hot')

#cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
fig_name = op.join(out_pth, 'modeltypeD_rsq.png')
print('saving %s' %fig_name)
_ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

flatmap = cortex.Vertex(results_glmsingle['typed']['FRACvalue'], 
                  params['processing']['space'],
                   vmin = 0, vmax = 1, #.7,
                   cmap='copper')

#cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
fig_name = op.join(out_pth, 'modeltypeD_fracridge.png')
print('saving %s' %fig_name)
_ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

flatmap = cortex.Vertex(np.mean(results_glmsingle['typed']['betasmd'], axis = -1), 
                  params['processing']['space'],
                   vmin = -3, vmax = 3, #.7,
                   cmap='RdBu_r')

#cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
fig_name = op.join(out_pth, 'modeltypeD_avgbetas.png')
print('saving %s' %fig_name)
_ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
