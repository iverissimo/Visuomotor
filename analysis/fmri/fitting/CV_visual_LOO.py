# cross validate pRF runs, saving CV_rsq

import os
import yaml
import sys
import os.path as op

import numpy as np

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter


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

# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# gii file extension, for both tasks
file_extension = params['fitting']['prf']['extension']
# fit model used
fit_model = params['fitting']['prf']['fit_model']

# estimates file extensions
estimate_prf_ext = '_CV_estimates.npz'

# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
postfmriprep_pth = op.join(deriv_pth,'post_fmriprep','prf','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

loo_runs = ['leave_01_out','leave_02_out','leave_03_out','leave_04_out','leave_05_out']

### define model

# get screenshot pngs
png_path = op.join(os.getcwd(),'prf_stimuli')
png_filename = [op.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

# set design matrix filename
dm_filename = op.join(os.getcwd(), 'prf_dm_square.npy')

# subjects that did pRF task with linux computer, so res was full HD
HD_subs = [str(num).zfill(2) for num in params['general']['HD_screen_subs']] 
res = params['general']['screenRes_HD'] if str(sj).zfill(2) in HD_subs else params['general']['screenRes']

# create design matrix
screenshot2DM(png_filename, 0.1, res, dm_filename, dm_shape = 'square')  # create it

print('computed %s' % (dm_filename))

# actually load DM
prf_dm = np.load(dm_filename,allow_pickle=True)
prf_dm = prf_dm.T # then it'll be (x, y, t)

# shift it to be average of every 2TRs DM
prf_dm = shift_DM(prf_dm)

# crop DM because functional data also cropped 
prf_dm = prf_dm[:,:,params['processing']['crop_pRF_TR']:]

# define model params
TR = params['general']['TR']

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['general']['screen_width'],
                         screen_distance_cm = params['general']['screen_distance'],
                         design_matrix = prf_dm,
                         TR = TR)

# set grid parameters
grid_nr = params['fitting']['prf']['grid_steps']
sizes = params['fitting']['prf']['max_size'] * np.linspace(np.sqrt(params['fitting']['prf']['min_size']/params['fitting']['prf']['max_size']),1,grid_nr)**2
eccs = params['fitting']['prf']['max_eccen'] * np.linspace(np.sqrt(params['fitting']['prf']['min_eccen']/params['fitting']['prf']['max_eccen']),1,grid_nr)**2
polars = np.linspace(0, 2*np.pi, grid_nr)

# to set up parameter bounds in iterfit
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees
xtol = 1e-7
ftol = 1e-6

# model parameter bounds
gauss_bounds = [(-2*ss, 2*ss),  # x
                (-2*ss, 2*ss),  # y
                (eps, 2*ss),  # prf size
                (0, +inf),  # prf amplitude
                (-5, +inf)]  # bold baseline

css_bounds = [(-2*ss, 2*ss),  # x
              (-2*ss, 2*ss),  # y
              (eps, 2*ss),  # prf size
              (0, +inf),  # prf amplitude
              (-5, +inf),  # bold baseline
              (params['fitting']['prf']['min_n'], params['fitting']['prf']['max_n'])]  # CSS exponent


# define model 
gauss_model = Iso2DGaussianModel(stimulus = prf_stim
                                )

# compute CV rsq for each run

#idx = 0
#LOO = loo_runs[idx]

for idx, LOO in enumerate(loo_runs):
    
    # where to get fits 
    fits_pth = op.join(deriv_pth,'prf_fit','sub-{sj}'.format(sj = sj), 'iterative_gauss','loo_run',LOO) # path to pRF fits

    # to save CV rsq 
    out_pth = op.join(deriv_pth,'prf_fit','sub-{sj}'.format(sj = sj), 'CV', 'rsq', LOO)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth) 
        
    # load estimates from fit
    estimates_filename = []

    # for each hemi field
    for _,field in enumerate(['hemi-L','hemi-R']): 

        # check if chunks already combined
        combined_filename = 'sub-{sj}_task-prf_run-{run}_hemi-{h}_{ext}'.format(sj = sj, 
                                                                                 run = LOO,
                                                                                 h = field,
                                                                                 ext = file_extension.replace('.func.gii','_iterative_gauss_estimates.npz')) 

        # if combined estimates doesn't exist, make it
        if not op.isfile(op.join(fits_pth,combined_filename)): 

            join_chunks(LOO_pth, combined_filename, field,
                        chunk_num = total_chunks, fit_model = 'iterative_gauss')      

        estimates_filename.append(op.join(fits_pth,combined_filename))

    print('loading %s'%estimates_filename)

    estimates_L = np.load(estimates_filename[0],allow_pickle = True)
    estimates_R = np.load(estimates_filename[1],allow_pickle = True)

    # join estimates from both hemispheres
    estimates = {'x': np.concatenate((estimates_L['x'],estimates_R['x'])),
                 'y': np.concatenate((estimates_L['y'],estimates_R['y'])),
                 'size': np.concatenate((estimates_L['size'],estimates_R['size'])),
                 'betas': np.concatenate((estimates_L['betas'],estimates_R['betas'])),
                 'baseline': np.concatenate((estimates_L['baseline'],estimates_R['baseline'])),
                 'r2': np.concatenate((estimates_L['r2'],estimates_R['r2']))
                }

    # load data from the left out run

    # list with left out run path (iff gii, then 2 hemispheres)
    gii_lo_run = [op.join(postfmriprep_pth, h) for h in os.listdir(postfmriprep_pth) if 'run-'+LOO[6:8] in h and 
                  h.endswith(params['fitting']['soma']['extension'])]

    # load data from both hemispheres
    data_loo = []
    for _,loo_file in enumerate(gii_lo_run):
        print('loading data from left out run %s' % loo_file)    
        data_loo.append(np.array(surface.load_surf_data(loo_file)))

    data_loo = np.vstack(data_loo) # will be (vertex, TR)
    print('data array with shape %s'%str(data_loo.shape))
    
    if not op.exists(op.join(out_pth,'CV_estimates.npz')):

        # get model prediciton for all vertices
        prediction = Parallel(n_jobs=-1, verbose=10)(delayed(gauss_model.return_prediction)(estimates['x'][vert],
                                            estimates['y'][vert],
                                            estimates['size'][vert],
                                            estimates['betas'][vert],
                                            estimates['baseline'][vert]) for vert in range(data_loo.shape[0]))

        prediction = np.squeeze(prediction, axis=1)
        
        #calculate CV-rsq        
        CV_rsq = np.nan_to_num(1-np.sum((data_loo-prediction)**2, axis=-1)/(data_loo.shape[-1]*data_loo.var(-1)))
        
        # save CV_rsq
        np.savez(op.join(out_pth,'CV_estimates.npz'),
                r2 = CV_rsq,
                prediction = prediction)



