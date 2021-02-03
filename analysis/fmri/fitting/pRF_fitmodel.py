
################################################
#      Do pRF fit on median run, 
# make iterative fit and save outputs 
################################################


import re, os
import glob, yaml
import sys

import numpy as np

from nilearn import surface

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder,CSS_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number and which chunk of data to fit

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add data chunk number to be fitted '
                    'as 3rd argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    chunk_num = str(sys.argv[2]).zfill(3)


# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# use smoothed data?
fit_smooth = params['fitting']['prf']['fit_smooth']

# file extension
file_extension = params['fitting']['prf']['extension']
if fit_smooth:
    new_ext = '_smooth%d'%params['processing']['smooth_fwhm']+params['processing']['extension']
    file_extension = file_extension.replace(params['processing']['extension'],new_ext)


# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','prf','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

out_pth = os.path.join(deriv_pth,'pRF_fit',params['fitting']['prf']['fit_model'],'sub-{sj}'.format(sj=sj)) # path to save estimates

if not os.path.exists(out_pth): # check if path to save processed files exist
    os.makedirs(out_pth) 
    os.makedirs(out_pth.replace(params['fitting']['prf']['fit_model'],'gauss')) #also make gauss dir, to save intermediate estimates

# send message to user
print('fitting functional files from %s'%postfmriprep_pth)

# list of functional files
filename = [os.path.join(postfmriprep_pth,run) for run in os.listdir(postfmriprep_pth) if 'prf' in run and params['processing']['space'] in run and run.endswith(file_extension)]
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


# get screenshot pngs
png_path = os.path.join(os.getcwd(),'prf_stimuli')
png_filename = [os.path.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

# set design matrix filename
dm_filename = os.path.join(os.getcwd(), 'prf_dm_square.npy')

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
hrf = utilities.spm_hrf(0,TR)

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


# sets up stimulus and hrf for this gaussian gridder
gg = Iso2DGaussianGridder(stimulus = prf_stim,
                          hrf = hrf,
                          filter_predictions = True,
                          window_length = params['processing']['sg_filt_window_length'],
                          polyorder = params['processing']['sg_filt_polyorder'],
                          highpass = True,
                          task_lengths = np.array([prf_dm.shape[-1]]))

# and css gridder
gg_css = CSS_Iso2DGaussianGridder(stimulus = prf_stim,
                                  hrf = hrf,
                                  filter_predictions = True,
                                  window_length = params['processing']['sg_filt_window_length'],
                                  polyorder = params['processing']['sg_filt_polyorder'],
                                  highpass = True,
                                  task_lengths = np.array([prf_dm.shape[-1]]))


# fit model per hemisphere
for gii_file in med_gii:
    print('loading data from %s' % gii_file)
    data_all = np.array(surface.load_surf_data(gii_file))
    print('data array with shape %s'%str(data_all.shape))
    
    # number of vertices of chunk
    num_vox_chunk = int(data_all.shape[0]/total_chunks) 
    
    # new data chunk to fit
    data = data_all[num_vox_chunk*(int(chunk_num)-1):num_vox_chunk*int(chunk_num),:]
    
    print('fitting chunk %s/%d of data with shape %s'%(chunk_num,total_chunks,str(data.shape)))
    
    # gaussian fitter
    gf = Iso2DGaussianFitter(data = data, gridder = gg, n_jobs = 16)
    
    #filename for the numpy array with the estimates of the grid fit
    grid_estimates_filename = gii_file.replace('.func.gii', '_chunk-%s_of_%s_gauss_estimates.npz'%(chunk_num,str(total_chunks).zfill(3)))
    grid_estimates_filename = os.path.join(out_pth.replace(params['fitting']['prf']['fit_model'],'gauss'),os.path.split(grid_estimates_filename)[-1])


    if not os.path.isfile(grid_estimates_filename): # if estimates file doesn't exist
        print('%s not found, fitting grid'%grid_estimates_filename)
        # do gaussian grid fit and save estimates
        gf.grid_fit(ecc_grid = eccs,
                    polar_grid = polars,
                    size_grid = sizes,
                    verbose = False,
                    pos_prfs_only = True)
            
        np.savez(grid_estimates_filename,
                 x = gf.gridsearch_params[..., 0],
                 y = gf.gridsearch_params[..., 1],
                 size = gf.gridsearch_params[..., 2],
                 betas = gf.gridsearch_params[...,3],
                 baseline = gf.gridsearch_params[..., 4],
                 r2 = gf.gridsearch_params[..., 5])
    
    else: # if file exists, then load params
        print('%s already exists, loading params from previous fit'%grid_estimates_filename)
        loaded_gf_pars = np.load(grid_estimates_filename)

        gf.gridsearch_params = np.array([loaded_gf_pars[par] for par in ['x', 'y', 'size', 'betas', 'baseline','r2']]) 
        gf.gridsearch_params = np.transpose(gf.gridsearch_params)


    # gaussian iterative fit
    iterative_out = gii_file.replace('.func.gii', '_chunk-%s_of_%s_iterative_gauss_estimates.npz'%(chunk_num,str(total_chunks).zfill(3)))
    iterative_out = os.path.join(out_pth.replace(params['fitting']['prf']['fit_model'],'gauss'),os.path.split(iterative_out)[-1])

    if not os.path.isfile(iterative_out): # if estimates file doesn't exist
        print('doing iterative fit')
        gf.iterative_fit(rsq_threshold = 0.05, 
                        verbose = False, 
                        xtol = xtol,
                        ftol = ftol,
                        bounds = gauss_bounds)
            
        np.savez(iterative_out,
                  x = gf.iterative_search_params[..., 0],
                  y = gf.iterative_search_params[..., 1],
                  size = gf.iterative_search_params[..., 2],
                  betas = gf.iterative_search_params[...,3],
                  baseline = gf.iterative_search_params[..., 4],
                  r2 = gf.iterative_search_params[..., 5])
    else:
        print('%s already exists, loading params from previous fit'%iterative_out)

        loaded_gf_it_pars = np.load(iterative_out)

        gf.iterative_search_params = np.array([loaded_gf_it_pars[par] for par in ['x', 'y', 'size', 'betas', 'baseline','r2']]) 
        gf.iterative_search_params = np.transpose(gf.iterative_search_params)


    # css fitter
    gf_css = CSS_Iso2DGaussianFitter(data = data, gridder = gg_css, n_jobs = 16,
                                     previous_gaussian_fitter = gf)

    iterative_out = gii_file.replace('.func.gii', '_chunk-%s_of_%s_iterative_css_estimates.npz'%(chunk_num,str(total_chunks).zfill(3)))
    iterative_out = os.path.join(out_pth,os.path.split(iterative_out)[-1])
        
    if not os.path.isfile(iterative_out): # if estimates file doesn't exist
        print('doing CSS iterative fit')
        gf_css.iterative_fit(rsq_threshold = 0.05, 
                            verbose = False,
                            xtol = xtol,
                            ftol = ftol,
                            bounds = css_bounds)
            
        np.savez(iterative_out,
                  x = gf_css.iterative_search_params[..., 0],
                  y = gf_css.iterative_search_params[..., 1],
                  size = gf_css.iterative_search_params[..., 2],
                  betas = gf_css.iterative_search_params[...,3],
                  baseline = gf_css.iterative_search_params[..., 4],
                  ns = gf_css.iterative_search_params[..., 5],
                  r2 = gf_css.iterative_search_params[..., 6])
    else:
        print('%s already exists'%iterative_out)
    
    
    
    