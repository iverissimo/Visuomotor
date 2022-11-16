
################################################
#      Do pRF fit on averaged run (median or loo), 
# make iterative fit and save outputs 
################################################


import os
import os.path as op
import yaml, sys

import numpy as np

from nilearn import surface

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds visuomotor_utils.py
from visuomotor_utils import make_prf_dm, save_estimates #import script to use relevante functions


# requires pfpy be installed - preferably with python setup.py develop
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

import datetime

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number and which chunk of data to fit

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add data chunk number to be fitted '
                    'as 2nd argument in the command line!')

elif len(sys.argv) < 4:
    raise NameError('Please add type of run to be fitted (ex: leave_01_out vs median) '
                    'as 3rd argument in the command line!')


else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    chunk_num = str(sys.argv[2]).zfill(3)
    run_type = str(sys.argv[3])

# print start time, for bookeeping
start_time = datetime.datetime.now()

# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# fit hrf?
fit_hrf = params['fitting']['prf']['fit_hrf']

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
postfmriprep_pth = op.join(deriv_pth,'post_fmriprep','prf','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files

# type of fit, if on median run or loo runs
fit_type = params['fitting']['prf']['type']

if params['fitting']['prf']['type'] == 'loo_run':
    # list with left out run path (iff gii, then 2 hemispheres)
    gii_lo_run = [op.join(postfmriprep_pth, h) for h in os.listdir(postfmriprep_pth) if 'run-'+run_type[6:8] in h and 
                  h.endswith(params['fitting']['prf']['extension'])]

# type of model to fit
model_type = params['fitting']['prf']['fit_model']

# output path
out_pth = op.join(deriv_pth,'pRF_fit','sub-{sj}'.format(sj=sj), model_type, fit_type) # path to save estimates
if fit_type == 'loo_run':
    out_pth = op.join(out_pth,run_type) 
 
# check if path to save processed files exist
os.makedirs(out_pth, exist_ok=True) 
if model_type!='gauss':
    os.makedirs(out_pth.replace(model_type,'gauss'), exist_ok=True) #also make gauss dir, to save intermediate estimates

# get list of files to fit
input_file_pth = op.join(postfmriprep_pth,fit_type)

# send message to user
print('fitting functional files from %s'%input_file_pth)

# list with file path to be fitted (iff gii, then 2 hemispheres)
med_gii = [op.join(input_file_pth, h) for h in os.listdir(input_file_pth) if run_type in h]

# fit model per hemisphere
for w,gii_file in enumerate(med_gii):

    ### define filenames for grid and search estimates

    #filename the estimates of the grid fit
    grid_estimates_filename = gii_file.replace('.func.gii', '_chunk-%s_of_%s_gauss_estimates.npz'%(str(chunk_num).zfill(3), str(total_chunks).zfill(3)))
    grid_estimates_filename = op.join(out_pth.replace(params['fitting']['prf']['fit_model'],'gauss'),op.split(grid_estimates_filename)[-1])

    #filename the estimates of the iterative fit
    it_estimates_filename = grid_estimates_filename.replace('gauss', 'iterative_gauss')
    
    os.makedirs(op.split(it_estimates_filename)[0], exist_ok=True) 

    if model_type=='css':
        #filename the estimates of the css fit
        grid_css_estimates_filename = grid_estimates_filename.replace('gauss', 'css')
        os.makedirs(op.split(grid_css_estimates_filename)[0], exist_ok=True) 

        it_css_estimates_filename = grid_estimates_filename.replace('gauss', 'iterative_css')
        os.makedirs(op.split(it_css_estimates_filename)[0], exist_ok=True) 
       
    ### now actually fit the data, if it was not fit before
    
    if op.exists(it_estimates_filename) and model_type!='css': # if iterative fit exists, then gaussian was run
        print('already exists %s'%it_estimates_filename)

    else:
        # load data
        print('loading data from %s' % gii_file)
        data_all = np.array(surface.load_surf_data(gii_file))
        print('data array with shape %s'%str(data_all.shape))
        
        # number of vertices of chunk
        num_vox_chunk = int(data_all.shape[0]/total_chunks) 

        # new data chunk to fit
        data = data_all[num_vox_chunk*(int(chunk_num)-1):num_vox_chunk*int(chunk_num),:]

        print('fitting chunk %s/%d of data with shape %s'%(chunk_num,total_chunks,str(data.shape)))
        
        # define non nan voxels for sanity check
        not_nan_vox = np.where(~np.isnan(data[...,0]))[0]
        print('masked data with shape %s'%(str(data[not_nan_vox].shape)))
        
        # if there are nan voxels
        if len(not_nan_vox)>0:
            # mask them out
            # to avoid errors in fitting (all nan batches) and make fitting faster
            masked_data = data[not_nan_vox]


        if len(not_nan_vox)==0: # if all voxels nan, skip fitting completely 
            estimates_grid = np.zeros((data.shape[0],6)); estimates_grid[:] = np.nan
            estimates_it = np.zeros((data.shape[0],6)); estimates_it[:] = np.nan

        else:
            # subjects that did pRF task with linux computer, so res was full HD
            HD_subs = [str(num).zfill(2) for num in params['general']['HD_screen_subs']] 
            res = params['general']['screenRes_HD'] if str(sj).zfill(2) in HD_subs else params['general']['screenRes']

            ## make prf dm (x, y, t)
            prf_dm = make_prf_dm(res_scaling = .1, screen_res = res, dm_shape = 'square',
                                bar_width_ratio = 0.125, iti = params['fitting']['prf']['PRF_ITI_in_TR'],
                                bar_pass_in_TRs = params['fitting']['prf']['bar_pass_in_TRs'],
                                bar_pass_direction = params['fitting']['prf']['bar_pass_direction'],
                                run_lenght_TR = 90, 
                                crop_nr = params['processing']['crop_pRF_TR'], shift = -1)

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

            # for css
            n_grid = np.linspace(params['fitting']['prf']['min_n'], 
                                 params['fitting']['prf']['max_n'], 
                                    20, dtype='float32')

            # to set up parameter bounds in iterfit
            inf = np.inf
            eps = 1e-1
            ss = prf_stim.screen_size_degrees
            xtol = 1e-4
            ftol = 1e-4

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

            ## if we want to make baseline fixed
            fixed_grid_baseline = None

            # define model 
            gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                            filter_predictions = True,
                                            filter_type = 'dc',
                                            filter_params = {'highpass': True,
                                                            'add_mean': True,
                                                            'window_length': 201,
                                                            'polyorder': 3}
                                        )
        
            ## GRID FIT
            print("Gauss Grid fit")
            gauss_fitter = Iso2DGaussianFitter(data = masked_data, 
                                               model = gauss_model, 
                                               n_jobs = 16,
                                               fit_hrf = fit_hrf)

            gauss_fitter.grid_fit(ecc_grid = eccs, 
                                  polar_grid = polars, 
                                  size_grid = sizes, 
                                  fixed_grid_baseline = fixed_grid_baseline,
                                  grid_bounds = [(0,1000)]) #only prf amplitudes between 0 and 1000


            estimates_grid = gauss_fitter.gridsearch_params
            
            # save grid estimates
            save_estimates(grid_estimates_filename, estimates_grid, 
                           not_nan_vox, orig_num_vert = data.shape[0], model_type = 'gauss')
            
            ## ITERATIVE FIT
            print("Gauss Iterative fit")

            gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                                       verbose = True,
                                       bounds=gauss_bounds,
                                       xtol = xtol,
                                       ftol = ftol)


            estimates_it = gauss_fitter.iterative_search_params
            
            # save iterative estimates
            save_estimates(it_estimates_filename, estimates_it, 
                           not_nan_vox, orig_num_vert = data.shape[0], model_type = 'gauss')

            # cross validate on left out run, if such is the case
            if params['fitting']['prf']['type'] == 'loo_run':
    
                print('loading data from left out run %s' % gii_lo_run[w])
                data_loo = np.array(surface.load_surf_data(gii_lo_run[w]))
                
                print('data array with shape %s'%str(data_loo.shape))
                
                # masked data to be equal size from fitted data
                masked_data_loo = data_loo[not_nan_vox]
                print('masked data loo with shape %s'%(str(masked_data_loo.shape)))
                
                # crossvalidate, just to obtain the CV R2 values
                gauss_fitter.crossvalidate_fit(masked_data_loo)
                
                # save estimates
                estimates_it_CV = gauss_fitter.iterative_search_params
                
                # file path 
                CV_estimates_filename = it_estimates_filename.replace('iterative_gauss', 'CV')
                
                if not op.exists(op.split(CV_estimates_filename)[0]): # check if path to save CV files exist
                    os.makedirs(op.split(CV_estimates_filename)[0]) 
                
                # save iterative estimates
                save_estimates(CV_estimates_filename, estimates_it_CV, 
                            not_nan_vox, orig_num_vert = data.shape[0], model_type = 'gauss')   

            ## NOW DO CSS
            if model_type=='css':

                print("CSS Grid fit")
                css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = 'dc',
                                                    filter_params = {'highpass': True,
                                                                    'add_mean': True,
                                                                    'window_length': 201,
                                                                    'polyorder': 3}
                                                )

                css_fitter = CSS_Iso2DGaussianFitter(data = masked_data, 
                                                    model = css_model, 
                                                    n_jobs = 16,
                                                    fit_hrf = fit_hrf,
                                                    previous_gaussian_fitter = gauss_fitter)

                css_fitter.grid_fit(n_grid,
                                    fixed_grid_baseline = fixed_grid_baseline,
                                    grid_bounds = [(0,1000)], #only prf amplitudes between 0 and 1000
                                    rsq_threshold = 0.05)

                # save grid estimates
                save_estimates(grid_css_estimates_filename, css_fitter.gridsearch_params, 
                            not_nan_vox, orig_num_vert = data.shape[0], model_type = 'css')


                ## ITERATIVE FIT
                print("CSS Iterative fit")

                gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                                        verbose = True,
                                        bounds = css_bounds,
                                        xtol = xtol,
                                        ftol = ftol)

                # save iterative estimates
                save_estimates(it_css_estimates_filename, css_fitter.iterative_search_params, 
                            not_nan_vox, orig_num_vert = data.shape[0], model_type = 'css')

      
            
# Print duration
end_time = datetime.datetime.now()
print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                start_time = start_time,
                end_time = end_time,
                dur  = end_time - start_time))