import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob
from visuomotor_utils import crop_shift_arr

from PIL import Image, ImageOps, ImageDraw

from scipy.optimize import LinearConstraint, NonlinearConstraint

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter

from nilearn import surface
import nibabel as nib
import cortex

class prfModel:

    def __init__(self, MRIObj, outputdir = None):

        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in preproc_mridata
        outputdir: str
            absolute path to save fits
            
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit')
        else:
            self.outputdir = outputdir

        ### some relevant params ###

        # screen specs
        self.screen_res = self.MRIObj.params['general']['screenRes']
        self.screen_res_HD = self.MRIObj.params['general']['screenRes_HD']
        self.screen_dist = self.MRIObj.params['general']['screen_distance']
        self.screen_width = self.MRIObj.params['general']['screen_width']

        # screen resolution changed for a few subs, save in var
        self.HD_screen_sj = [str(s).zfill(2) for s in self.MRIObj.params['general']['HD_screen_subs']]

        # type of model to fit
        self.model_type = self.MRIObj.params['fitting']['prf']['fit_model']

        # type of optimizer to use
        self.optimizer = self.MRIObj.params['fitting']['prf']['optimizer']

        # if we are fitting HRF params
        self.fit_hrf = self.MRIObj.params['fitting']['prf']['fit_hrf']

        # processed file extension
        self.proc_file_ext = self.MRIObj.params['fitting']['prf']['extension']

        # total number of chunks we divide data when fitting
        self.total_chunks = self.MRIObj.params['fitting']['prf']['total_chunks']

        # bar width ratio
        self.bar_width_ratio = 0.125
        self.hrf_onset = 0

    def get_prf_file_list(self, participant, file_ext = 'cropped_dc_psc.func.gii'):

        """
        Helper function to get list of bold file names
        to then be loaded and used

        Parameters
        ----------
        participant: str
            participant ID
        """

        ## get list of possible input paths
        # (sessions)
        input_list = op.join(self.MRIObj.postfmriprep_pth, self.MRIObj.sj_space, 'prf', 'sub-{sj}'.format(sj = participant))

        # list with absolute file names to be fitted
        bold_filelist = [op.join(input_list, file) for file in os.listdir(input_list) if file.endswith(file_ext)]

        return bold_filelist

    def get_run_list(self, file_list):

        """
        Helper function to get unique run number from list of strings (filenames)

        Parameters
        ----------
        file_list: list
            list with file names
        """
        return np.unique([int(re.findall(r'run-\d{1,3}', op.split(input_name)[-1])[0][4:]) for input_name in file_list])
   

    def load_data4fitting(self, file_list):

        """
        Helper function to load data for fitting

        Parameters
        ----------
        file_list: list
            list with file names
        """

        # get run IDs
        run_num_list = self.get_run_list(file_list)

        ## load data of all runs
        # will be [runs, vertex, TR]

        all_data = []
        for run_id in run_num_list:
            
            run_data = []
            for hemi in self.MRIObj.hemispheres:
                
                hemi_file = [file for file in file_list if 'run-{r}'.format(r=str(run_id).zfill(2)) in file and hemi in file][0]
                print('loading %s' %hemi_file)    
                run_data.append(np.array(surface.load_surf_data(hemi_file)))
                
            all_data.append(np.vstack(run_data)) # will be (vertex, TR)

        return all_data


    def make_dm(self, participant, run_length_TR = 90, crop_nr = None, shift = 0, 
                        res_scaling = .1, dm_shape = 'square'):

        """ 
        Make design matrix array for pRF task

        Parameters
        ----------
        participant : str
            participant number
        res_scaling: float
            spatial rescaling factor
        run_length_TR: int
            total run length (without cropping)
        crop_nr: int
            number of TRs to crop
        shift: int
            number of TRs to shift onset
        dm_shape: str
            if we want square DM or not
        """

        # set screen resolution
        if str(participant).zfill(2) in self.HD_screen_sj:
            screen_res = self.screen_res_HD
        else:
            screen_res = self.screen_res

        # get array of bar condition label per TR
        condition_per_TR = []
        for cond in self.MRIObj.pRF_bar_pass_direction:
            condition_per_TR += list(np.tile(cond, self.MRIObj.pRF_bar_pass_in_TRs[cond]))
            
            if np.ceil(self.MRIObj.pRF_ITI_in_TR)>0: # if ITI in TR, 
                condition_per_TR += list(np.tile('empty', int(np.ceil(self.MRIObj.pRF_ITI_in_TR))))
            
        # drop last TRs, for DM to have same time length as data
        condition_per_TR = condition_per_TR[:run_length_TR]

        ## crop and shift if such was the case
        condition_per_TR = crop_shift_arr(np.array(condition_per_TR)[np.newaxis], 
                                            crop_nr = crop_nr, shift = shift)[0]

        # all possible positions in pixels for for midpoint of
        # y position for vertical bar passes, 
        ver_y = screen_res[1]*np.linspace(0,1, self.MRIObj.pRF_bar_pass_in_TRs['U-D'])#+1)
        # x position for horizontal bar passes 
        hor_x = screen_res[0]*np.linspace(0,1, self.MRIObj.pRF_bar_pass_in_TRs['L-R'])#+1)

        # coordenates for bar pass, for PIL Image
        coordenates_bars = {'L-R': {'upLx': hor_x - 0.5 * self.bar_width_ratio * screen_res[0], 
                                    'upLy': np.repeat(screen_res[1], self.MRIObj.pRF_bar_pass_in_TRs['L-R']),
                                    'lowRx': hor_x + 0.5 * self.bar_width_ratio * screen_res[0], 
                                    'lowRy': np.repeat(0, self.MRIObj.pRF_bar_pass_in_TRs['L-R'])},
                            'R-L': {'upLx': np.array(list(reversed(hor_x - 0.5 * self.bar_width_ratio * screen_res[0]))), 
                                    'upLy': np.repeat(screen_res[1], self.MRIObj.pRF_bar_pass_in_TRs['R-L']),
                                    'lowRx': np.array(list(reversed(hor_x+ 0.5 * self.bar_width_ratio * screen_res[0]))), 
                                    'lowRy': np.repeat(0, self.MRIObj.pRF_bar_pass_in_TRs['R-L'])},
                            'U-D': {'upLx': np.repeat(0, self.MRIObj.pRF_bar_pass_in_TRs['U-D']), 
                                    'upLy': ver_y+0.5 * self.bar_width_ratio * screen_res[1],
                                    'lowRx': np.repeat(screen_res[0], self.MRIObj.pRF_bar_pass_in_TRs['U-D']), 
                                    'lowRy': ver_y - 0.5 * self.bar_width_ratio * screen_res[1]},
                            'D-U': {'upLx': np.repeat(0, self.MRIObj.pRF_bar_pass_in_TRs['D-U']), 
                                    'upLy': np.array(list(reversed(ver_y + 0.5 * self.bar_width_ratio * screen_res[1]))),
                                    'lowRx': np.repeat(screen_res[0], self.MRIObj.pRF_bar_pass_in_TRs['D-U']), 
                                    'lowRy': np.array(list(reversed(ver_y - 0.5 * self.bar_width_ratio * screen_res[1])))}
                            }

        # save screen display for each TR 
        visual_dm_array = np.zeros((len(condition_per_TR), round(screen_res[1] * res_scaling), round(screen_res[0] * res_scaling)))
        i = 0

        for trl, bartype in enumerate(condition_per_TR): # loop over bar pass directions

            img = Image.new('RGB', tuple(screen_res)) # background image

            if bartype not in np.array(['empty','empty_long']): # if not empty screen

                #print(bartype)

                # set draw method for image
                draw = ImageDraw.Draw(img)
                # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                draw.rectangle(tuple([coordenates_bars[bartype]['upLx'][i],coordenates_bars[bartype]['upLy'][i],
                                    coordenates_bars[bartype]['lowRx'][i],coordenates_bars[bartype]['lowRy'][i]]), 
                            fill = (255,255,255),
                            outline = (255,255,255))

                # increment counter
                if trl < (len(condition_per_TR) - 1):
                    i = i+1 if condition_per_TR[trl] == condition_per_TR[trl+1] else 0    

            ## save in array
            visual_dm_array[int(trl):int(trl + 1), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0])

        if dm_shape == 'square':
            ## make it square
            # add padding (top and bottom borders) 
            new_visual_dm = np.zeros((round(np.max(screen_res) * res_scaling), round(np.max(screen_res) * res_scaling),
                                    len(condition_per_TR)))

            pad_ind = int(np.ceil((screen_res[0] - screen_res[1])/2 * res_scaling))
            new_visual_dm[pad_ind:int(visual_dm.shape[0]+pad_ind),:,:] = visual_dm.copy()
        else:
            new_visual_dm = visual_dm.copy()
            
        return new_visual_dm


    def set_models(self, participant_list = []):

        """
        define pRF models to be used for each participant in participant list
                
        Parameters
        ----------
        participant_list: list
            list with participant ID
        """  

        ## loop over participants

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num
        
        # empty dict where we'll store all participant models
        pp_models = {}
        
        for pp in participant_list:

            pp_models['sub-{sj}'.format(sj=pp)] = {}
            
            # for each participant, save DM and models
            visual_dm = self.make_dm(pp, run_length_TR = 90, 
                                        crop_nr = self.MRIObj.params['processing']['crop_TR']['prf'], 
                                        shift = 0, res_scaling = .1, dm_shape = 'square')

            # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
            prf_stim = PRFStimulus2D(screen_size_cm = self.screen_width,
                                    screen_distance_cm = self.screen_dist,
                                    design_matrix = visual_dm,
                                    TR = self.MRIObj.TR)

            pp_models['sub-{sj}'.format(sj=pp)]['prf_stim'] = prf_stim     

            ## define models ##
            # GAUSS
            gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['processing']['filter']['prf'],
                                                filter_params = {'highpass': True,
                                                                'add_mean': True,
                                                                'first_modes_to_remove': self.MRIObj.params['processing']['first_modes_to_remove'],
                                                                'window_length': 201,
                                                                'polyorder': 3},
                                                hrf_onset = self.hrf_onset,
                                            )

            pp_models['sub-{sj}'.format(sj=pp)]['gauss_model'] = gauss_model

            # CSS
            css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['processing']['filter']['prf'],
                                                filter_params = {'highpass': True,
                                                                'add_mean': True,
                                                                'first_modes_to_remove': self.MRIObj.params['processing']['first_modes_to_remove'],
                                                                'window_length': 201,
                                                                'polyorder': 3},
                                                hrf_onset = self.hrf_onset,
                                            )

            pp_models['sub-{sj}'.format(sj=pp)]['css_model'] = css_model

            # DN 
            dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['processing']['filter']['prf'],
                                                filter_params = {'highpass': True,
                                                                'add_mean': True,
                                                                'first_modes_to_remove': self.MRIObj.params['processing']['first_modes_to_remove'],
                                                                'window_length': 201,
                                                                'polyorder': 3},
                                                hrf_onset = self.hrf_onset,
                                            )

            pp_models['sub-{sj}'.format(sj=pp)]['dn_model'] = dn_model

            # DOG
            dog_model = DoG_Iso2DGaussianModel(stimulus = prf_stim,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['processing']['filter']['prf'],
                                                filter_params = {'highpass': True,
                                                                'add_mean': True,
                                                                'first_modes_to_remove': self.MRIObj.params['processing']['first_modes_to_remove'],
                                                                'window_length': 201,
                                                                'polyorder': 3},
                                                hrf_onset = self.hrf_onset,
                                            )

            pp_models['sub-{sj}'.format(sj=pp)]['dog_model'] = dog_model

        return pp_models


    def get_fit_startparams(self, max_ecc_size = 6, fix_bold_baseline = False):

        """
        Helper function that loads all fitting starting params
        and bounds into a dictionary
        Parameters
        ----------
        max_ecc_size: int/float
            max eccentricity (and size) to set grid array
        """

        eps = 1e-1

        fitpar_dict = {'gauss': {}, 'css': {}, 'dn': {}, 'dog': {}}

        ######################### GAUSS #########################

        ## number of grid points 
        fitpar_dict['gauss']['grid_nr'] = self.MRIObj.params['fitting']['prf']['grid_nr']

        # size, ecc, polar angle
        fitpar_dict['gauss']['sizes'] = max_ecc_size * np.linspace(0.25, 1, fitpar_dict['gauss']['grid_nr'])**2
        fitpar_dict['gauss']['eccs'] = max_ecc_size * np.linspace(0.1, 1, fitpar_dict['gauss']['grid_nr'])**2
        fitpar_dict['gauss']['polars'] = np.linspace(0, 2*np.pi, fitpar_dict['gauss']['grid_nr'])

        ## bounds
        fitpar_dict['gauss']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000)] # bold baseline
        
        fitpar_dict['gauss']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['gauss']['grid_bounds'] = [(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### CSS #########################

        ## grid exponent parameter
        fitpar_dict['css']['n_grid'] = np.linspace(self.MRIObj.params['fitting']['prf']['min_n'], 
                                                   self.MRIObj.params['fitting']['prf']['max_n'], 
                                                   self.MRIObj.params['fitting']['prf']['n_nr'], dtype='float32')

        ## bounds
        fitpar_dict['css']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0.01, 1.1)]  # CSS exponent

        fitpar_dict['css']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['css']['grid_bounds'] = [(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### DN #########################

        # Surround amplitude (Normalization parameter C)
        fitpar_dict['dn']['surround_amplitude_grid'] = np.array([0.1,0.2,0.4,0.7,1,3], dtype='float32') 
        
        # Surround size (gauss sigma_2)
        fitpar_dict['dn']['surround_size_grid'] = np.array([3,5,8,12,18], dtype='float32')
        
        # Neural baseline (Normalization parameter B)
        fitpar_dict['dn']['neural_baseline_grid'] = np.array([0,1,10,100], dtype='float32')

        # Surround baseline (Normalization parameter D)
        fitpar_dict['dn']['surround_baseline_grid'] = np.array([0.1,1.0,10.0,100.0], dtype='float32')

        ## bounds
        fitpar_dict['dn']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0, 1000),  # surround amplitude
                                        (eps, 3 * (max_ecc_size * 2)),  # surround size
                                        (0, 1000),  # neural baseline
                                        (1e-6, 1000)]  # surround baseline

        fitpar_dict['dn']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['dn']['grid_bounds'] = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### DOG #########################
        
        # amplitude for surround 
        fitpar_dict['dog']['surround_amplitude_grid'] = np.array([0.05,0.1,0.25,0.5,0.75,1,2], dtype='float32')

        # size for surround
        fitpar_dict['dog']['surround_size_grid'] = np.array([3,5,8,11,14,17,20,23,26], dtype='float32')

        ## bounds
        fitpar_dict['dog']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0, 1000),  # surround amplitude
                                        (eps, 3 * (max_ecc_size * 2))]  # surround size

        fitpar_dict['dog']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['dog']['grid_bounds'] = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000

        ### EXTRA ###

        # if we want to also fit hrf
        if self.fit_hrf:
            fitpar_dict['gauss']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['css']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['dn']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['dog']['bounds'] += [(0,10),(0,0)]
        
        # if we want to keep the baseline fixed at 0
        if fix_bold_baseline:
            fitpar_dict['gauss']['bounds'][4] = (0,0)
            fitpar_dict['gauss']['fixed_grid_baseline'] = 0 
            
            fitpar_dict['css']['bounds'][4] = (0,0)
            fitpar_dict['css']['fixed_grid_baseline'] = 0 

            fitpar_dict['dn']['bounds'][4] = (0,0)
            fitpar_dict['dn']['fixed_grid_baseline'] = 0 

            fitpar_dict['dog']['bounds'][4] = (0,0)
            fitpar_dict['dog']['fixed_grid_baseline'] = 0 

                                        
        return fitpar_dict

    def get_fit_constraints(self, method = 'L-BFGS-B', ss_larger_than_centre = True, 
                        positive_centre_only = False, normalize_RFs = False):

        """
        Helper function sets constraints - which depend on minimizer used -
        for all model types and saves in dictionary
        Parameters
        ----------
        method: str
            minimizer that we want to use, ex: 'L-BFGS-B', 'trust-constr'
        """

        constraints = {'gauss': {}, 'css': {}, 'dn': {}, 'dog': {}}

        for key in constraints.keys():

            if method == 'L-BFGS-B':
                
                constraints[key] = None
            
            elif method == 'trust-constr':
                
                constraints[key] = []
                
                if 'dn' in key:
                    if ss_larger_than_centre:
                        #enforcing surround size larger than prf size
                        if self.fit_hrf:
                            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
                        else:
                            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0]])
                            
                        constraints[key].append(LinearConstraint(A_ssc_norm,
                                                lb=0,
                                                ub=np.inf))
                        
                    if positive_centre_only:
                        #enforcing positive central amplitude in norm
                        def positive_centre_prf_norm(x):
                            if normalize_RFs:
                                return (x[3]/(2*np.pi*x[2]**2)+x[7])/(x[5]/(2*np.pi*x[6]**2)+x[8]) - x[7]/x[8]
                            else:
                                return (x[3]+x[7])/(x[5]+x[8]) - x[7]/x[8]

                        constraints[key].append(NonlinearConstraint(positive_centre_prf_norm,
                                                                    lb=0,
                                                                    ub=np.inf))
                elif 'dog' in key:
                    if ss_larger_than_centre:
                        #enforcing surround size larger than prf size
                        if self.fit_hrf:
                             A_ssc_dog = np.array([[0,0,-1,0,0,0,1,0,0]])
                        else:
                            A_ssc_dog = np.array([[0,0,-1,0,0,0,1]])
                            
                        constraints[key].append(LinearConstraint(A_ssc_dog,
                                                lb=0,
                                                ub=np.inf))
                        
                    if positive_centre_only:
                        #enforcing positive central amplitude in DoG
                        def positive_centre_prf_dog(x):
                            if normalize_RFs:
                                return x[3]/(2*np.pi*x[2]**2)-x[5]/(2*np.pi*x[6]**2)
                            else:
                                return x[3] - x[5]

                        constraints[key].append(NonlinearConstraint(positive_centre_prf_dog,
                                                                    lb=0,
                                                                    ub=np.inf))

        return constraints


    def fit_data(self, participant, pp_prf_models = None, 
                    fit_type = 'mean_run', chunk_num = None, vertex = None, ROI = None,
                    model2fit = 'gauss', outdir = None, save_estimates = False,
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16):

        """
        fit inputted pRF models to each participant in participant list
                
        Parameters
        ----------
        participant: str
            participant ID
        input_pth: str or None
            path to look for files, if None then will get them from derivatives/postfmriprep/<space>/sub-X folder
        fit_type: string 
            type of run to fit, mean_run (default), loo_run
        """  

        # get participant models
        if pp_prf_models is None:
            pp_prf_models = self.set_models(participant_list = [participant])

        ## set model parameters 
        # relevant for grid and iterative fitting
        fit_params = self.get_fit_startparams(max_ecc_size = pp_prf_models['sub-{sj}'.format(sj = participant)]['prf_stim'].screen_size_degrees/2.0)

        ## set constraints
        # for now only changes minimizer used, but can also be useful to put contraints on dog and dn
        constraints = self.get_fit_constraints(method = self.optimizer, ss_larger_than_centre = True, 
                                                positive_centre_only = True, normalize_RFs = False)

        ## LOAD RUN DATA
        # get list with gii files
        gii_filenames = self.get_prf_file_list(participant, 
                                            file_ext = self.MRIObj.params['fitting']['prf']['extension'])

        if fit_type == 'mean_run':         
            # load data of all runs
            all_data = self.load_data4fitting(gii_filenames) # [runs, vertex, TR]

            # average runs
            data2fit = np.nanmean(all_data, axis = 0)[np.newaxis,...]

        elif fit_type == 'loo_run':
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            # leave one run out, load other and average
            data2fit = []

            for lo_run_key in run_loo_list:
                print('Leaving run-{r} out'.format(r = str(lo_run_key).zfill(2)))

                all_data = self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(lo_run_key).zfill(2)) not in file])
                data2fit.append(np.nanmean(all_data, axis = 0)[np.newaxis,...])

            data2fit = np.vstack(data2fit)

        ## loop over runs
        for r_ind in range(data2fit.shape[0]):

            # set output directory
            if outdir is None:
                if fit_type == 'loo_run':
                    outdir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), fit_type, 
                                                    'run-{r}'.format(r = str(run_loo_list[r_ind]).zfill(2)))
                else:
                    outdir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), fit_type)

            os.makedirs(outdir, exist_ok = True)
            print('saving files in %s'%outdir)

            ## set base filename that will be used for estimates
            basefilename = 'sub-{sj}_task-pRF'.format(sj = participant)
            if chunk_num is not None:
                basefilename += '_chunk-{ch}'.format(ch = str(chunk_num).zfill(3))
            elif vertex is not None:
                basefilename += '_vertex-{ver}'.format(ver = str(vertex))
            elif ROI:
                basefilename += '_ROI-{roi}'.format(roi = str(ROI))
            
            basefilename += ('_'+self.MRIObj.params['fitting']['prf']['extension'].replace('.func.gii', '.npz'))

            ## chunk data
            chunk2fit = self.chunk_data(data2fit[r_ind], chunk_num = chunk_num, vertex = vertex, ROI = ROI)

            ## ACTUALLY FIT 

            # always start with gauss of course
            grid_gauss_filename = op.join(outdir, 'grid_gauss', basefilename.replace('.npz', '_grid_gauss_estimates.npz'))
            it_gauss_filename = op.join(outdir, 'it_gauss', basefilename.replace('.npz', '_it_gauss_estimates.npz'))

            # if we want to fit hrf, change output name
            if self.fit_hrf:
                grid_gauss_filename = grid_gauss_filename.replace('_estimates.npz', '_HRF_estimates.npz')
                it_gauss_filename = it_gauss_filename.replace('_estimates.npz', '_HRF_estimates.npz')

            # already set other model name
            grid_model_filename = grid_gauss_filename.replace('gauss', model2fit)
            it_model_filename = it_gauss_filename.replace('gauss', model2fit)

            if not op.isfile(it_model_filename):
                print("Gauss model GRID fit")
                gauss_fitter = Iso2DGaussianFitter(data = chunk2fit, 
                                                    model = pp_prf_models['sub-{sj}'.format(sj = participant)]['gauss_model'], 
                                                    n_jobs = n_jobs,
                                                    fit_hrf = self.fit_hrf)

                gauss_fitter.grid_fit(ecc_grid = fit_params['gauss']['eccs'], 
                                        polar_grid = fit_params['gauss']['polars'], 
                                        size_grid = fit_params['gauss']['sizes'], 
                                        fixed_grid_baseline = fit_params['gauss']['fixed_grid_baseline'],
                                        grid_bounds = fit_params['gauss']['grid_bounds'])

                # iterative fit
                print("Gauss model ITERATIVE fit")
                
                gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                                        verbose = True,
                                        bounds = fit_params['gauss']['bounds'],
                                        constraints = constraints['gauss'],
                                        #starting_params = gauss_fitter.gridsearch_params,
                                        xtol = xtol,
                                        ftol = ftol)

                # if we want to save estimates
                if save_estimates and not op.isfile(it_gauss_filename):
                    # for grid
                    print('saving %s'%grid_gauss_filename)
                    self.save_pRF_model_estimates(grid_gauss_filename, gauss_fitter.gridsearch_params, 
                                                    model_type = 'gauss', grid = True) 
                    # for it
                    print('saving %s'%it_gauss_filename)
                    self.save_pRF_model_estimates(it_gauss_filename, gauss_fitter.iterative_search_params, 
                                                    model_type = 'gauss')

                if model2fit != 'gauss':

                    print("{key} model GRID fit".format(key = model2fit))
                    
                    if model2fit == 'css':

                        fitter = CSS_Iso2DGaussianFitter(data = chunk2fit, 
                                                        model = pp_prf_models['sub-{sj}'.format(sj = participant)]['{key}_model'.format(key = model2fit)], 
                                                        n_jobs = n_jobs,
                                                        fit_hrf = self.fit_hrf,
                                                        previous_gaussian_fitter = gauss_fitter)

                        fitter.grid_fit(fit_params['css']['n_grid'],
                                    fixed_grid_baseline = fit_params['css']['fixed_grid_baseline'],
                                    grid_bounds = fit_params['css']['grid_bounds'],
                                    rsq_threshold = 0.05)
                    
                    elif model2fit == 'dn':

                        fitter = Norm_Iso2DGaussianFitter(data = chunk2fit, 
                                                        model = pp_prf_models['sub-{sj}'.format(sj = participant)]['{key}_model'.format(key = model2fit)], 
                                                        n_jobs = n_jobs,
                                                        fit_hrf = self.fit_hrf,
                                                        previous_gaussian_fitter = gauss_fitter)

                        fitter.grid_fit(fit_params['dn']['surround_amplitude_grid'],
                                        fit_params['dn']['surround_size_grid'],
                                        fit_params['dn']['neural_baseline_grid'],
                                        fit_params['dn']['surround_baseline_grid'],
                                    fixed_grid_baseline = fit_params['dn']['fixed_grid_baseline'],
                                    grid_bounds = fit_params['dn']['grid_bounds'],
                                    rsq_threshold = 0.05)

                    
                    elif model2fit == 'dog':

                        fitter = DoG_Iso2DGaussianFitter(data = chunk2fit, 
                                                        model = pp_prf_models['sub-{sj}'.format(sj = participant)]['{key}_model'.format(key = model2fit)], 
                                                        n_jobs = n_jobs,
                                                        fit_hrf = self.fit_hrf,
                                                        previous_gaussian_fitter = gauss_fitter)

                        fitter.grid_fit(fit_params['dog']['surround_amplitude_grid'],
                                        fit_params['dog']['surround_size_grid'],
                                    fixed_grid_baseline = fit_params['dog']['fixed_grid_baseline'],
                                    grid_bounds = fit_params['dog']['grid_bounds'],
                                    rsq_threshold = 0.05)


                    # iterative fit
                    print("{key} model ITERATIVE fit".format(key = model2fit))

                    fitter.iterative_fit(rsq_threshold = 0.05, 
                                        verbose = True,
                                        bounds = fit_params[model2fit]['bounds'],
                                        constraints = constraints[model2fit],
                                        #starting_params = fitter.gridsearch_params, 
                                        xtol = xtol,
                                        ftol = ftol)

                    # if we want to save estimates
                    if save_estimates:
                        # for grid
                        print('saving %s'%grid_model_filename)
                        self.save_pRF_model_estimates(grid_model_filename, fitter.gridsearch_params, 
                                                        model_type = model2fit, grid = True)
                        # for it
                        print('saving %s'%it_model_filename)
                        self.save_pRF_model_estimates(it_model_filename, fitter.iterative_search_params, 
                                                        model_type = model2fit)

        if not save_estimates:
            # if we're not saving them, assume we are running on the spot
            # and want to get back the estimates
            estimates = {}
            estimates['grid_gauss'] = gauss_fitter.gridsearch_params
            estimates['it_gauss'] = gauss_fitter.iterative_search_params
            if model2fit != 'gauss':
                estimates['grid_{key}'.format(key = model2fit)] = fitter.gridsearch_params
                estimates['it_{key}'.format(key = model2fit)] = fitter.iterative_search_params

            return estimates, chunk2fit

    
    def save_pRF_model_estimates(self, filename, final_estimates, model_type = 'gauss', grid = False):
    
        """
        re-arrange estimates that were masked
        and save all in numpy file
        
        (only works for gii files, should generalize for nii and cifti also)
        
        Parameters
        ----------
        filename : str
            absolute filename of estimates to be saved
        final_estimates : arr
            2d estimates (datapoints,estimates)
        model_type: str
            model type used for fitting
        
        """ 

        # make dir if it doesnt exist already
        os.makedirs(op.split(filename)[0], exist_ok = True)
                
        if model_type == 'gauss':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        hrf_derivative = final_estimates[..., 5],
                        hrf_dispersion = final_estimates[..., 6], 
                        r2 = final_estimates[..., 7])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        r2 = final_estimates[..., 5])
        
        elif model_type == 'css':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        ns = final_estimates[..., 5],
                        hrf_derivative = final_estimates[..., 6],
                        hrf_dispersion = final_estimates[..., 7], 
                        r2 = final_estimates[..., 8])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        ns = final_estimates[..., 5],
                        r2 = final_estimates[..., 6])

        elif model_type == 'dn':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        nb = final_estimates[..., 7], 
                        sb = final_estimates[..., 8], 
                        hrf_derivative = final_estimates[..., 9],
                        hrf_dispersion = final_estimates[..., 10], 
                        r2 = final_estimates[..., 11])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        nb = final_estimates[..., 7], 
                        sb = final_estimates[..., 8], 
                        r2 = final_estimates[..., 9])

        elif model_type == 'dog':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        hrf_derivative = final_estimates[..., 7],
                        hrf_dispersion = final_estimates[..., 8], 
                        r2 = final_estimates[..., 9])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        r2 = final_estimates[..., 7])

    
    def chunk_data(self, data, chunk_num = None, vertex = None, ROI = None):

        """
        Helper function to chunk data into subset that will be fitted
        require 2D data [vertex, TR]
        """
        if ROI is not None:
            roi_ind = cortex.get_roi_verts(self.MRIObj.params['processing']['space'], ROI)
            out_data = data[roi_ind[ROI]][vertex][np.newaxis, ...] if vertex is not None else data[roi_ind[ROI]]
        
        elif vertex is not None:
            out_data = data[vertex][np.newaxis, ...] 

        elif chunk_num is not None:
            # number of vertices of chunk
            num_vox_chunk = int(data.shape[0]/self.total_chunks)

            # chunk it
            out_data = data[num_vox_chunk * int(chunk_num):num_vox_chunk * int(chunk_num + 1), :]

        return out_data


    def load_pRF_model_chunks(self, fit_path, fit_model = 'css', fit_hrf = False, basefilename = None, overwrite = False, iterative = True):

        """ 
        combine all chunks 
        into one single estimate numpy array
        assumes input is whole brain ("vertex", time)
        Parameters
        ----------
        fit_path : str
            absolute path to files
        fit_model: str
            fit model of estimates
        fit_hrf: bool
            if we fitted hrf or not
        
        Outputs
        -------
        estimates : npz 
            numpy array of estimates
        
        """

        # if we are fitting HRF, then we want to look for those files
        if fit_hrf:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-000' in x and 'HRF' in x]
        else:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-000' in x and 'HRF' not in x]
        
        ## if we defined a base filename that should be used to fish out right estimates
        if basefilename:
            filename = [file for file in filename_list if basefilename in file][0]
        else:
            filename = filename_list[0]
        
        filename = filename.replace('_chunk-000', '')

        if not op.exists(filename) or overwrite:
        
            for ch in np.arange(self.total_chunks['pRF']):
                
                # if we are fitting HRF, then we want to look for those files
                if fit_hrf:
                    chunk_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-%s'%str(ch).zfill(3) in x and 'HRF' in x]
                else:
                    chunk_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-%s'%str(ch).zfill(3) in x and 'HRF' not in x]
                
                ## if we defined a base filename that should be used to fish out right estimates
                if basefilename:
                    chunk_name = [file for file in chunk_name_list if basefilename in file][0]
                else:
                    chunk_name = chunk_name_list[0]

                print('loading chunk %s'%chunk_name)
                chunk = np.load(chunk_name) # load chunk
                
                if ch == 0:
                    xx = chunk['x']
                    yy = chunk['y']

                    size = chunk['size']

                    beta = chunk['betas']
                    baseline = chunk['baseline']

                    if 'css' in fit_model: 
                        ns = chunk['ns']
                    elif fit_model in ['dn', 'dog']:
                        sa = chunk['sa']
                        ss = chunk['ss']
                    
                    if 'dn' in fit_model:
                        nb = chunk['nb']
                        sb = chunk['sb']

                    rsq = chunk['r2']

                    if fit_hrf and iterative:
                        hrf_derivative = chunk['hrf_derivative']
                        hrf_dispersion = chunk['hrf_dispersion']
                    else: # assumes standard spm params
                        hrf_derivative = np.ones(xx.shape)
                        hrf_dispersion = np.zeros(xx.shape) 

                else:
                    xx = np.concatenate((xx, chunk['x']))
                    yy = np.concatenate((yy, chunk['y']))

                    size = np.concatenate((size, chunk['size']))

                    beta = np.concatenate((beta, chunk['betas']))
                    baseline = np.concatenate((baseline, chunk['baseline']))

                    if 'css' in fit_model:
                        ns = np.concatenate((ns, chunk['ns']))
                    elif fit_model in ['dn', 'dog']:
                        sa = np.concatenate((sa, chunk['sa']))
                        ss = np.concatenate((ss, chunk['ss']))

                    if 'dn' in fit_model:
                        nb = np.concatenate((nb, chunk['nb']))
                        sb = np.concatenate((sb, chunk['sb']))

                    rsq = np.concatenate((rsq, chunk['r2']))
                    
                    if fit_hrf and iterative:
                        hrf_derivative = np.concatenate((hrf_derivative, chunk['hrf_derivative']))
                        hrf_dispersion = np.concatenate((hrf_dispersion, chunk['hrf_dispersion']))
                    else: # assumes standard spm params
                        hrf_derivative = np.concatenate((hrf_derivative, np.ones(xx.shape)))
                        hrf_dispersion = np.concatenate((hrf_dispersion, np.zeros(xx.shape))) 
            
            print('shape of estimates is %s'%(str(xx.shape)))

            # save file
            print('saving %s'%filename)

            if 'gauss' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'css' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        ns = ns,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'dn' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        sa = sa,
                        ss = ss,
                        nb = nb,
                        sb = sb,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'dog' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        sa = sa,
                        ss = ss,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)
            
        else:
            print('file already exists, loading %s'%filename)
        
        return np.load(filename)
    

               