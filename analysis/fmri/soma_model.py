import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob
from nilearn import surface

import datetime

from glmsingle.glmsingle import GLM_single
from glmsingle.ols.make_poly_matrix import make_polynomial_matrix, make_projection_matrix

import scipy

import cortex

class somaModel:

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
        self.outputdir = outputdir

        ### some relevant params ###

        # processed file extension
        self.proc_file_ext = self.MRIObj.params['fitting']['soma']['extension']

        # get trial condition labels, in order 
        self.soma_stim_labels = [op.splitext(val)[0] for val in self.MRIObj.params['fitting']['soma']['soma_stimulus']]
        
        # get unique labels of conditions
        _, idx = np.unique(self.soma_stim_labels, return_index=True) 
        self.soma_cond_unique = np.array(self.soma_stim_labels)[np.sort(idx)]


    def get_soma_file_list(self, participant, file_ext = 'sg_psc.func.gii'):

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
        input_list = op.join(self.MRIObj.postfmriprep_pth, self.MRIObj.sj_space, 'soma', 'sub-{sj}'.format(sj = participant))

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


class GLMsingle_Model(somaModel):

    def __init__(self, MRIObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str or None
            path to general output directory            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir)

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, 'glmsingle_fits')
        else:
            self.outputdir = outputdir

        # some options to feed into glm single
        self.glm_single_ops = self.MRIObj.params['fitting']['soma']['glm_single_ops']

    
    def get_dm_glmsing(self, nr_TRs = 141, nr_runs = 1):

        """
        make glmsingle DM for all runs
        
        Parameters
        ----------
        nr_TRs: int
            number of TRs in run          
        """

        ## get trial timings, in TR
        # initial baseline period - NOTE!! rounding up, will compensate by shifting hrf onset
        start_baseline_dur = int(np.round(self.MRIObj.soma_event_time_in_sec['empty']/self.MRIObj.TR)) 
        
        # trial duration (including ITIs)
        trial_dur = sum([self.MRIObj.soma_event_time_in_sec[name] for name in self.MRIObj.soma_trial_order])/self.MRIObj.TR

        ## define DM [TR, conditions]
        # initialize at 0
        design_array = np.zeros((nr_TRs, len(self.soma_cond_unique)))

        # fill it with ones on stim onset
        for t, trl_cond in enumerate(self.soma_stim_labels):
            
            # index for condition column
            cond_ind = np.where(self.soma_cond_unique == trl_cond)[0][0]
            
            # fill it 
            design_array[int(start_baseline_dur + t*trial_dur), cond_ind] = 1
            #print(int(start_baseline_dur + t*trial_dur))
            
        # and stack it for each run
        all_dm = []
        for r in np.arange(nr_runs):
            all_dm.append(design_array)

        return all_dm


    def average_betas_per_cond(self, single_trial_betas):

        """
        average obtained beta values 
        for each condition and run
        
        Parameters
        ----------
        single_trial_betas: arr
            single trial betas for all runs [vertex, betas*runs]         
        """

        # number of single trials
        nr_sing_trial = len(self.soma_stim_labels)

        # number of runs fitted
        num_runs = int(single_trial_betas.shape[-1]/nr_sing_trial)

        # where we store condition betas [runs, vertex, cond]
        avg_betas_cond = np.zeros((num_runs, single_trial_betas.shape[0], len(self.soma_cond_unique)))

        # iterate over runs
        for run_ind in np.arange(num_runs): 

            # beta values for run
            betas_run = single_trial_betas[..., 
                                            int(run_ind * nr_sing_trial):int(nr_sing_trial + run_ind * nr_sing_trial)]

            # for each unique condition
            for i, cond_name in enumerate(self.soma_cond_unique):

                # find indices where condition happens in trial
                ind_c = [ind for ind, name in enumerate(self.soma_stim_labels) if name == cond_name]

                # average beta values for that condition, and store
                avg_beta = np.mean(betas_run[..., ind_c], axis = -1)
                avg_betas_cond[run_ind,:, i] = avg_beta

        return avg_betas_cond

    def get_nuisance_matrix(self, maxpolydeg, pcregressors = None, pcnum = 1, nr_TRs = 141):

        """
        construct projection matrices 
        for the nuisance components
        
        Parameters
        ----------
        maxpolydeg: list
            list with ints which represent 
            polynomial degree to use for polynomial nuisance functions [runs, poly] 
        pcregressors: list
            list with pcregressors to add to nuisance matrix [runs, TR, nr_pcregs]
        pcnum: int
            number of pc regressors to actually use
        nr_TRs: int
            number of TRs in run
               
        """
        # if we didnt add pcregressors, just make list of None
        if pcregressors is None:
            extra_regressors_all = [None for r in range(np.array(maxpolydeg).shape[0])] # length of nr of runs
        else:
            extra_regressors_all = [pcregressors[r][:, :pcnum] for r in range(np.array(maxpolydeg).shape[0])]

        # construct projection matrices for the nuisance components
        polymatrix = []
        combinedmatrix = []
        
        for p in range(np.array(maxpolydeg).shape[0]): # for each run

            # this projects out polynomials
            pmatrix = make_polynomial_matrix(nr_TRs,
                                            maxpolydeg[p])
            polymatrix.append(make_projection_matrix(pmatrix))

            extra_regressors = extra_regressors_all[p]

            # this projects out both of them
            if extra_regressors is not None:
                if extra_regressors.any():
                    combinedmatrix.append(
                        make_projection_matrix(
                            np.c_[pmatrix, extra_regressors]
                        )
                    )
            else:
                combinedmatrix.append(
                    make_projection_matrix(pmatrix))

        return polymatrix, combinedmatrix

    def get_hrf_tc(self, hrf_library, hrf_index):

        """
        make hrf timecourse
        for all surface vertex
        
        Parameters
        ----------
        hrf_library: arr
            hrf library that was used in fit
        hrf_index: arr
            hrf index for each vertex  
        """
        hrf_surf = [hrf_library[:,ind_hrf] for ind_hrf in hrf_index]
        return np.vstack(hrf_surf)

    def get_prediction_tc(self, dm, hrf_tc, betas, psc_tc=False, combinedmatrix=None, meanvol=None):

        """
        get prediction timecourse for vertex
        
        Parameters
        ----------
        dm: arr
            design matrix [TR, nr_cond]
        hrf_tc: arr
            hrf timecourse for vertex
        betas: arr
            beta values for vertex [nr_cond]
        psc_tc: bool
            if we want to percent signal change predicted timecourse
        combinedmatrix: arr
            nuisance component matrix - NOTE! If we want to psc accurately, combinedmatrix should NOT have intercept
        meanvol: float
            mean volume for vertex
        """
        # convolve HRF into design matrix
        design0 = scipy.signal.convolve2d(dm, hrf_tc[..., np.newaxis])[0:dm.shape[0],:]  

        # predicted timecourse is design times betas from fit
        prediction = design0 @ betas if psc_tc == True else design0 @ betas/100 * meanvol

        # denoise
        if combinedmatrix is not None:
            prediction = combinedmatrix.astype(np.float32) @ prediction

        return prediction

    def get_denoised_data(self, participant, maxpolydeg = [], pcregressors = None, pcnum = 1,
                                            run_ID = None, psc_tc = False):

        """
        get denoised data 
        (this is, taking into account nuisance regressors glsingle uses during fit)
        
        Parameters
        ----------
        participant: str
            participant ID  
        maxpolydeg: list
            list with ints which represent 
            polynomial degree to use for polynomial nuisance functions [runs, poly] 
        pcregressors: list
            list with pcregressors to add to nuisance matrix [runs, TR, nr_pcregs]
        pcnum: int
            number of pc regressors to actually use
        run_ID: int
            run identifier, if we only want to load and denoise specific run
            NOTE! in this case, expects input of pcregressors and maxpoly for that run 
        psc_tc: bool
            if we want to percent signal change predicted timecourse
    
        """

        ## get list with gii files
        gii_filenames = self.get_soma_file_list(participant, 
                              file_ext = 'hemi-L{ext}'.format(ext = self.MRIObj.file_ext)) + \
                        self.get_soma_file_list(participant, 
                                    file_ext = 'hemi-R{ext}'.format(ext = self.MRIObj.file_ext))
        
        ## if we want a specific run, filter
        if run_ID is not None:
            gii_filenames = [file for file in gii_filenames if 'run-{r}'.format(r = str(run_ID).zfill(2)) in file]

        ## load data of all runs
        all_data = self.load_data4fitting(gii_filenames) # [runs, vertex, TR]

        ## get nuisance matrix
        # if we want PSC data, then dont remove intercept (mean vol)
        if psc_tc == True:
            maxpolydeg = [[val for val in r_list if val != 0] for r_list in maxpolydeg]

        combinedmatrix = self.get_nuisance_matrix(maxpolydeg,
                                                pcregressors = pcregressors,
                                                pcnum = pcnum,
                                                nr_TRs = np.array(all_data).shape[-1])[-1]
        ## denoise data
        data_out = [] 
        for r, cm in enumerate(combinedmatrix):

            data_denoised = cm.astype(np.float32) @ np.transpose(all_data[r])

            if psc_tc == True:
                mean_signal = data_denoised.mean(axis = 0)[np.newaxis, ...]
                data_psc = (data_denoised - mean_signal)/np.absolute(mean_signal)
                data_psc *= 100
                data_out.append(data_psc) 
            else:
                data_out.append(data_denoised) 

        return data_out


    def fit_data(self, participant):

        """ fit glm single model to participant data
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        # get list with gii files
        gii_filenames = self.get_soma_file_list(participant, 
                              file_ext = 'hemi-L{ext}'.format(ext = self.MRIObj.file_ext)) + \
                        self.get_soma_file_list(participant, 
                                    file_ext = 'hemi-R{ext}'.format(ext = self.MRIObj.file_ext))

        # load data of all runs
        all_data = self.load_data4fitting(gii_filenames) # [runs, vertex, TR]
        
        # make design matrix
        all_dm = self.get_dm_glmsing(nr_TRs = np.array(all_data).shape[-1], 
                                   nr_runs = len(self.get_run_list(gii_filenames)))

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            'hrf_{h}'.format(h = self.glm_single_ops['hrf']))
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        ## create a directory for saving GLMsingle outputs
        opt = dict()

        # set important fields 
        if self.glm_single_ops['hrf'] in ['canonical', 'average']: # turn hrf fitting off
            opt['wantlibrary'] = 0
        else:
            opt['wantlibrary'] = 1 # will fit hrf

        if self.glm_single_ops['denoise']: # if we are denoising 
            opt['wantglmdenoise'] = 1
        else:
            opt['wantglmdenoise'] = 0

        if self.glm_single_ops['fracr']: # if we are doing frac ridge 
            opt['wantfracridge'] = 1
        else:
            opt['wantfracridge'] = 0

        # shift onset by remainder, to make start point accurate
        opt['hrfonset'] = -(self.MRIObj.soma_event_time_in_sec['empty'] % self.MRIObj.TR) 

        #opt['hrftoassume'] = hrf_final
        #opt['brainexclude'] = final_mask.astype(int) #prf_mask.astype(int)
        #opt['brainR2'] = 100

        opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold

        # limit polynomials used, to only intercept and linear slope
        opt['maxpolydeg'] = [[0, 1] for _ in range(np.array(all_data).shape[0])]

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
        
        # get start time
        start_time = datetime.datetime.now()

        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
                                            all_dm,
                                            all_data,
                                            self.MRIObj.soma_event_time_in_sec['stim'],
                                            self.MRIObj.TR,
                                            outputdir = out_dir)

        # Print duration, for bookeeping
        end_time = datetime.datetime.now()
        print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                        start_time = start_time,
                        end_time = end_time,
                        dur  = end_time - start_time))

        ## save opt for easy access later (and consistency)
        print('Saving opts dict')
        np.save(op.join(out_dir, 'OPTS.npy'), opt)

        ## save some plots for sanity check ##

        ## MODEL D
        # CV-rsq
        flatmap = cortex.Vertex(results_glmsingle['typed']['R2'], 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 80, 
                        cmap='hot')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeD_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        # Frac value
        flatmap = cortex.Vertex(results_glmsingle['typed']['FRACvalue'], 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 1, 
                        cmap='copper')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeD_fracridge.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        # Average betas 
        flatmap = cortex.Vertex(np.mean(results_glmsingle['typed']['betasmd'], axis = -1), 
                        self.MRIObj.sj_space,
                        vmin = -3, vmax = 3, 
                        cmap='RdBu_r')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeD_avgbetas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        # Noise pool
        flatmap = cortex.Vertex(results_glmsingle['typed']['noisepool'], 
                  self.MRIObj.sj_space,
                   vmin = 0, vmax = 1, #.7,
                   cmap='hot')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeD_noisepool.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## MODEL A
        # RSQ
        flatmap = cortex.Vertex(results_glmsingle['typea']['onoffR2'], 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 50, #.7,
                        cmap='hot')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeA_ONOFF_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        # Betas
        flatmap = cortex.Vertex(results_glmsingle['typea']['betasmd'][...,0], 
                        self.MRIObj.sj_space,
                        vmin = -2, vmax = 2, 
                        cmap='RdBu_r')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)

        fig_name = op.join(out_dir, 'modeltypeA_ONOFF_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)



    def load_results_dict(self, participant, fits_dir = None, load_opts = False):

        """ helper function to 
        load glm single model estimates
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        # path to estimates
        if fits_dir is None:
            fits_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            'hrf_{h}'.format(h = self.glm_single_ops['hrf']))

        ## load existing file outputs if they exist
        results_glmsingle = dict()
        results_glmsingle['typea'] = np.load(op.join(fits_dir,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
        results_glmsingle['typeb'] = np.load(op.join(fits_dir,'TYPEB_FITHRF.npy'),allow_pickle=True).item()
        results_glmsingle['typec'] = np.load(op.join(fits_dir,'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
        results_glmsingle['typed'] = np.load(op.join(fits_dir,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()

        ## if we saved fit params, load them too
        if load_opts:
            opt = dict()
            opt = np.load(op.join(fits_dir, 'OPTS.npy'), allow_pickle=True).item()
            return results_glmsingle, opt
        
        else:
            return results_glmsingle



