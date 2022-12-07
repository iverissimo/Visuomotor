import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob
from nilearn import surface

import datetime

from glmsingle.glmsingle import GLM_single
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
        helper function to make glmsingle DM for all runs
        
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

    
    def fit_data(self, participant):

        """ function to fit glm single model to participant data
        
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

        ## save some plots for sanity check
        flatmap = cortex.Vertex(results_glmsingle['typed']['R2'], 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 80, #.7,
                        cmap='hot')

        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
        fig_name = op.join(out_dir, 'modeltypeD_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        flatmap = cortex.Vertex(results_glmsingle['typed']['FRACvalue'], 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 1, #.7,
                        cmap='copper')

        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
        fig_name = op.join(out_dir, 'modeltypeD_fracridge.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        flatmap = cortex.Vertex(np.mean(results_glmsingle['typed']['betasmd'], axis = -1), 
                        self.MRIObj.sj_space,
                        vmin = -3, vmax = 3, #.7,
                        cmap='RdBu_r')

        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
        fig_name = op.join(out_dir, 'modeltypeD_avgbetas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

                                                        