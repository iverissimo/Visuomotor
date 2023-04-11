import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob
from nilearn import surface
import nibabel as nib

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from statsmodels.stats.multitest import fdrcorrection
from scipy import ndimage

import datetime


from model import Model

import scipy

import cortex
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm


class somaModel(Model):

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

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir)

        # processed file extension
        self.proc_file_ext = self.MRIObj.params['fitting']['soma']['extension']

        # path to postfmriprep files
        self.proc_file_pth = op.join(self.MRIObj.postfmriprep_pth, self.MRIObj.sj_space, 'soma')

        # get trial condition labels, in order 
        self.soma_stim_labels = [op.splitext(val)[0] for val in self.MRIObj.params['fitting']['soma']['soma_stimulus']]
        
        # get unique labels of conditions
        _, idx = np.unique(self.soma_stim_labels, return_index=True) 
        self.soma_cond_unique = np.array(self.soma_stim_labels)[np.sort(idx)]

    def set_contrast(self, dm_col, tasks, contrast_val = [1], num_cond = 1):
    
        """ define contrast matrix

        Parameters
        ----------
        dm_col : list/arr
            design matrix columns (all possible task names in list)
        tasks : list/arr
            list with list of tasks to give contrast value
            if num_cond=1 : [tasks]
            if num_cond=2 : [tasks1,tasks2], contrast will be tasks1 - tasks2     
        contrast_val : list/arr 
            list with values for contrast
            if num_cond=1 : [value]
            if num_cond=2 : [value1,value2], contrast will be tasks1 - tasks2
        num_cond : int
            if one task vs the implicit baseline (1), or if comparing 2 conditions (2)

        Outputs
        -------
        contrast : list/arr
            contrast array

        """
        contrast = np.zeros(len(dm_col))

        if num_cond == 1: # if only one contrast value to give ("task vs implicit intercept")

            for j,name in enumerate(tasks[0]):
                for i in range(len(contrast)):
                    if dm_col[i] == name:
                        contrast[i] = contrast_val[0]

        elif num_cond == 2: # if comparing 2 conditions (task1 - task2)

            for k,lbl in enumerate(tasks):
                idx = []
                for i,val in enumerate(lbl):
                    idx.extend(np.where([1 if val == label else 0 for _,label in enumerate(dm_col)])[0])

                val = contrast_val[0] if k==0 else contrast_val[1] # value to give contrast

                for j in range(len(idx)):
                    for i in range(len(dm_col)):
                        if i==idx[j]:
                            contrast[i]=val

        print('contrast for %s is %s'%(tasks,contrast))
        return contrast

    def design_variance(self, X, which_predictor=[]):
        
        """Returns the design variance of a predictor (or contrast) in X.

        Parameters
        ----------
        X : numpy array
            Array of shape (N, P)
        which_predictor : list/array
            contrast-vector of the predictors you want the design var from.

        Outputs
        -------
        des_var : float
            Design variance of the specified predictor/contrast from X.
        """

        idx = np.array(which_predictor) != 0

        c = np.zeros(X.shape[1])
        c[idx] = which_predictor[idx]
        des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)

        return des_var

    def calc_contrast_stats(self, betas = [], contrast = [], 
                                sse = None, df = None, design_var = None, pval_1sided = True):

        """Calculates stats for a given contrast and beta values
        
        Parameters
        ----------
        betas : arr
            array of beta values
        contrast: arr/list
            contrast vector       
        sse: float
            sum of squared errors between model prediction and data
        df: int
            degrees of freedom (timepoints - predictores)   
        design_var: float
            design variance 
        pval_1sided: bool
            if we want one or two sided p-value

        """
        # t statistic for vertex
        t_val = contrast.dot(betas) / np.sqrt((sse/df) * design_var)

        if pval_1sided == True:
            # compute the p-value (right-tailed)
            p_val = scipy.stats.t.sf(t_val, df) 

            # z-score corresponding to certain p-value
            z_score = scipy.stats.norm.isf(np.clip(p_val, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy
        else:
            # take the absolute by np.abs(t)
            p_val = scipy.stats.t.sf(np.abs(t_val), df) * 2 # multiply by two to create a two-tailed p-value

            # z-score corresponding to certain p-value
            z_score = scipy.stats.norm.isf(np.clip(p_val/2, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

        return t_val,p_val,z_score

    def calc_contrast_effect(self, betas = [], contrast = [], return_zscore = True,
                                sse = None, df = None, design_var = None, pval_1sided = True):

        """Calculates effect size and variance 
        for a given contrast and beta values
        
        Parameters
        ----------
        betas : arr
            array of beta values
        contrast: arr/list
            contrast vector       
        sse: float
            sum of squared errors between model prediction and data
        df: int
            degrees of freedom (timepoints - predictores)   
        design_var: float
            design variance 
        pval_1sided: bool
            if we want one or two sided p-value
        """

        # effect size
        cb = contrast.dot(betas)

        # effect variance
        effect_var = (sse/df) * design_var

        if return_zscore:
            z_score = self.calc_contrast_stats(betas = betas, contrast = contrast, 
                                                sse = sse, df = df, design_var = design_var, pval_1sided = pval_1sided)[-1]

            return cb, effect_var, z_score
        else:
            return cb, effect_var

    def get_avg_events(self, participant, keep_b_evs = False):

        """ get events for participant (averaged over runs)
        
        Parameters
        ----------
        participant: str
            participant ID 
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)           
        """

        events_list = [run for run in glob.glob(op.join(self.MRIObj.sourcedata_pth,
                                'sub-{sj}'.format(sj=participant),'*','func/*')) if 'soma' in run and run.endswith('events.tsv')]

        # list of stimulus onsets
        print('averaging %d event files'%len(events_list))

        all_events = []

        for _,val in enumerate(events_list):

            events_pd = pd.read_csv(val,sep = '\t')

            new_events = []

            for ev in events_pd.iterrows():
                row = ev[1]   
                if (row['trial_type'][0] == 'b') and (keep_b_evs == False): # if both hand/leg then add right and left events with same timings
                    new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
                    new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
                else:
                    new_events.append([row['onset'],row['duration'],row['trial_type']])

            df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
            all_events.append(df)

        # make median event dataframe
        onsets = []
        durations = []
        for w in range(len(all_events)):
            onsets.append(all_events[w]['onset'])
            durations.append(all_events[w]['duration'])

        events_avg = pd.DataFrame({'onset':np.median(np.array(onsets),axis=0),
                                   'duration':np.median(np.array(durations),axis=0),
                                   'trial_type':all_events[0]['trial_type']})
        
        print('computed median events')

        return events_avg

    def create_hrf_tc(self, hrf_params=[1.0, 1.0, 0.0], osf = 1, onset = 0):
        
        """
        construct single or multiple HRFs - taken from prfpy   

        Parameters
        ----------
        hrf_params : TYPE, optional
            DESCRIPTION. The default is [1.0, 1.0, 0.0].
        Returns
        -------
        hrf : ndarray
            the hrf.
        """

        hrf = np.array(
            [
                np.ones_like(hrf_params[1])*hrf_params[0] *
                spm_hrf(
                    tr=self.MRIObj.TR,
                    oversampling=osf,
                    onset=onset,
                    time_length=40)[...,np.newaxis],
                hrf_params[1] *
                spm_time_derivative(
                    tr=self.MRIObj.TR,
                    oversampling=osf,
                    onset=onset,
                    time_length=40)[...,np.newaxis],
                hrf_params[2] *
                spm_dispersion_derivative(
                    tr=self.MRIObj.TR,
                    oversampling=osf,
                    onset=onset,
                    time_length=40)[...,np.newaxis]]).sum(
            axis=0)                    

        return hrf.T/hrf.T.max()

    def convolve_tc_hrf(self, tc, hrf, pad_length = 20):
        """
        Helper function to
        Convolve timecourse with hrf
        
        Parameters
        ----------
        tc : ndarray, 1D or 2D
            The timecourse(s) to be convolved.
        hrf : ndarray, 1D or 2D
            The HRF
        Returns
        -------
        convolved_tc : ndarray
            Convolved timecourse.
        """
        #scipy fftconvolve does not have padding options so doing it manually
        pad = np.tile(tc[:,0], (pad_length,1)).T
        padded_tc = np.hstack((pad,tc))

        convolved_tc = scipy.signal.fftconvolve(padded_tc, hrf, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length] 

        return convolved_tc

    def resample_arr(self, upsample_data, osf = 10, final_sf = 1.6, anti_aliasing = True):

        """ resample array
        using cubic interpolation
        
        Parameters
        ----------
        upsample_data : arr
            1d array that is upsampled
        osf : int
            oversampling factor (that data was upsampled by)
        final_sf: float
            final sampling rate that we want to obtain
            
        """
        
        if anti_aliasing:
            out_arr = scipy.signal.decimate(upsample_data, int(osf * final_sf), ftype = 'fir')

        else:
            # original scale of data in seconds
            original_scale = np.arange(0, upsample_data.shape[-1]/osf, 1/osf)

            # cubic interpolation of predictor
            interp = scipy.interpolate.interp1d(original_scale, 
                                        upsample_data, 
                                        kind = "linear", axis=-1) #"cubic", axis=-1)
            
            desired_scale = np.arange(0, upsample_data.shape[-1]/osf, final_sf) # we want the predictor to be sampled in TR

            out_arr = interp(desired_scale)
        
        return out_arr

    def COM(self, data):
    
        """ given an array of values x vertices, 
        compute center of mass 

        Parameters
        ----------
        data : List/arr
            array with values to COM  (elements,vertices)

        Outputs
        -------
        center_mass : arr
            array with COM for each vertex
        """
        
        # first normalize data, to fix issue of negative values
        norm_data = np.array([self.normalize(data[...,x]) for x in range(data.shape[-1])])
        norm_data = norm_data.T
        
        #then calculate COM for each vertex
        center_mass = np.array([ndimage.measurements.center_of_mass(norm_data[...,x]) for x in range(norm_data.shape[-1])])

        return center_mass.T[0]

class GLM_Model(somaModel):

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
            self.outputdir = op.join(self.MRIObj.derivatives_pth, 'glm_fits')

        # set path for stats and COM
        self.stats_outputdir = op.join(op.split(self.outputdir)[0], 'glm_stats')
        self.COM_outputdir = op.join(op.split(self.outputdir)[0], 'glm_COM')

    def load_design_matrix(self, participant, keep_b_evs = True, custom_dm = True, osf = 100, nTRs = 141, hrf_model = 'glover'):

        """
        Load participant design matrix
        
        Parameters
        ----------
        participant : str
            participant ID
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands) 
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        osf: int
            oversampling factor for when custom_DM = True
        nTRs: int
            number of TRs in data run
        hrf_model: str
            type of hrf to use (when custom_hrf = False)        
        """

        # get average event DataFrame for participant, based on events file
        events_avg = self.get_avg_events(participant, keep_b_evs = keep_b_evs)

        # make DM based on specific HRF params
        if custom_dm:
            design_matrix = self.make_custom_dm(events_avg, 
                                                osf = osf, data_len_TR = nTRs, 
                                                hrf_params = self.MRIObj.params['fitting']['soma']['hrf_params'], 
                                                hrf_onset = self.MRIObj.params['fitting']['soma']['hrf_onset'])
        # or use nilearn HRF model
        else:
            design_matrix = make_first_level_design_matrix(self.MRIObj.TR * (np.arange(nTRs)),
                                                        events = events_avg,
                                                        hrf_model = hrf_model
                                                        ) 
        return design_matrix

    def make_custom_dm(self, events_df, osf = 100, data_len_TR = 141, hrf_params = [1,1,0], hrf_onset = 0):

        """
        Helper function to make custom dm, 
        from custom HRF

        Parameters
        ----------
        events_df: DataFrame
            events dataframe
        osf: int
            oversampling factor for when custom_DM = True
        data_len_TR: int
            number of TRs in data run
        hrf_model: str
            type of hrf to use (when custom_hrf = False)  
        hrf_params: list
            HRF params to use when creating HRF
        hrf_onset: float
            hrf onset, in TR
        """

        # list with regressor names
        regressor_names = events_df.trial_type.unique()

        # task duration in seconds
        task_dur_sec = data_len_TR * self.MRIObj.TR 

        # hrf timecourse in sec * TR !!
        hrf_tc = self.create_hrf_tc(hrf_params=hrf_params, osf = osf * self.MRIObj.TR, onset = hrf_onset)

        all_regs_dict = {}

        # for each regressor
        for ind, reg_name in enumerate(regressor_names):

            onsets = (events_df[events_df['trial_type'] == reg_name].onset.values * osf).astype(int)
            stim_dur = (events_df[events_df['trial_type'] == reg_name].duration.values * osf).astype(int)

            reg_pred_osf = np.zeros(int(task_dur_sec * osf)) # oversampled array to be filled with predictor onset and dur

            for i in range(len(onsets)): ## fill in regressor with ones, given onset and stim duration
                reg_pred_osf[onsets[i]:int(onsets[i]+stim_dur[i])] = 1

            # now convolve with hrf
            reg_pred_osf_conv = self.convolve_tc_hrf(reg_pred_osf[np.newaxis,...], 
                                                    hrf_tc, 
                                                    pad_length = 20 * osf)

            # and resample back to TR 
            reg_resampled = self.resample_arr(reg_pred_osf_conv[0], 
                                                osf = osf, 
                                                final_sf = self.MRIObj.TR)/(osf) # dividing by osf so amplitude scaling not massive (but irrelevante because a.u.)
            
            all_regs_dict[reg_name] = reg_resampled
            
        # add intercept
        all_regs_dict['constant'] = np.ones(data_len_TR)
        # convert to DF
        dm_df = pd.DataFrame.from_dict(all_regs_dict)

        return dm_df

    def fit_glm_tc(self, voxel, dm):
    
        """ GLM fit on timeseries
        Regress a created design matrix on the input_data.

        Parameters
        ----------
        voxel : arr
            timeseries of a single voxel
        dm : arr
            DM array (#TR,#regressors)
        """

        if np.isnan(voxel).any():
            betas = np.repeat(np.nan, dm.shape[-1])
            prediction = np.repeat(np.nan, dm.shape[0])
            mse = np.nan
            r2 = np.nan

        else:   # if not nan (some vertices might have nan values)
            betas = np.linalg.lstsq(dm, voxel, rcond = -1)[0]
            prediction = dm.dot(betas)

            mse = np.mean((voxel - prediction) ** 2) # calculate mean of squared residuals
            r2 = np.nan_to_num(1 - (np.nansum((voxel - prediction)**2, axis=0)/ np.nansum(((voxel - np.mean(voxel))**2), axis=0)))# and the rsq
        
        return prediction, betas, r2, mse

    def get_region_keys(self, reg_keys = ['face', 'upper_limb', 'lower_limb'], keep_b_evs = False):

        """
        Helper function to get gross motor region 
        key dictionary 
        (usefull to make general region contrasts)

        Parameters
        ----------
        reg_keys : list
            list with strings, listing name of gross regions to contrast
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands) 
        """

        region_regs = {}

        for reg in reg_keys:

            if (keep_b_evs == False) and (reg != 'face'):
                region_regs[reg] = [val for val in self.MRIObj.params['fitting']['soma']['all_contrasts'][reg] if 'bhand' not in val and 'bleg' not in val]
            else:
                region_regs[reg] = self.MRIObj.params['fitting']['soma']['all_contrasts'][reg]
            
        return region_regs
    
    def load_GLMestimates(self, participant, fit_type = 'mean_run', run_id = None):

        """ 
        Load GLM estimates dictionary,
        for folder where it was fit
        
        Parameters
        ----------
        participant: str
            participant ID  
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        run_id: str
            id of run, for case of loo-run
        """

        if fit_type == 'loo_run':
            ## load estimates, and get betas and prediction
            soma_estimates = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), fit_type, 
                                            'estimates_loo_run-{ri}.npy'.format(ri = str(run_id).zfill(2))), 
                                            allow_pickle=True).item()

        elif fit_type == 'mean_run':     
            ## load estimates, and get betas and prediction
            soma_estimates = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                            allow_pickle=True).item()
            
        return soma_estimates
        
    def contrast_regions(self, participant, hrf_model = 'glover', custom_dm = True, fit_type = 'mean_run',
                                            z_threshold = 3.1, pval_1sided = True, keep_b_evs = False,
                                            reg_keys = ['face', 'upper_limb', 'lower_limb']):

        """ Make simple contrasts to localize body regions
        (body, hands, legs) and contralateral regions (R vs L hand)

        Requires GLM fitting to have been done before
        
        Parameters
        ----------
        participant: str
            participant ID  
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)   
        reg_keys : list
            list with strings, listing name of gross regions to contrast
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)    
        hrf_model: str
            type of hrf to use (when custom_hrf = False) 
        z_threshold: float
            z-score threshold
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.stats_outputdir, 'sub-{sj}'.format(sj = participant), fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        ## Load data
        # get list with gii files
        gii_filenames = self.get_proc_file_list(participant, file_ext = self.proc_file_ext)

        if fit_type == 'mean_run':         
            # load data of all runs and average
            data2fit = self.load_data4fitting(gii_filenames, average = True)[np.newaxis,...] # [1, vertex, TR]
            run_id = None

        elif fit_type == 'loo_run':
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            # leave one run out, load others and average
            data2fit = []
            for lo_run_key in run_loo_list:
                print('Leaving run-{r} out'.format(r = str(lo_run_key).zfill(2)))
                data2fit.append(self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(lo_run_key).zfill(2)) not in file], 
                                                        average = True)[np.newaxis,...])
            data2fit = np.vstack(data2fit) # [runs, vertex, TR]

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = data2fit.shape[-1], 
                                                hrf_model = hrf_model)

        ## get stats for all runs
        for rind, avg_data in enumerate(data2fit): # [vertex, TR]

            if fit_type == 'loo_run':
                # update outdir for current run we are running
                out_dir = op.join(out_dir, 'loo_run-{ri}'.format(ri = str(run_loo_list[rind]).zfill(2)))
                os.makedirs(out_dir, exist_ok = True)

                run_id = run_loo_list[rind]

            ## load estimates, and get betas and prediction
            soma_estimates = self.load_GLMestimates(participant, fit_type = fit_type, run_id = run_id)

            betas = soma_estimates['betas']
            prediction = soma_estimates['prediction']

            # now make simple contrasts
            print('Computing simple contrasts')
            print('Using z-score of %0.2f as threshold for localizer' %z_threshold)
            
            # get region keys
            region_regs = self.get_region_keys(reg_keys = reg_keys, keep_b_evs = keep_b_evs)
            # loo for keys 
            loo_keys = self.leave_one_out(reg_keys) 

            # one broader region vs all the others
            for index,region in enumerate(reg_keys): 

                print('contrast for %s ' %region)

                # list of other contrasts
                other_contr = np.append(region_regs[loo_keys[index][0]],
                                        region_regs[loo_keys[index][1]])

                # main contrast calculated
                contrast = self.set_contrast(design_matrix.columns, 
                                        [region_regs[str(region)], other_contr],
                                    [1,-len(region_regs[str(region)])/len(other_contr)],
                                    num_cond=2)

                # set filename
                stats_filename = op.join(out_dir, 'stats_{reg}_vs_all_contrast.npy'.format(reg = region))
                
                # compute contrast-related statistics
                soma_stats = Parallel(n_jobs=16)(delayed(self.calc_contrast_stats)(betas = betas[v], 
                                                                                contrast = contrast, 
                                                                                sse = ((avg_data[v] - prediction[v]) ** 2).sum() , 
                                                                                df = (design_matrix.values.shape[0] - design_matrix.values.shape[1]), 
                                                                                design_var = self.design_variance(design_matrix.values, contrast), 
                                                                                pval_1sided = pval_1sided) for v in np.arange(avg_data.shape[0]))
                soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore
                soma_stats_dict = {}
                soma_stats_dict['t_val'] = soma_stats[..., 0]
                soma_stats_dict['p_val'] = soma_stats[..., 1]
                soma_stats_dict['z_score'] = soma_stats[..., 2]
                
                np.save(stats_filename, soma_stats_dict)

                ## now do rest of the contrasts within region (if lateralized) ###
                if region != 'face':

                    # compare left and right
                    print('Right vs Left contrasts')

                    if region == 'upper_limb':
                        limbs = ['hand', self.MRIObj.params['fitting']['soma']['all_contrasts']['upper_limb']]
                    else:
                        limbs = ['leg', self.MRIObj.params['fitting']['soma']['all_contrasts']['lower_limb']]
                                
                    rtask = [s for s in limbs[-1] if 'r'+limbs[0] in s]
                    ltask = [s for s in limbs[-1] if 'l'+limbs[0] in s]
                    tasks = [rtask,ltask] # list with right and left elements
                                
                    contrast = self.set_contrast(design_matrix.columns, tasks, [1, -1], num_cond=2)

                    # set filename
                    LR_stats_filename = op.join(out_dir, 'stats_{reg}_RvsL_contrast.npy'.format(reg = region))

                    # mask data - only significant voxels for region
                    region_ind = np.where((soma_stats_dict['z_score'] >= z_threshold))[0]

                    # compute contrast-related statistics
                    LR_stats = Parallel(n_jobs=16)(delayed(self.calc_contrast_stats)(betas = betas[v], 
                                                                                contrast = contrast, 
                                                                                sse = ((avg_data[v] - prediction[v]) ** 2).sum() , 
                                                                                df = (design_matrix.values.shape[0] - design_matrix.values.shape[1]), 
                                                                                design_var = self.design_variance(design_matrix.values, contrast), 
                                                                                pval_1sided = pval_1sided) for v in tqdm(region_ind))
                    LR_stats = np.vstack(LR_stats) # t_val, p_val, zscore

                    # fill for whole surface
                    LR_surf_stats = np.zeros((soma_stats_dict['z_score'].shape[0], LR_stats.shape[-1]))
                    LR_surf_stats[:] = np.nan
                    LR_surf_stats[region_ind,:] = LR_stats

                    # save in dict
                    LR_stats_dict = {}
                    LR_stats_dict['t_val'] = LR_surf_stats[..., 0]
                    LR_stats_dict['p_val'] = LR_surf_stats[..., 1]
                    LR_stats_dict['z_score'] = LR_surf_stats[..., 2]
                    
                    np.save(LR_stats_filename, LR_stats_dict)

    def fixed_effects_contrast_regions(self, participant, hrf_model = 'glover', custom_dm = True, fit_type = 'loo_run',
                                            z_threshold = 3.1, pval_1sided = True, keep_b_evs = False,
                                            reg_keys = ['face', 'upper_limb', 'lower_limb']):

        """ Calculate fixed effects across runs to make
        Make simple contrasts to localize body regions
        (body, hands, legs) and contralateral regions (R vs L hand)

        Requires GLM fitting to have been done before
        
        Parameters
        ----------
        participant: str
            participant ID  
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)   
        reg_keys : list
            list with strings, listing name of gross regions to contrast
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)    
        hrf_model: str
            type of hrf to use (when custom_hrf = False) 
        z_threshold: float
            z-score threshold          
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.stats_outputdir, 'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # get list with gii files
        gii_filenames = self.get_proc_file_list(participant, file_ext = self.proc_file_ext)

        if fit_type == 'loo_run':
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            # leave one run out, load others and average
            data2fit = []
            for lo_run_key in run_loo_list:
                print('Leaving run-{r} out'.format(r = str(lo_run_key).zfill(2)))
                data2fit.append(self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(lo_run_key).zfill(2)) not in file], 
                                                        average = True)[np.newaxis,...])
            data2fit = np.vstack(data2fit) # [runs, vertex, TR]

        elif fit_type == 'all_run':
            raise ValueError('Would load and make fix effects for each individual run fit. Not implemented yet')

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = data2fit.shape[-1], 
                                                hrf_model = hrf_model)

        ## calculate contrast effects for all runs
        all_runs_effects = {'face': {'effect_size': [], 'effect_variance': []},
                            'upper_limb': {'effect_size': [], 'effect_variance': []},
                            'upper_limb_RvsL': {'effect_size': [], 'effect_variance': []},
                            'lower_limb': {'effect_size': [], 'effect_variance': []},
                            'lower_limb_RvsL': {'effect_size': [], 'effect_variance': []}} # to append all and calculate fix effects in the end

        for rind, avg_data in enumerate(data2fit): # [vertex, TR]

            ## load estimates, and get betas and prediction
            soma_estimates = self.load_GLMestimates(participant, fit_type = fit_type, run_id = run_loo_list[rind])

            betas = soma_estimates['betas']
            prediction = soma_estimates['prediction']

            # now make simple contrasts
            print('Computing simple contrasts')

            # get region keys
            region_regs = self.get_region_keys(reg_keys = reg_keys, keep_b_evs = keep_b_evs)
            # loo for keys 
            loo_keys = self.leave_one_out(reg_keys) 
        
            # one broader region vs all the others
            for index,region in enumerate(reg_keys): 

                print('contrast for %s ' %region)

                # list of other contrasts
                other_contr = np.append(region_regs[loo_keys[index][0]],
                                        region_regs[loo_keys[index][1]])

                # main contrast calculated
                contrast = self.set_contrast(design_matrix.columns, 
                                        [region_regs[str(region)], other_contr],
                                    [1,-len(region_regs[str(region)])/len(other_contr)],
                                    num_cond=2)

                # set filename
                effect_filename = op.join(out_dir, 'run-{ri}_effect_{reg}_vs_all_contrast.npy'.format(reg = region,
                                                                                                    ri = str(run_loo_list[rind]).zfill(2)))
                
                # compute contrast-related effect size and variance 
                # for run
                soma_run_effect = Parallel(n_jobs=16)(delayed(self.calc_contrast_effect)(betas = betas[v], 
                                                                                contrast = contrast, 
                                                                                sse = ((avg_data[v] - prediction[v]) ** 2).sum() , 
                                                                                df = (design_matrix.values.shape[0] - design_matrix.values.shape[1]), 
                                                                                design_var = self.design_variance(design_matrix.values, contrast),
                                                                                return_zscore = True 
                                                                                ) for v in np.arange(avg_data.shape[0]))

                soma_run_effect = np.vstack(soma_run_effect) # cb, effect_var
                soma_run_effect_dict = {}
                soma_run_effect_dict['effect_size'] = soma_run_effect[..., 0]
                soma_run_effect_dict['effect_variance'] = soma_run_effect[..., 1]

                np.save(effect_filename, soma_run_effect_dict)

                # append run
                all_runs_effects[region]['effect_size'].append(soma_run_effect[..., 0][np.newaxis,...])
                all_runs_effects[region]['effect_variance'].append(soma_run_effect[..., 1][np.newaxis,...])

                ## now do rest of the contrasts within region (if lateralized) ###

                if region != 'face':

                    # compare left and right
                    print('Right vs Left contrasts')

                    if region == 'upper_limb':
                        limbs = ['hand', self.MRIObj.params['fitting']['soma']['all_contrasts']['upper_limb']]
                    else:
                        limbs = ['leg', self.MRIObj.params['fitting']['soma']['all_contrasts']['lower_limb']]
                                
                    rtask = [s for s in limbs[-1] if 'r'+limbs[0] in s]
                    ltask = [s for s in limbs[-1] if 'l'+limbs[0] in s]
                    tasks = [rtask,ltask] # list with right and left elements
                                
                    contrast = self.set_contrast(design_matrix.columns, tasks, [1, -1], num_cond=2)

                    # set filename
                    LR_effect_filename = op.join(out_dir, 'run-{ri}_effect_{reg}_RvsL_contrast.npy'.format(reg = region,
                                                                                                        ri = str(run_loo_list[rind]).zfill(2)))

                    # mask data - only significant voxels for region
                    region_ind = np.where((soma_run_effect[..., 2] >= z_threshold))[0]

                    # compute contrast-related effect size and variance 
                    # for run
                    LR_run_effect = Parallel(n_jobs=16)(delayed(self.calc_contrast_effect)(betas = betas[v], 
                                                                                contrast = contrast, 
                                                                                sse = ((avg_data[v] - prediction[v]) ** 2).sum() , 
                                                                                df = (design_matrix.values.shape[0] - design_matrix.values.shape[1]), 
                                                                                design_var = self.design_variance(design_matrix.values, contrast), 
                                                                                return_zscore = False) for v in tqdm(region_ind))
                    LR_run_effect = np.vstack(LR_run_effect) # cb, effect_var

                    # fill for whole surface
                    LR_surf_effect = np.zeros((soma_run_effect[..., 2].shape[0], LR_run_effect.shape[-1]))
                    LR_surf_effect[:] = np.nan
                    LR_surf_effect[region_ind,:] = LR_run_effect

                    # save in dict
                    LR_run_effect_dict = {}
                    LR_run_effect_dict['effect_size'] = LR_surf_effect[..., 0]
                    LR_run_effect_dict['effect_variance'] = LR_surf_effect[..., 1]
                    
                    np.save(LR_effect_filename, LR_run_effect_dict)

                    # append run
                    all_runs_effects['{reg}_RvsL'.format(reg = region)]['effect_size'].append(LR_surf_effect[..., 0][np.newaxis,...])
                    all_runs_effects['{reg}_RvsL'.format(reg = region)]['effect_variance'].append(LR_surf_effect[..., 1][np.newaxis,...])

        ## NOW ACTUALLY CALCULATE FIXED EFFECTS ACROSS RUNS
        
        for key_name in all_runs_effects.keys():

            fixed_effects_T = np.nanmean(all_runs_effects[key_name]['effect_size'], 
                                        axis = 0)/np.sqrt((np.nanmean(all_runs_effects[key_name]['effect_variance'], 
                                                            axis = 0))) 

            # save
            filename = op.join(out_dir, 'fixed_effects_T_{kn}_contrast.npy'.format(kn = key_name))
            np.save(filename, fixed_effects_T[0])

    def average_betas(self, participant, fit_type = 'loo_run', weighted_avg = True, runs2load = [1,2,3,4], use_cv_r2 = True):

        """
        Helper function to load and average beta values
        from several runs

        Parameters
        ----------
        participant: str
            participant ID    
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        weighted_avg: bool
            if we want to average weighted by r2
        runs2load: list
            list with number ID for runs to load
        """

        all_betas = []
        all_r2 = []
        weight_arr = []

        # loop over runs
        for run_key in runs2load: 

            ## load estimates, and get betas and prediction
            soma_estimates = self.load_GLMestimates(participant, fit_type = fit_type, run_id = str(run_key).zfill(2))

            # append beta values
            all_betas.append(soma_estimates['betas'][np.newaxis,...])
            
            # append r2/CV-r2
            if fit_type == 'loo_run' and use_cv_r2 == True:
                tmp_arr = soma_estimates['cv_r2'].copy()
                tmp_arr[tmp_arr <= 0] = 0 # to remove negative values, that result in unpredictable outcome for weights
                weight_arr.append(tmp_arr[np.newaxis,...])
                all_r2.append(soma_estimates['cv_r2'][np.newaxis,...])
            else:
                all_r2.append(soma_estimates['r2'][np.newaxis,...])
                weight_arr.append(soma_estimates['r2'][np.newaxis,...])

        # stack
        all_betas = np.vstack(all_betas) #[runs, vertex, betas]
        all_r2 = np.vstack(all_r2) #[runs, vertex]
        weight_arr = np.vstack(weight_arr) #[runs, vertex]

        # calculate weighted average for each beta regressor
        if weighted_avg: 
            avg_betas = []
            for ind in np.arange(all_betas.shape[-1]):
                avg_betas.append(np.ma.average(all_betas[...,ind], axis=0, weights=weight_arr))
            avg_betas = np.stack(avg_betas, axis = -1).data
        else:
            avg_betas = np.nanmean(all_betas, axis = 0)
        
        avg_r2 = np.nanmean(all_r2, axis = 0)

        return avg_betas, avg_r2

    def fit_data(self, participant, fit_type = 'mean_run', hrf_model = 'glover', 
                                    custom_dm = True, keep_b_evs = False):

        """ fit glm model to participant data (averaged over runs)
        
        Parameters
        ----------
        participant: str
            participant ID    
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands) 
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM   
        hrf_model: str
            type of hrf to use (when custom_hrf = False)     
        """

        print('running GLM...')
        
        # get start time
        start_time = datetime.datetime.now()

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        ## Load data
        # get list with gii files
        gii_filenames = self.get_proc_file_list(participant, file_ext = self.proc_file_ext)

        if fit_type == 'mean_run':         
            # load data of all runs and average
            data2fit = self.load_data4fitting(gii_filenames, average = True)[np.newaxis,...] # [1, vertex, TR]

        elif fit_type == 'loo_run':
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            # leave one run out, load others and average
            data2fit = []
            for lo_run_key in run_loo_list:
                print('Leaving run-{r} out'.format(r = str(lo_run_key).zfill(2)))
                data2fit.append(self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(lo_run_key).zfill(2)) not in file], 
                                                        average = True)[np.newaxis,...])
            data2fit = np.vstack(data2fit) # [runs, vertex, TR]

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = data2fit.shape[-1], 
                                                hrf_model = hrf_model)

        # plot design matrix and save just to check if everything fine
        plot = plot_design_matrix(design_matrix)
        fig = plot.get_figure()
        hrf_model = 'custom' if custom_dm else hrf_model
        fig.savefig(op.join(out_dir,'design_matrix_HRF-%s.png'%hrf_model), dpi=100,bbox_inches = 'tight')

        ## Fit glm for all vertices
        for rind, data in enumerate(data2fit):
            
            print('Fitting GLM')

            # set filename to save estimates
            if fit_type == 'mean_run':      
                estimates_filename = op.join(out_dir, 'estimates_run-mean.npy')
            elif fit_type == 'loo_run':
                estimates_filename = op.join(out_dir, 'estimates_loo_run-{ri}.npy'.format(ri = str(run_loo_list[rind]).zfill(2)))

            ## actually fit
            soma_params = Parallel(n_jobs=16)(delayed(self.fit_glm_tc)(vert, design_matrix.values) for _,vert in enumerate(data))

            estimates_dict = {}
            estimates_dict['prediction'] = np.array([soma_params[i][0] for i in range(data.shape[0])])
            estimates_dict['betas'] = np.array([soma_params[i][1] for i in range(data.shape[0])])
            estimates_dict['r2'] = np.array([soma_params[i][2] for i in range(data.shape[0])])
            estimates_dict['mse'] = np.array([soma_params[i][3] for i in range(data.shape[0])])

            # calculate CV rsq
            if fit_type == 'loo_run':

                # load left out data
                other_run_data = self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(run_loo_list[rind]).zfill(2)) in file], average = True)
                
                # get CV-r2
                estimates_dict['cv_r2'] = np.nan_to_num(1-np.sum((other_run_data - estimates_dict['prediction'])**2, axis=-1)/(other_run_data.shape[-1]*other_run_data.var(-1)))

            # save estimates dict
            np.save(estimates_filename, estimates_dict)

        # Print duration, for bookeeping
        end_time = datetime.datetime.now()
        print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                        start_time = start_time,
                        end_time = end_time,
                        dur  = end_time - start_time))
        
    def make_COM_maps(self, participant, region = 'face', custom_dm = True, fixed_effects = True, nr_TRs = 141,
                        hrf_model = 'glover', z_threshold = 3.1, fit_type = 'mean_run', keep_b_evs = False):

        """ Make COM maps for a specific region
        given betas from GLM fit
        
        Parameters
        ----------
        participant: str
            participant ID 
        region: str
            region name 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        fixed_effects: bool
            if we want to use fixed effects across runs  
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)    
        hrf_model: str
            type of hrf to use (when custom_hrf = False) 
        z_threshold: float
            z-score threshold            
        """

        # if we want to used loo betas, and fixed effects t-stat
        if (fit_type == 'loo_run') and (fixed_effects == True): 
            ## make new out dir, depeding on our HRF approach
            out_dir = op.join(self.COM_outputdir, 'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)
            # if output path doesn't exist, create it
            os.makedirs(out_dir, exist_ok = True)

            # path where Region contrasts were stored
            stats_dir = op.join(self.stats_outputdir, 'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)

            # get list with gii files
            gii_filenames = self.get_proc_file_list(participant, file_ext = self.proc_file_ext)
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            ## get average beta values 
            betas, _ = self.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)

        else:
            ## make new out dir, depeding on our HRF approach
            out_dir = op.join(self.COM_outputdir, 'sub-{sj}'.format(sj = participant), fit_type)
            # if output path doesn't exist, create it
            os.makedirs(out_dir, exist_ok = True)

            # path where Region contrasts were stored
            stats_dir = op.join(self.stats_outputdir, 'sub-{sj}'.format(sj = participant), fit_type)

            # load GLM estimates, and get betas and prediction
            betas = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                            allow_pickle=True).item()['betas']

        # load z-score localizer area, for region movements
        region_mask = self.load_zmask(region = region, filepth = stats_dir, fit_type = fit_type, 
                                    fixed_effects = fixed_effects, z_threshold = z_threshold, keep_b_evs = keep_b_evs)

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = nr_TRs, 
                                                hrf_model = hrf_model)

        ## get beta values for region 
        # and z mask
        if region == 'face':
            region_betas = self.get_region_betas(betas, region = 'face', dm = design_matrix)
        elif region == 'upper_limb':
            region_betas = {}
            region_betas['R'] = self.get_region_betas(betas, region = 'right_hand', dm = design_matrix)
            region_betas['L'] = self.get_region_betas(betas, region = 'left_hand', dm = design_matrix)
            if keep_b_evs: # if we are looking at both hands
                region_betas['B'] = self.get_region_betas(betas, region = 'both_hand', dm = design_matrix)
        
        # calculate COM for all vertices
        if isinstance(region_betas, dict):
            for side in region_betas.keys():
                COM_all = self.COM(region_betas[side])
            
                COM_surface = np.zeros(region_mask[side].shape); COM_surface[:] = np.nan
                COM_surface[np.where((~np.isnan(region_mask[side])))[0]] = COM_all[np.where((~np.isnan(region_mask[side])))[0]]
                # save
                np.save(op.join(out_dir, 'COM_reg-{r}_{s}.npy'.format(r = region, s = side)), COM_surface)
                np.save(op.join(out_dir, 'zmask_reg-{r}_{s}.npy'.format(r = region, s = side)), region_mask[side])

        else:
            COM_all = self.COM(region_betas)
            
            COM_surface = np.zeros(region_mask.shape); COM_surface[:] = np.nan
            COM_surface[np.where((~np.isnan(region_mask)))[0]] = COM_all[np.where((~np.isnan(region_mask)))[0]]
            # save
            np.save(op.join(out_dir, 'COM_reg-{r}.npy'.format(r = region)), COM_surface)
            np.save(op.join(out_dir, 'zmask_reg-{r}.npy'.format(r = region)), region_mask)

    def get_region_betas(self, betas, region = 'face', dm = None):

        """ Helper function to subselect betas for a given region identifier
        
        Parameters
        ----------
        betas: arr
            array with all beta values [vertex, nregs]
        region: str
            region name (ex: face, right_hand, etc)
        dm: dataframe
            design matrix 
        """
        region_regs = self.MRIObj.params['fitting']['soma']['all_contrasts'][region]
        region_betas = [betas[..., np.where((dm.columns == reg))[0][0]] for reg in region_regs]
        region_betas = np.vstack(region_betas)

        return region_betas

    def load_zmask(self, region = 'face', filepth = None, fit_type = 'loo_run', 
                        fixed_effects = True, z_threshold = 3.1, keep_b_evs = True):

        """ load z score mask,
        that delimites functional region of interest
        
        Parameters
        ----------
        region: str
            region name
        filepth: str
            absolute path where contrast stats are stored
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        fixed_effects: bool
            if we want to use fixed effects across runs  
        z_threshold: float
            z-score threshold   
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)         
        """

        # load z-score localizer area, for region movements
        if region == 'face':
            if (fit_type == 'loo_run') and (fixed_effects == True): 
                z_score_region = np.load(op.join(filepth, 'fixed_effects_T_{r}_contrast.npy'.format(r=region)), 
                                    allow_pickle=True)
            else:
                z_score_region = np.load(op.join(filepth, 'stats_{r}_vs_all_contrast.npy'.format(r=region)), 
                                    allow_pickle=True).item()['z_score']

            # mask for relevant vertices
            region_mask = z_score_region.copy(); 
            region_mask[z_score_region < z_threshold] = np.nan
        else:
            if (fit_type == 'loo_run') and (fixed_effects == True): 
                z_score_region = np.load(op.join(filepth, 'fixed_effects_T_{r}_RvsL_contrast.npy'.format(r=region)), 
                                    allow_pickle=True)
            else:
                # we are going to look at left and right individually, so there are 2 region masks
                z_score_region = np.load(op.join(filepth, 'stats_{r}_RvsL_contrast.npy'.format(r=region)), 
                                        allow_pickle=True).item()['z_score']

            region_mask = {}
            region_mask['R'] = z_score_region.copy()
            region_mask['R'][z_score_region < 0] = np.nan
            region_mask['L'] = z_score_region.copy()
            region_mask['L'][z_score_region > 0] = np.nan

            if keep_b_evs: # if we are looking at both hands/legs
                if (fit_type == 'loo_run') and (fixed_effects == True): 
                    z_score_region = np.load(op.join(filepth, 'fixed_effects_T_{r}_contrast.npy'.format(r=region)), 
                                        allow_pickle=True)
                else:
                    z_score_region = np.load(op.join(filepth, 'stats_{r}_vs_all_contrast.npy'.format(r=region)), 
                                        allow_pickle=True).item()['z_score']

                region_mask['B'] = z_score_region.copy()
                region_mask['B'][z_score_region < z_threshold] = np.nan

        return region_mask # note, for face is array, for upper_limb is dict with mask for each side (or both)

    def f_goodness_of_fit(self, participant, hrf_model = 'glover', custom_dm = True, fit_type = 'mean_run',
                                keep_b_evs = False, alpha_fdr = 0.01):

        """ Calculate F value for participant
        Comparing full model vs reduced model (only intercept)
        Requires GLM fitting to have been done before
        
        Parameters
        ----------
        participant: str
            participant ID  
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)   
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)    
        hrf_model: str
            type of hrf to use (when custom_hrf = False)         
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.stats_outputdir, 'sub-{sj}'.format(sj = participant), 'F_stat', fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # get list with gii files
        gii_filenames = self.get_proc_file_list(participant, file_ext = self.proc_file_ext)
        
        if fit_type == 'mean_run':         
            # load data of all runs and average
            data2fit = self.load_data4fitting(gii_filenames, average = True)[np.newaxis,...] # [1, vertex, TR]

        elif fit_type == 'loo_run':
            # get all run lists
            run_loo_list = self.get_run_list(gii_filenames)

            # leave one run out, load others and average
            data2fit = []
            for lo_run_key in run_loo_list:
                print('Leaving run-{r} out'.format(r = str(lo_run_key).zfill(2)))
                data2fit.append(self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(lo_run_key).zfill(2)) not in file], 
                                                        average = True)[np.newaxis,...])
            data2fit = np.vstack(data2fit) # [runs, vertex, TR]

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = data2fit.shape[-1], 
                                                hrf_model = hrf_model)

        for rind, data in enumerate(data2fit):

            # fit intercept only model

            # set filename to save estimates
            if fit_type == 'mean_run':      
                estimates_filename = op.join(out_dir, 'reduced_model_run-mean.npy')
                stats_filename = op.join(out_dir, 'Fstat_run-mean.npy')
            elif fit_type == 'loo_run':
                estimates_filename = op.join(out_dir, 'reduced_model_loo_run-{ri}.npy'.format(ri = str(run_loo_list[rind]).zfill(2)))
                stats_filename = op.join(out_dir, 'Fstat_loo_run-{ri}.npy'.format(ri = str(run_loo_list[rind]).zfill(2)))

            if not op.isfile(estimates_filename):
                reduced_model_params = Parallel(n_jobs=16)(delayed(self.fit_glm_tc)(vert, 
                                                                                    design_matrix['constant'].values[..., np.newaxis]) for _,vert in enumerate(data))

                reduced_model = {}
                reduced_model['prediction'] = np.array([reduced_model_params[i][0] for i in range(data.shape[0])])
                reduced_model['betas'] = np.array([reduced_model_params[i][1] for i in range(data.shape[0])])
                reduced_model['r2'] = np.array([reduced_model_params[i][2] for i in range(data.shape[0])])
                reduced_model['mse'] = np.array([reduced_model_params[i][3] for i in range(data.shape[0])])

                # save estimates dict
                np.save(estimates_filename, reduced_model)
            else:
                reduced_model = np.load(estimates_filename, allow_pickle=True).item()

            ## load full model prediction
            full_model = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                                fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                                allow_pickle=True).item()

            ## now calculate F stat
            F_stat, p_values_F = self.get_Fstat(data, rmodel_pred = reduced_model['prediction'], 
                                                fmodel_pred = full_model['prediction'], 
                                                num_regs = len([val for val in design_matrix.columns if 'constant' not in val]))
            
            # and do FDR correction
            F_stat_fdr = self.fdr_correct(alpha_fdr = alpha_fdr, p_values = p_values_F, stat_values = F_stat)

            np.save(stats_filename, {'Fstat': F_stat, 'p_val': p_values_F, 'Fstat_FDR': F_stat_fdr})


class somaRF_Model(somaModel):

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
            self.outputdir = op.join(self.MRIObj.derivatives_pth, 'somaRF_fits')

    
    def gauss1D_cart(self, x, mu=0.0, sigma=1.0):
        
        """gauss1D_cart
        gauss1D_cart takes a 1D array x, a mean and standard deviation,
        and produces a gaussian with given parameters, with a peak of height 1.
        Parameters
        ----------
        x : numpy.ndarray (1D)
            space on which to calculate the gauss
        mu : float, optional
            mean/mode of gaussian (the default is 0.0)
        sigma : float, optional
            standard deviation of gaussian (the default is 1.0)
        Returns
        -------
        numpy.ndarray
            gaussian values at x
        """

        return np.exp(-((x-mu)**2)/(2*sigma**2))
    
    def create_grid_predictions(self, grid_centers, grid_sizes, reg_names = []):

        """ 
            create grid prediction timecourses

        Parameters
        ----------
        grid_centers: arr
            grid array with center positions for RF
        grid_sizes: arr
            grid array with RF sizes
        reg_names: list/arr
            regressor names (cond)

        """

        # all combinations of center position and size for grid values given
        self.grid_mus, self.grid_sigmas = np.meshgrid(grid_centers, grid_sizes)

        ## create grid RFs
        grid_rfs = self.gauss1D_cart(np.arange(len(reg_names))[..., np.newaxis], 
                        mu = self.grid_mus.ravel(), 
                        sigma = self.grid_sigmas.ravel())

        return grid_rfs # [nr_betas, nr predictions] 

    def find_best_pred(self, pred_arr, data_tc):

        """ 
        computes best-fit rsq and slope for vertex 
        (assumes baseline is 0)

        Parameters
        ----------
        pred_arr: arr
            (grid) prediction array [TR, nr predictions]
        data_tc: arr
            data timecourse

        """

        # find least-squares solution to a linear matrix equation
        grid_fits = [scipy.linalg.lstsq(pred_arr[...,i][...,np.newaxis], data_tc)[:2] for i in range(pred_arr.shape[-1])]
        grid_fits = np.vstack(grid_fits)

        # best prediction is one that minimizes residuals
        best_pred_ind = np.nanargmin(grid_fits[...,-1])

        # get slope and residual of best fitting prediction
        slope = grid_fits[best_pred_ind][0][0]
        resid = grid_fits[best_pred_ind][-1]

        # calculate r2 = (1 - residual / (n * y.var()))
        r2 = np.nan_to_num(1 - (resid / (data_tc.shape[0] * data_tc.var())))

        return best_pred_ind, slope, r2

    def get_grid_params(self, grid_predictions, betas_vert):

        """
        fit grid predictions, and return best fitting
        center, size, slope and r2
        """
        
        if np.isnan(betas_vert).any(): # if nan values, then return nan
            return {'mu': np.nan, 'size': np.nan, 'slope': np.nan,'r2': np.nan}

        else:
            # first find best grid prediction for vertex
            best_pred_ind, slope, r2 = self.find_best_pred(grid_predictions, betas_vert)

            # return grid params in dict
            return {'mu': self.grid_mus.ravel()[best_pred_ind],
                    'size': self.grid_sigmas.ravel()[best_pred_ind],
                    'slope': slope,
                    'r2': r2}

    def fit_betas(self, betas, regressor_names = [], region2fit = None, nr_grid = 100, n_jobs = 8):

        """
        given a data array [vertex, betas]
        fit gaussian population Response Field 
        for betas of region

         Parameters
        ----------
        betas: arr
            data to be fitted [vertex, betas]. 
            betas must be in same number as regressor_names (columns of design matrix)
        regressor_names: list
            list with all regressor names, in same order as betas
        region2fit: str
            region that we are fitting (face, right_hand, left_hand)

        """

        # set list with regressor names for given region
        regs2fit = self.MRIObj.params['fitting']['soma']['all_contrasts'][region2fit]

        # make grid of center position and size
        grid_center = np.linspace(0, len(regs2fit), nr_grid) - .5
        grid_size = np.linspace(0.25, len(regs2fit)-1, nr_grid)

        # create grid predictions
        grid_predictions = self.create_grid_predictions(grid_center, grid_size, reg_names = regs2fit)

        ## get regressor indices
        reg_inds = [ind for ind, name in enumerate(regressor_names) if name in regs2fit]
        # and only select those vertices
        betas2fit = betas[...,reg_inds]

        ## actually fit
        results = Parallel(n_jobs=n_jobs)(delayed(self.get_grid_params)(grid_predictions,
                                                                    betas2fit[vert])
                                                                for vert in tqdm(range(betas2fit.shape[0])))

        return results

    def fit_data(self, participant, somaModelObj = None, betas_model = 'glm',
                                    fit_type = 'mean_run', nr_grid = 100, n_jobs = 16,
                                    region_keys = ['face', 'right_hand', 'left_hand'], nr_TRs = 141,
                                    hrf_model = 'glover', custom_dm = True, keep_b_evs = False):

        """ fit gauss population Response Field model to participant betas 
        (from previously run GLM)
        
        Parameters
        ----------
        participant: str
            participant ID   
        somaModelObj : soma Model object
            object from one of the classes defined in soma_model  
        betas_model: str
            name of the model
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        nr_grid: int
            resolution of grid fit   
        n_jobs: int
            number of jobs to use in parallel fitting
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)    
        hrf_model: str
            type of hrf to use (when custom_hrf = False) 
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), betas_model, fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # if leave one out, get average of CV betas
        if fit_type == 'loo_run':
            # get all run lists
            run_loo_list = somaModelObj.get_run_list(somaModelObj.get_proc_file_list(participant, 
                                                                                     file_ext = self.proc_file_ext))

            ## get average beta values 
            betas, _ = somaModelObj.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)
        else:
            # load GLM estimates, and get betas and prediction
            betas = somaModelObj.load_GLMestimates(participant, fit_type = fit_type, run_id = None)['betas']

        ## Get DM
        design_matrix = self.load_design_matrix(participant, keep_b_evs = keep_b_evs, 
                                                custom_dm = custom_dm, nTRs = nr_TRs, 
                                                hrf_model = hrf_model)

        # fit RF model per region
        for region in region_keys:

            print('fitting RF model for %s'%region)

            results = self.fit_betas(betas, 
                                    regressor_names = design_matrix.columns,
                                    region2fit = region,
                                    nr_grid = nr_grid, n_jobs = n_jobs)

            # converting dict to proper format
            final_results = {k: [dic[k] for dic in results] for k in results[0].keys()}

            # save RF estimates dict
            np.save(op.join(out_dir, 'RF_grid_estimates_region-{r}.npy'.format(r=region)), final_results)

        # also save betas and dm in same directory
        np.save(op.join(out_dir, 'betas_glm.npy'), betas)
        design_matrix.to_csv(op.join(out_dir, 'DM.csv'), index=False)

    def return_prediction(self, mu = None, size = None, slope = None, nr_points = 4):

        """
        Helper function to return prediction
        """
        if isinstance(nr_points, np.ndarray) or isinstance(nr_points, list):
            timepoints = nr_points
        else:
            timepoints = np.arange(nr_points)
        return self.gauss1D_cart(timepoints, mu = mu, sigma = size) * slope

    def get_RF_COM(self, mu_arr = [], size_arr = [], slope_arr = [], nr_points = 4):

        """
        Helper function to get 
        receptive field center of mass

        Parameters
        ----------
        mu_arr: array
            RF center values
        size_arr: array
            RF size values
        slope_arr: array
            RF slope values
        nr_points: int
            number of points in RF
        """
        com_arr = [scipy.ndimage.measurements.center_of_mass(self.return_prediction(mu = mu_arr[i], 
                               size = size_arr[i], slope = slope_arr[i], nr_points = nr_points))[0] for i in np.arange(mu_arr.shape[0])]
        return np.array(com_arr)

    def load_estimates(self, participant, betas_model = 'glm', fit_type = 'mean_run',
                            region_keys = ['face', 'right_hand', 'left_hand', 'both_hand']):

        """
        Helper function to load fitted estimates
        """

        # set outputdir for participant
        RF_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                    betas_model, fit_type)

        # make estimates dict, for all regions of interest
        RF_estimates = {}
        for region in region_keys:
            RF_estimates[region] = np.load(op.join(RF_dir, 
                                                'RF_grid_estimates_region-{r}.npy'.format(r=region)), 
                                                allow_pickle=True).item()

        return RF_estimates


        

