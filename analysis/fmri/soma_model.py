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

import datetime

from glmsingle.glmsingle import GLM_single
from glmsingle.ols.make_poly_matrix import make_polynomial_matrix, make_projection_matrix

from visuomotor_utils import leave_one_out, split_half_comb, correlate_arrs, COM

import scipy

import cortex
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm


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
            array of bata values
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

    def get_atlas_roi_df(self, annot_pth = None, hemi_labels = {'lh': 'L', 'rh': 'R'}, 
                                base_str = 'HCP-MMP1.annot', hemi_vert_num = 163842, return_RGBA = False):

        """
        Get all atlas ROI labels and vertex indices, for both hemispheres
        and return it as pandas DF

        Assumes Glasser atlas (2016), might generalize to others in the future
        
        Parameters
        ----------
        annot_pth: str
            absolute path to atlas annotation files
        hemi_labels: dict
            key value pair of hemi labels (key: label for annot file, value: hemisphere label we will use later)
        base_str: str
            base name for annotation file
        hemi_vert_num: int
            number of vertices in one hemisphere (for bookeeping)  
        return_RGBA: bool
            if we want to return rgbt(a) as used in Glasser atlas figures, for each vertex
        """

        if annot_pth is None:
            annot_pth = op.join(self.MRIObj.derivatives_pth, 'Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG',
                            'HCP_PhaseTwo', 'Q1-Q6_RelatedParcellation210','MNINonLinear','fsaverage_LR32k')

        # make empty rgb dict (although we might not use it)
        atlas_rgb_dict = {'R': [], 'G': [], 'B': [], 'A': []}

        # fill atlas dataframe per hemi
        atlas_df = pd.DataFrame({'ROI': [], 'hemi_vertex': [], 'merge_vertex': [], 'hemisphere': []})

        for hemi in hemi_labels.keys():
            
            # get annotation file for hemisphere
            annotfile = [op.join(annot_pth,x) for x in os.listdir(annot_pth) if base_str in x and hemi in x][0]
            print('Loading annotations from %s'%annotfile)

            # read annotation file, save:
            # labels - annotation id at each vertex.
            # ctab - RGBT + label id colortable array.
            # names - The names of the labels
            h_labels, h_ctab, h_names = nib.freesurfer.io.read_annot(annotfile)

            # get labels as strings
            label_inds_names = [[ind, re.split('_', str(name))[1]] for ind,name in enumerate(h_names) if '?' not in str(name)]
    
            # fill df for each hemi roi
            for hemi_roi in label_inds_names:
                
                # vertice indices for that roi in that hemisphere
                hemi_roi_verts = np.where((h_labels == hemi_roi[0]))[0]
                # and for full surface
                surf_roi_verts = hemi_roi_verts + hemi_vert_num if hemi_labels[hemi] == 'R' else hemi_roi_verts
                
                atlas_df = pd.concat((atlas_df,
                                    pd.DataFrame({'ROI': np.tile(hemi_roi[-1], len(hemi_roi_verts)), 
                                                'hemi_vertex': hemi_roi_verts, 
                                                'merge_vertex': surf_roi_verts, 
                                                'hemisphere': np.tile(hemi_labels[hemi], len(hemi_roi_verts))})
                                    
                                    ),ignore_index=True)

            # if we want RGB + A save values, 
            # scaled to go from 0-1 (for pycortex plotting) and to have alpha and not T
            if return_RGBA:
                for _,lbl in enumerate(h_labels):
                    atlas_rgb_dict['R'].append(h_ctab[lbl,0]/255) 
                    atlas_rgb_dict['G'].append(h_ctab[lbl,1]/255) 
                    atlas_rgb_dict['B'].append(h_ctab[lbl,2]/255) 
                    atlas_rgb_dict['A'].append((255 - h_ctab[lbl,3])/255) 
                
        # allow to be used later one
        self.atlas_df = atlas_df

        if return_RGBA:
            return atlas_rgb_dict

    def get_roi_vert(self, roi_df, roi_list = [], hemi = 'BH'):

        """
        get vertex indices for an ROI, given an ROI df (usually for atlas, but not necessarily)
        and a list of labels
        for a specific hemisphere (or both)
        
        Parameters
        ----------
        roi_df: pd DataFrame
            dataframe with all ROIs names, hemispheres and vertices
        roi_list: list
            list of strings with ROI labels to load
        hemi: str
            which hemisphere (LH, RH or BH - both)
        """

        roi_vert = []

        for roi2plot in roi_list:
            if hemi == 'BH':
                roi_vert += list(roi_df[roi_df['ROI'] == roi2plot].merge_vertex.values.astype(int))
            else:
                roi_vert += list(roi_df[(roi_df['ROI'] == roi2plot) & \
                                (roi_df['hemisphere'] == hemi[0])].merge_vertex.values.astype(int))

        return np.array(roi_vert)

    def transform_roi_coords(self, orig_coords, fig_pth = None, roi_name = ''):

        """
        Use PCA to rotate x,y ROI coordinates along major axis (usually y)
        Note - Assumes we are providing ROI from 1 hemisphere only
        
        Parameters
        ----------
        orig_coords: arr
            x,y coordinate array for ROI [2, vertex]
        fig_pth: str
            if provided, will plot some sanity check plots and save in absolute dir
        """

        ## center the coordinates (to have zero mean)
        roi_coord_zeromean = np.vstack((orig_coords[0] - np.mean(orig_coords[0]),
                                        orig_coords[1] - np.mean(orig_coords[1])))

        ## get covariance matrix and eigen vector and values
        cov = np.cov(roi_coord_zeromean)
        evals, evecs = np.linalg.eig(cov)

        # Sort eigenvalues in decreasing order
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        # rotate
        theta = np.arctan((x_v1)/(y_v1))  
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation_mat * roi_coord_zeromean

        # get transformed coordenates
        x_transformed, y_transformed = transformed_mat.A
        roi_coord_transformed = np.vstack((x_transformed, y_transformed))

        if fig_pth is not None:
            
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            # plot zero centered ROI and major axis 
            fig, axes = plt.subplots(1,3, figsize=(18,4))

            axes[0].scatter(orig_coords[0], orig_coords[1])
            axes[0].axis('equal')
            axes[0].set_title('ROI original coords')
            
            scale = 20
            axes[1].plot([x_v1*-scale*2, x_v1*scale*2],
                        [y_v1*-scale*2, y_v1*scale*2], color='red')
            axes[1].plot([x_v2*-scale, x_v2*scale],
                        [y_v2*-scale, y_v2*scale], color='blue')
            axes[1].scatter(roi_coord_zeromean[0],
                            roi_coord_zeromean[1])
            axes[1].axis('equal')
            axes[1].set_title('ROI zero mean + major axis')

            axes[2].plot(roi_coord_zeromean[0],
                        roi_coord_zeromean[1], 'b.', alpha=.1)
            axes[2].plot(roi_coord_transformed[0],
                        roi_coord_transformed[1], 'g.')
            axes[2].axis('equal')
            axes[2].set_title('ROI rotated')

            fig.savefig(op.join(fig_pth, 'ROI_PCA_%s.png'%roi_name))

        return roi_coord_transformed

    def get_fs_coords(self, pysub = 'fsaverage', merge = True):

        """
        get freesurfer surface mesh coordinates
        """

        ## FreeSurfer surface file format: Contains a brain surface mesh in a binary format
        # Such a mesh is defined by a list of vertices (each vertex is given by its x,y,z coords) 
        # and a list of faces (each face is given by three vertex indices)

        #left, right = cortex.db.get_surf(params['processing']['space'], 'flat', merge=False)
        pts, polys = cortex.db.get_surf(pysub, 'flat', merge=True)

        return pts[:,0], pts[:,1], pts[:,2] # [vertex, axis] --> x, y, z

    def get_avg_events(self, participant, keep_b_evs = False):

        """ get events for participant (averaged over runs)
        
        Parameters
        ----------
        participant: str
            participant ID           
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

    def resample_arr(self, upsample_data, osf = 10, final_sf = 1.6):

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
        
        # original scale of data in seconds
        original_scale = np.arange(0, upsample_data.shape[-1]/osf, 1/osf)

        # cubic interpolation of predictor
        interp = scipy.interpolate.interp1d(original_scale, 
                                    upsample_data, 
                                    kind = "linear", axis=-1) #"cubic", axis=-1)
        
        desired_scale = np.arange(0, upsample_data.shape[-1]/osf, final_sf) # we want the predictor to be sampled in TR

        out_arr = interp(desired_scale)
        
        return out_arr


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
        else:
            self.outputdir = outputdir


    def make_custom_dm(self, events_df, osf = 100, data_len_TR = 141, 
                            TR = 1.6, hrf_params = [1,1,0], hrf_onset = 0):

        """
        Helper function to make custom dm, 
        from custom HRF
        """

        # list with regressor names
        regressor_names = events_df.trial_type.unique()

        # task duration in seconds
        task_dur_sec = data_len_TR * TR 

        # hrf timecourse in sec * TR !!
        hrf_tc = self.create_hrf_tc(hrf_params=hrf_params, osf = osf * TR, onset = hrf_onset)

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
            reg_resampled = self.resample_arr(reg_pred_osf_conv[0], osf = osf, final_sf = TR)/(osf) # dividing by osf so amplitude scaling not massive (but irrelevante because a.u.)
            #reg_resampled/=reg_resampled.max()
            
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
            r2 = np.nan_to_num(1 - (np.nansum((voxel - prediction)**2, axis=0)/ np.nansum((voxel**2), axis=0)))# and the rsq
        
        return prediction, betas, r2, mse

    def contrast_regions(self, participant, hrf_model = 'glover', custom_dm = True,
                                            z_threshold = 3.1, pval_1sided = True):

        """ Make simple contrasts to localize body regions
        (body, hands, legs) and contralateral regions (R vs L hand)

        Requires GLM fitting to have been done before
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.MRIObj.derivatives_pth, 'glm_stats', 
                                        'sub-{sj}'.format(sj = participant))
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # get list with gii files
        gii_filenames = self.get_soma_file_list(participant, 
                                            file_ext = self.MRIObj.params['fitting']['soma']['extension'])

        # load and average data of all runs
        all_data = self.load_data4fitting(gii_filenames)
        avg_data = np.nanmean(all_data, axis = 0) # [vertex, TR]

        # make average event file for pp, based on events file
        events_avg = self.get_avg_events(participant)

        if custom_dm: # if we want to make the dm 

            design_matrix = self.make_custom_dm(events_avg, 
                                                osf = 100, data_len_TR = avg_data.shape[-1], 
                                                TR = self.MRIObj.TR, 
                                                hrf_params = [1,1,0], hrf_onset = 0)
            hrf_model = 'custom'

        else: # if we want to use nilearn function

            # specifying the timing of fMRI frames
            frame_times = self.MRIObj.TR * (np.arange(avg_data.shape[-1]))

            # Create the design matrix, hrf model containing Glover model 
            design_matrix = make_first_level_design_matrix(frame_times,
                                                        events = events_avg,
                                                        hrf_model = hrf_model
                                                        )

        ## load estimates, and get betas and prediction
        soma_estimates = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                        'mean_run', 'estimates_run-mean.npy'), allow_pickle=True).item()
        betas = soma_estimates['betas']
        prediction = soma_estimates['prediction']

        # now make simple contrasts
        print('Computing simple contrasts')
        print('Using z-score of %0.2f as threshold for localizer' %z_threshold)

        reg_keys = ['face', 'upper_limb', 'lower_limb']
        region_regs = {'face': self.MRIObj.params['fitting']['soma']['all_contrasts']['face'],
                       'upper_limb': [val for val in self.MRIObj.params['fitting']['soma']['all_contrasts']['upper_limb'] if 'bhand' not in val],
                       'lower_limb': [val for val in self.MRIObj.params['fitting']['soma']['all_contrasts']['lower_limb'] if 'bleg' not in val]}

        loo_keys = leave_one_out(reg_keys) # loo for keys 

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
    
    def fit_data(self, participant, fit_type = 'mean_run', hrf_model = 'glover', custom_dm = True):

        """ fit glm model to participant data (averaged over runs)
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        print('running GLM...')
        
        # get start time
        start_time = datetime.datetime.now()

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # get list with gii files
        gii_filenames = self.get_soma_file_list(participant, 
                                            file_ext = self.MRIObj.params['fitting']['soma']['extension'])

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

        # make average event file for pp, based on events file
        events_avg = self.get_avg_events(participant)

        if custom_dm: # if we want to make the dm 

            design_matrix = self.make_custom_dm(events_avg, 
                                                osf = 100, data_len_TR = data2fit.shape[-1], 
                                                TR = self.MRIObj.TR, 
                                                hrf_params = [1,1,0], hrf_onset = 0)
            hrf_model = 'custom'

        else: # if we want to use nilearn function

            # specifying the timing of fMRI frames
            frame_times = self.MRIObj.TR * (np.arange(data2fit.shape[-1]))

            # Create the design matrix, hrf model containing Glover model 
            design_matrix = make_first_level_design_matrix(frame_times,
                                                        events = events_avg,
                                                        hrf_model = hrf_model
                                                        )

        # plot design matrix and save just to check if everything fine
        plot = plot_design_matrix(design_matrix)
        fig = plot.get_figure()
        fig.savefig(op.join(out_dir,'design_matrix_HRF-%s.png'%hrf_model), dpi=100,bbox_inches = 'tight')

        ## fit glm for all vertices
        for rind, data in enumerate(data2fit):
            
            print('Fitting GLM')

            if fit_type == 'mean_run':      
                estimates_filename = op.join(out_dir, 'estimates_run-mean.npy')
            elif fit_type == 'loo_run':
                estimates_filename = op.join(out_dir, 'estimates_loo_run-{ri}.npy'.format(ri = str(run_loo_list[rind]).zfill(2)))

            soma_params = Parallel(n_jobs=16)(delayed(self.fit_glm_tc)(vert, design_matrix.values) for _,vert in enumerate(data))

            estimates_dict = {}
            estimates_dict['prediction'] = np.array([soma_params[i][0] for i in range(data.shape[0])])
            estimates_dict['betas'] = np.array([soma_params[i][1] for i in range(data.shape[0])])
            estimates_dict['r2'] = np.array([soma_params[i][2] for i in range(data.shape[0])])
            estimates_dict['mse'] = np.array([soma_params[i][3] for i in range(data.shape[0])])

            # calculate CV rsq
            if fit_type == 'loo_run':

                # load left out data
                other_run_data = self.load_data4fitting([file for file in gii_filenames if 'run-{r}'.format(r = str(run_loo_list[rind]).zfill(2)) in file])
                
                estimates_dict['cv_r2'] = np.nan_to_num(1-np.sum((other_run_data - estimates_dict['prediction'])**2, axis=-1)/(other_run_data.shape[-1]*other_run_data.var(-1)))

            # save estimates dict
            np.save(estimates_filename, estimates_dict)

        # Print duration, for bookeeping
        end_time = datetime.datetime.now()
        print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                        start_time = start_time,
                        end_time = end_time,
                        dur  = end_time - start_time))
        
    def make_COM_maps(self, participant, region = 'face', custom_dm = True,
                        hrf_model = 'glover', z_threshold = 3.1):

        """ Make COM maps for a specific region
        given betas from GLM fit
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.MRIObj.derivatives_pth, 'glm_COM', 
                                        'sub-{sj}'.format(sj = participant))
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # path where Region contrasts were stored
        stats_dir = op.join(self.MRIObj.derivatives_pth, 'glm_stats', 
                                                'sub-{sj}'.format(sj = participant))

        # load GLM estimates, and get betas and prediction
        soma_estimates = np.load(op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                        'mean_run', 'estimates_run-mean.npy'), allow_pickle=True).item()
        betas = soma_estimates['betas']
        prediction = soma_estimates['prediction']
        r2 = soma_estimates['r2']

        # load z-score localizer area, for region movements
        if region == 'face':
            z_score_region = np.load(op.join(stats_dir, 'stats_{r}_vs_all_contrast.npy'.format(r=region)), 
                                    allow_pickle=True).item()['z_score']

            # mask for relevant vertices
            region_mask = z_score_region.copy(); 
            region_mask[z_score_region < z_threshold] = np.nan
        else:
            # we are going to look at left and right individually, so there are 2 region masks
            z_score_region = np.load(op.join(stats_dir, 'stats_{r}_RvsL_contrast.npy'.format(r=region)), 
                                    allow_pickle=True).item()['z_score']

            region_mask = {}
            region_mask['R'] = z_score_region.copy()
            region_mask['R'][z_score_region < 0] = np.nan
            region_mask['L'] = z_score_region.copy()
            region_mask['L'][z_score_region > 0] = np.nan

        # make average event file for pp, based on events file
        events_avg = self.get_avg_events(participant)

        if custom_dm: # if we want to make the dm 

            design_matrix = self.make_custom_dm(events_avg, 
                                                osf = 100, data_len_TR = prediction.shape[-1], 
                                                TR = self.MRIObj.TR, 
                                                hrf_params = [1,1,0], hrf_onset = 0)
            hrf_model = 'custom'

        else: # if we want to use nilearn function

            # specifying the timing of fMRI frames
            frame_times = self.MRIObj.TR * (np.arange(prediction.shape[-1]))

            # Create the design matrix, hrf model containing Glover model 
            design_matrix = make_first_level_design_matrix(frame_times,
                                                        events = events_avg,
                                                        hrf_model = hrf_model
                                                        )
        ## get beta values for region 
        # and z mask
        if region == 'face':
            region_regs = self.MRIObj.params['fitting']['soma']['all_contrasts'][region]
            region_betas = [betas[..., np.where((design_matrix.columns == reg))[0][0]] for reg in region_regs]
            region_betas = np.vstack(region_betas)

        elif region == 'upper_limb':

            region_betas = {}
            region_betas['R'] = [betas[..., np.where((design_matrix.columns == reg))[0][0]] for reg in self.MRIObj.params['fitting']['soma']['all_contrasts']['right_hand']]
            region_betas['R'] = np.vstack(region_betas['R'])

            region_betas['L'] = [betas[..., np.where((design_matrix.columns == reg))[0][0]] for reg in self.MRIObj.params['fitting']['soma']['all_contrasts']['left_hand']]
            region_betas['L'] = np.vstack(region_betas['L'])
        
        # calculate COM for all vertices
        if isinstance(region_betas, dict):
            
            for side in ['L', 'R']:
                COM_all = COM(region_betas[side])
            
                COM_surface = np.zeros(region_mask[side].shape); COM_surface[:] = np.nan
                COM_surface[np.where((~np.isnan(region_mask[side])))[0]] = COM_all[np.where((~np.isnan(region_mask[side])))[0]]
                # save
                np.save(op.join(out_dir, 'COM_reg-{r}_{s}.npy'.format(r = region, s = side)), COM_surface)
                np.save(op.join(out_dir, 'zmask_reg-{r}_{s}.npy'.format(r = region, s = side)), region_mask[side])

        else:
            COM_all = COM(region_betas)
            
            COM_surface = np.zeros(region_mask.shape); COM_surface[:] = np.nan
            COM_surface[np.where((~np.isnan(region_mask)))[0]] = COM_all[np.where((~np.isnan(region_mask)))[0]]
            # save
            np.save(op.join(out_dir, 'COM_reg-{r}.npy'.format(r = region)), COM_surface)
            np.save(op.join(out_dir, 'zmask_reg-{r}.npy'.format(r = region)), region_mask)



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

    
    def get_dm_glmsing(self, nr_TRs = 141, nr_runs = 1, sing_trial = False):

        """
        make glmsingle DM for all runs
        
        Parameters
        ----------
        nr_TRs: int
            number of TRs in run  
        sing_trial: bool
            if we want a dm of unique conditions or of single trial events        
        """

        ## get trial timings, in TR
        # initial baseline period - NOTE!! rounding up, will compensate by shifting hrf onset
        start_baseline_dur = int(np.round(self.MRIObj.soma_event_time_in_sec['empty']/self.MRIObj.TR)) 
        
        # trial duration (including ITIs)
        trial_dur = sum([self.MRIObj.soma_event_time_in_sec[name] for name in self.MRIObj.soma_trial_order])/self.MRIObj.TR

        if sing_trial == True:
            ## define DM [TR, conditions]
            # initialize at 0
            design_array = np.zeros((nr_TRs, len(self.soma_stim_labels)))

            # fill it with ones on stim onset
            for t, trl_cond in enumerate(self.soma_stim_labels):
                
                # fill it 
                design_array[int(start_baseline_dur + t*trial_dur), t] = 1
                #print(int(start_baseline_dur + t*trial_dur))

        else:
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

    def get_dm_from_events(self, participant, nr_TRs = 141, nr_runs = 1, sing_trial = False):

        """
        get dm from events timing
        
        Parameters
        ----------
        nr_TRs: int
            number of TRs in run  
        sing_trial: bool
            if we want a dm of unique conditions or of single trial events        
        """
        
        # get events onset
        events_df = self.get_avg_events(participant, keep_b_evs = True)
        onset_TR = np.ceil(events_df.onset.values/self.MRIObj.TR).astype(int)

        if sing_trial == True:

            design_array = np.zeros((nr_TRs, len(self.soma_stim_labels)))

            # fill it with ones on stim onset
            for t, trl_cond in enumerate(self.soma_stim_labels):
                
                # fill it 
                design_array[onset_TR[t], t] = 1

        else:
            design_array = np.zeros((nr_TRs, len(self.soma_cond_unique)))

            # fill it with ones on stim onset
            for t, trl_cond in enumerate(self.soma_stim_labels):
                
                # index for condition column
                cond_ind = np.where(self.soma_cond_unique == trl_cond)[0][0]
                
                # fill it 
                design_array[onset_TR[t], cond_ind] = 1
        
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
            nuisance component matrix 
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

        ## get run ID list for bookeeping
        self.run_list = self.get_run_list(gii_filenames)

        ## get nuisance matrix
        # if we want PSC data, then dont remove intercept (mean vol)
        # if psc_tc == True:
        #     maxpolydeg = [[val for val in r_list if val != 0] for r_list in maxpolydeg]

        combinedmatrix = self.get_nuisance_matrix(maxpolydeg,
                                                pcregressors = pcregressors,
                                                pcnum = pcnum,
                                                nr_TRs = np.array(all_data).shape[-1])[-1]
        ## denoise data
        data_out = [] 
        for r, cm in enumerate(combinedmatrix):

            data_denoised = cm.astype(np.float32) @ np.transpose(all_data[r])

            if psc_tc == True:
                mean_signal = all_data[r].mean(axis = -1)[np.newaxis, ...] #data_denoised.mean(axis = 0)[np.newaxis, ...]
                data_psc = data_denoised/np.absolute(mean_signal) #(data_denoised - mean_signal)/np.absolute(mean_signal)
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
        all_dm = self.get_dm_from_events(participant, 
                                        nr_TRs = np.array(all_data).shape[-1], 
                                        nr_runs = len(self.get_run_list(gii_filenames)))

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            'hrf_{h}'.format(h = self.glm_single_ops['hrf']))
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        ## make binary mask to exclude relevant voxels from noisepool
        binary_mask = self.make_correlation_mask(np.array(all_data), percentile_thresh = 99)

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
        opt['brainexclude'] = binary_mask.astype(int) 
        opt['brainR2'] = 100

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

        # save binary mask used
        flatmap = cortex.Vertex(binary_mask, 
                        self.MRIObj.sj_space,
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        #cortex.quickshow(flatmap, with_curvature=True,with_sulci=True)
        fig_name = op.join(out_dir, 'binary_mask_corr.png')
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

    def get_tc_stats(self, data_tc, betas, contrast = [], dm_run = [], hrf_tc = [], 
                            psc_tc = False, combinedmatrix = None, meanvol = None, pval_1sided = True):

        """ function to calculate the contrast statistics
        for a specific timecourse
        
        Parameters
        ----------
        data_tc: arr
            vertex timecourse     
        betas: arr
            beta values for vertex    
        contrast: arr/list
            contrast vector
        dm_run: arr
            design matrix for run [TR, cond] (not convolved)
        hrf_tc: arr
            hrf timecourse for vertex
        psc_tc: bool
            if we want to percent signal change predicted timecourse
        combinedmatrix: arr
            nuisance component matrix - NOTE! If we want to psc accurately, combinedmatrix should NOT have intercept
        meanvol: float
            mean volume for vertex

        """

        # convolve dm with hrf of vertex
        dm_conv = scipy.signal.convolve2d(dm_run, hrf_tc[..., np.newaxis])[0:dm_run.shape[0],:]  

        # calculate design variance
        design_var = self.design_variance(dm_conv, contrast)

        # get prediction timecourse
        prediction = self.get_prediction_tc(dm_run, hrf_tc, betas,
                                            psc_tc = psc_tc,
                                            combinedmatrix = combinedmatrix,
                                            meanvol = meanvol)

        # sum of squared errors
        sse = ((data_tc - prediction) ** 2).sum() 

        # degrees of freedom = N - P = timepoints - predictores
        df = (dm_conv.shape[0] - dm_conv.shape[1])

        t_val, p_val, z_score = self.calc_contrast_stats(betas = betas, 
                                                contrast = contrast, sse = sse, df = df, 
                                                design_var = design_var, pval_1sided = pval_1sided)

        return t_val, p_val, z_score

    def compute_roi_stats(self, participant, z_threshold = 3.1):

        """ 
            compute statistics for GLM single
            to localize general face, hand and leg region 
            and for left vs right regions

        Parameters
        ----------
        participant: str
            participant ID           
        """

        # print start time, for bookeeping
        start_time = datetime.datetime.now()

        ## outdir will be fits dir, which depends on our HRF approach
        out_dir = op.join(self.MRIObj.derivatives_pth, 'glmsingle_stats',
                                            'sub-{sj}'.format(sj = participant), 
                                            'hrf_{h}'.format(h = self.glm_single_ops['hrf']))
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        ## load estimates and fitting options
        results_glmsingle, fit_opts = self.load_results_dict(participant,
                                                            load_opts = True)

        ## Load PSC data for all runs
        data_psc = self.get_denoised_data(participant,
                                                maxpolydeg = fit_opts['maxpolydeg'],
                                                pcregressors = results_glmsingle['typed']['pcregressors'],
                                                pcnum = results_glmsingle['typed']['pcnum'],
                                                run_ID = None,
                                                psc_tc = True)

        ## get hrf for all vertices
        hrf_surf = self.get_hrf_tc(fit_opts['hrflibrary'], 
                                    results_glmsingle['typed']['HRFindex'])

        ## make design matrix
        all_dm = self.get_dm_glmsing(nr_TRs = np.array(data_psc).shape[1], 
                                   nr_runs = np.array(data_psc).shape[0])

        ## get average beta per condition
        avg_betas = self.average_betas_per_cond(results_glmsingle['typed']['betasmd'])

        ## get nuisance matrix
        combinedmatrix = self.get_nuisance_matrix(fit_opts['maxpolydeg'],
                                               pcregressors = results_glmsingle['typed']['pcregressors'],
                                               pcnum = results_glmsingle['typed']['pcnum'],
                                               nr_TRs = np.array(data_psc).shape[1])[-1]

        # now make simple contrasts
        print('Computing simple contrasts')
        print('Using z-score of %0.2f as threshold for localizer' %z_threshold)


        reg_keys = list(self.MRIObj.params['fitting']['soma']['all_contrasts'].keys())
        reg_keys.sort() # list of key names (of different body regions)

        loo_keys = leave_one_out(reg_keys) # loo for keys 

        # one broader region vs all the others
        for index,region in enumerate(reg_keys): 

            print('contrast for %s ' %region)

            # list of other contrasts
            other_contr = np.append(self.MRIObj.params['fitting']['soma']['all_contrasts'][loo_keys[index][0]],
                                    self.MRIObj.params['fitting']['soma']['all_contrasts'][loo_keys[index][1]])

            contrast = self.set_contrast(self.soma_cond_unique, 
                                    [self.MRIObj.params['fitting']['soma']['all_contrasts'][str(region)], other_contr],
                                [1,-len(self.MRIObj.params['fitting']['soma']['all_contrasts'][str(region)])/len(other_contr)],
                                num_cond=2)

            ## loop over runs
            run_stats = {}

            for run_ind, run_ID in enumerate(self.run_list):

                # set filename
                stats_filename = op.join(out_dir, 'stats_run-{rid}_{reg}_vs_all_contrast.npy'.format(rid = run_ID,
                                                                                                    reg = region))

                # compute contrast-related statistics
                soma_stats = Parallel(n_jobs=16)(delayed(self.get_tc_stats)(data_psc[run_ind][..., vert], 
                                                                        beta_vert, contrast = contrast, 
                                                                        dm_run = all_dm[run_ind], hrf_tc = hrf_surf[vert],
                                                                        psc_tc = True, combinedmatrix = combinedmatrix[run_ind], 
                                                                        meanvol = results_glmsingle['typed']['meanvol'][vert], 
                                                                        pval_1sided = True) for vert, beta_vert in enumerate(tqdm(avg_betas[run_ind])))
                soma_stats = np.vstack(soma_stats) # t_val, p_val, zscore

                run_stats[run_ID] = {}
                run_stats[run_ID]['t_val'] = soma_stats[..., 0]
                run_stats[run_ID]['p_val'] = soma_stats[..., 1]
                run_stats[run_ID]['zscore'] = soma_stats[..., 2]

                print('saving %s'%stats_filename)
                np.save(stats_filename, run_stats[run_ID])

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
                            
                contrast = self.set_contrast(self.soma_cond_unique,
                                                tasks, [1, -1], num_cond=2)

                ## loop over runs
                for run_ind, run_ID in enumerate(self.run_list):

                    # set filename
                    stats_filename = op.join(out_dir, 'stats_run-{rid}_{reg}_RvsL_contrast.npy'.format(rid = run_ID,
                                                                                                    reg = region))

                    # mask data - only significant voxels for region
                    region_ind = np.where((run_stats[run_ID]['zscore'] >= z_threshold))[0]

                    # compute contrast-related statistics
                    LR_stats = Parallel(n_jobs=16)(delayed(self.get_tc_stats)(data_psc[run_ind][..., vert], 
                                                                            avg_betas[run_ind][vert], contrast = contrast, 
                                                                            dm_run = all_dm[run_ind], hrf_tc = hrf_surf[vert],
                                                                            psc_tc = True, combinedmatrix = combinedmatrix[run_ind], 
                                                                            meanvol = results_glmsingle['typed']['meanvol'][vert], 
                                                                            pval_1sided = True) for vert in tqdm(region_ind))
                    LR_stats = np.vstack(LR_stats) # t_val, p_val, zscore

                    ## fill it for surface
                    LR_soma_stats = np.zeros((run_stats[run_ID]['zscore'].shape[0], LR_stats.shape[-1]))
                    LR_soma_stats[:] = np.nan
                    LR_soma_stats[region_ind,:] = LR_stats

                    print('saving %s'%stats_filename)
                    np.save(stats_filename, {'t_val': LR_soma_stats[..., 0],
                                            'p_val': LR_soma_stats[..., 1],
                                            'zscore': LR_soma_stats[..., 2]})

        # Print duration
        end_time = datetime.datetime.now()
        print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                        start_time = start_time,
                        end_time = end_time,
                        dur  = end_time - start_time))

    def zscore_reg_betas(self, betas):

        """
        z-score betas per condition
        """

        zbetas = []

        for i in np.arange(betas.shape[-1]): # assumes betas is [vert, regs]
            
            zbetas.append((betas[...,i] - np.mean(betas[...,i]))/np.std(betas[...,i]))

        return np.vstack(zbetas).T

    def make_correlation_mask(self, data_runs, percentile_thresh = 99, n_jobs = 8):

        """
        Calculate split-half correlation across all combinations of runs 
        Then shuffle timecourses in time and use that null distribution to find a
        threshold at a certain percentile
        Will return binary mask that can be used for noise pool 
        
        Parameters
        ----------
        data_runs: arr
            data array with all runs stacked [runs, vertex, TR]
        percentile_thresh: int/float
            qth percentile to use as threshold 
        """

        ## split runs in half and get unique combinations
        run_sh_lists = split_half_comb(np.arange(data_runs.shape[0]))

        # get correlation value for each combination
        corr_arr = []
        rnd_corr_arr = []

        for r in run_sh_lists:
            
            ## correlate the two halfs
            corr_arr.append(correlate_arrs(np.mean(np.array(data_runs)[list(r[0])], axis = 0), 
                                            np.mean(np.array(data_runs)[list(r[-1])], axis = 0), 
                                            n_jobs = n_jobs, shuffle_axis = None))
            
            ## correlate with randomized half
            rnd_corr_arr.append(correlate_arrs(np.mean(np.array(data_runs)[list(r[0])], axis = 0), 
                                            np.mean(np.array(data_runs)[list(r[-1])], axis = 0), 
                                            n_jobs = n_jobs, shuffle_axis = -1))

        # average values 
        avg_sh_corr = np.nanmean(corr_arr, axis = 0)
        avg_sh_rnd_corr = np.nanmean(rnd_corr_arr, axis = 0)

        ## make final mask
        # we want to exclude vertices below threshold
        binary_mask = np.ones(avg_sh_corr.shape)
        binary_mask[avg_sh_corr >= np.nanpercentile(avg_sh_rnd_corr, percentile_thresh)] = 0 # we want to set to 0 the ones that are not in the noise pool 

        return binary_mask


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
        else:
            self.outputdir = outputdir
    
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
        grid_size = np.linspace(0.2, len(regs2fit), nr_grid)

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
                                    region_keys = ['face', 'right_hand', 'left_hand'],
                                    hrf_model = 'glover', custom_dm = True):

        """ fit gauss population Response Field model to participant betas 
        (from previously run GLM)
        
        Parameters
        ----------
        participant: str
            participant ID           
        """

        ## make new out dir, depeding on our HRF approach
        out_dir = op.join(self.outputdir, 'sub-{sj}'.format(sj = participant), betas_model, fit_type)
        # if output path doesn't exist, create it
        os.makedirs(out_dir, exist_ok = True)

        # should add an if statement, for the case when we load glm single betas
        # load GLM estimates, and get betas and prediction
        soma_estimates = np.load(op.join(somaModelObj.outputdir, 'sub-{sj}'.format(sj = participant), 
                                        fit_type, 'estimates_run-mean.npy'), allow_pickle=True).item()
        betas = soma_estimates['betas']
        prediction = soma_estimates['prediction']
        r2 = soma_estimates['r2']

        # make average event file for pp, based on events file
        events_avg = somaModelObj.get_avg_events(participant)

        if custom_dm: # if we want to make the dm 

            # and design matrix
            design_matrix = somaModelObj.make_custom_dm(events_avg, 
                                                        osf = 100, data_len_TR = prediction.shape[-1], 
                                                        TR = self.MRIObj.TR, 
                                                        hrf_params = [1,1,0], hrf_onset = 0)

            hrf_model = 'custom'

        else: # if we want to use nilearn function

            # specifying the timing of fMRI frames
            frame_times = self.MRIObj.TR * (np.arange(prediction.shape[-1]))

            # Create the design matrix, hrf model containing Glover model 
            design_matrix = make_first_level_design_matrix(frame_times,
                                                        events = events_avg,
                                                        hrf_model = hrf_model
                                                        )

        # fit RF model per region
        for region in region_keys:

            results = self.fit_betas(betas, 
                                    regressor_names = design_matrix.columns,
                                    region2fit = region,
                                    nr_grid = nr_grid, n_jobs = n_jobs)

            # converting dict to proper format
            final_results = {k: [dic[k] for dic in results] for k in results[0].keys()}

            # save RF estimates dict
            np.save(op.join(out_dir, 'RF_grid_estimates_region-{r}.npy'), final_results)

        # also save betas and dm in same directory
        np.save(op.join(out_dir, 'betas_glm.npy'), betas)
        design_matrix.to_csv(op.join(out_dir, 'DM.csv'), index=False)


        

