import numpy as np
import os
import os.path as op

import cortex
import matplotlib.pyplot as plt

from prfpy.rf import gauss2D_iso_cart

from matplotlib.backend_bases import MouseButton

from nilearn.glm.first_level import make_first_level_design_matrix

class visualize_on_click:

    def __init__(self, MRIObj, pRFModelObj = None, SomaModelObj = None, SomaRF_ModelObj = None,
                        pRF_data = [], soma_data = [],
                        prf_dm = [], max_ecc_ext = 5.5,
                        pysub = 'fsaverage', flatmap_height = 2048, full_figsize = (12, 8)):

        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        pRFModelObj: pRF Model object
            object from one of the classes defined in prf_model.pRF_model
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        # Load pRF and model object
        self.pRFModelObj = pRFModelObj
        self.SomaModelObj = SomaModelObj
        self.SomaRF_ModelObj = SomaRF_ModelObj

        ## data to be plotted 
        self.pRF_data = pRF_data
        self.soma_data = soma_data

        ## figure settings
        self.flatmap_height = flatmap_height
        self.full_figsize = full_figsize
        self.images = {}
        
        ## create pycortex vars
        self.mask, extents = cortex.quickflat.utils.get_flatmask(pysub, height = self.flatmap_height)
        self.vc = cortex.quickflat.utils._make_vertex_cache(pysub, height = self.flatmap_height)

        self.mask_index = np.zeros(self.mask.shape)
        self.mask_index[self.mask] = np.arange(self.mask.sum())

        # set prf dm
        self.prf_dm = prf_dm

        ## set grid of possible points in downsampled space
        if len(self.prf_dm) > 0:
            self.point_grid_2D = np.array(np.meshgrid(np.linspace(-1, 1, prf_dm.shape[0]) * max_ecc_ext,
                                         np.linspace(1, -1, prf_dm.shape[0]) * max_ecc_ext))
        else:
            self.point_grid_2D = None


    def set_figure(self, participant, pp_prf_est_dict = None, pp_prf_models = None,
                    task2viz = 'soma', prf_run_type = 'mean_run', soma_run_type = 'mean_run',
                    pRFmodel_name = 'css', somamodel_name = 'glm', custom_dm = True):

        """
        Set base figure with placeholders 
        for relevant plots
        Parameters
        ----------
        task2viz : str
            task to visualize, can be 'prf', 'soma' or 'both'
            
        """

        ## set task of interest
        self.task2viz = task2viz

        ## set participant ID
        self.participant = participant

        ## set run and session to load
        self.run_type = {'prf': prf_run_type, 'soma': soma_run_type}

        self.pRFmodel_name = pRFmodel_name
        self.somamodel_name = somamodel_name

        # load prf estimates, if they were not provided
        if self.pRFModelObj is not None:

            if pp_prf_est_dict is None:
                print('Load prf estimates')
            else:
                self.pp_prf_est_dict = pp_prf_est_dict
                self.pp_prf_models = pp_prf_models

            # when loading, dict has key-value pairs stored,
            # need to convert it to make it in same format as when fitting on the spot
            self.pRF_keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = pRFmodel_name)


        if self.SomaModelObj is not None:

            if somamodel_name in ['glm', 'somaRF']: # load estimates from standard GLM fitting
                
                # load GLM estimates, and get betas and prediction
                soma_estimates = np.load(op.join(self.SomaModelObj.outputdir, 'sub-{sj}'.format(sj = participant), 
                                                'mean_run', 'estimates_run-mean.npy'), allow_pickle=True).item()
                self.soma_betas = soma_estimates['betas']
                self.soma_prediction = soma_estimates['prediction']
                self.soma_r2 = soma_estimates['r2']

                # make average event file for pp, based on events file
                events_avg = self.SomaModelObj.get_avg_events(participant)

                if custom_dm: # if we want to make the dm 

                    design_matrix = self.SomaModelObj.make_custom_dm(events_avg, 
                                                        osf = 100, data_len_TR = self.soma_prediction.shape[-1], 
                                                        TR = self.SomaModelObj.MRIObj.TR, 
                                                        hrf_params = [1,1,0], hrf_onset = 0)
                    hrf_model = 'custom'

                else: # if we want to use nilearn function

                    # specifying the timing of fMRI frames
                    frame_times = self.SomaModelObj.MRIObj.TR * (np.arange(self.soma_prediction.shape[-1]))

                    # Create the design matrix, hrf model containing Glover model 
                    design_matrix = make_first_level_design_matrix(frame_times,
                                                                events = events_avg,
                                                                hrf_model = 'glover'
                                                                )

                self.soma_regressors = design_matrix.columns

                if somamodel_name == 'somaRF':

                    ## dir where RF estimates are saved
                    RF_dir = op.join(self.SomaRF_ModelObj.outputdir, 'sub-{sj}'.format(sj = participant), 'glm', 'mean_run')

                    # load and save in dict for easy access
                    self.RF_estimates = {'face': np.load(op.join(RF_dir, 'RF_grid_estimates_region-face.npy'), 
                                                            allow_pickle=True).item(),
                                        'LH': np.load(op.join(RF_dir, 'RF_grid_estimates_region-left_hand.npy'), 
                                                            allow_pickle=True).item(),
                                        'RH': np.load(op.join(RF_dir, 'RF_grid_estimates_region-right_hand.npy'), 
                                                            allow_pickle=True).item()
                                                            }
                    # set region regressor names and indices, for easy access
                    self.RF_reg_names = {'face': self.SomaRF_ModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['face'],
                                        'LH': self.SomaRF_ModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['left_hand'],
                                        'RH': self.SomaRF_ModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['right_hand']}

                    self.RF_reg_inds = {'face': [ind for ind, name in enumerate(self.soma_regressors) if name in self.RF_reg_names['face']],
                                        'LH': [ind for ind, name in enumerate(self.soma_regressors) if name in self.RF_reg_names['LH']],
                                        'RH': [ind for ind, name in enumerate(self.soma_regressors) if name in self.RF_reg_names['RH']]
                                        }

                                        
            
        ## set figure grid 
        self.full_fig = plt.figure(constrained_layout = True, figsize = self.full_figsize)

        if task2viz == 'both':

            gs = self.full_fig.add_gridspec(4, 3)

            self.flatmap_ax = self.full_fig.add_subplot(gs[:2, :])

            self.prf_timecourse_ax = self.full_fig.add_subplot(gs[2, :2])
            self.soma_timecourse_ax = self.full_fig.add_subplot(gs[3, :2])

            self.prf_ax = self.full_fig.add_subplot(gs[2, 2])

            self.flatmap_ax.set_title('flatmap')
            self.soma_timecourse_ax.set_title('Soma timecourse')
            self.prf_timecourse_ax.set_title('pRF timecourse')
            self.prf_ax.set_title('prf')
        
        elif task2viz in ['prf', 'pRF']:

            gs = self.full_fig.add_gridspec(4, 2)

            self.flatmap_ax = self.full_fig.add_subplot(gs[:3, :])

            self.prf_timecourse_ax = self.full_fig.add_subplot(gs[3, :1])

            self.prf_ax = self.full_fig.add_subplot(gs[3, 1])

            self.flatmap_ax.set_title('flatmap')
            self.prf_timecourse_ax.set_title('pRF timecourse')
            self.prf_ax.set_title('prf')

        elif task2viz == 'soma':
            
            if self.somamodel_name == 'glm':
                gs = self.full_fig.add_gridspec(4, 2)

                self.flatmap_ax = self.full_fig.add_subplot(gs[:3, :])

                self.soma_timecourse_ax = self.full_fig.add_subplot(gs[3, :1])

                self.betas_ax = self.full_fig.add_subplot(gs[3, 1])

                self.flatmap_ax.set_title('flatmap')
                self.soma_timecourse_ax.set_title('Soma timecourse')
                self.betas_ax.set_title('betas')
            
            elif self.somamodel_name == 'somaRF':

                gs = self.full_fig.add_gridspec(4, 6)

                self.flatmap_ax = self.full_fig.add_subplot(gs[:3, :])
                self.flatmap_ax.set_title('flatmap')

                self.soma_timecourse_ax = self.full_fig.add_subplot(gs[3, :3])
                self.soma_timecourse_ax.set_title('Soma timecourse')

                self.face_betas_ax = self.full_fig.add_subplot(gs[3, 3])
                self.face_betas_ax.set_title('face')
                self.RH_betas_ax = self.full_fig.add_subplot(gs[3, 4], sharey = self.face_betas_ax)
                self.RH_betas_ax.set_title('RH')
                self.LH_betas_ax = self.full_fig.add_subplot(gs[3, 5], sharey = self.face_betas_ax)
                self.LH_betas_ax.set_title('LH')


    def get_vertex_prf_model_tc(self, vertex):

        """
        Get model estimates for that vertex
        Parameters
        ----------
        vertex : int
            vertex index
            
        """

        # if we fitted hrf, need to also get that from params
        # and set model array

        estimates_arr = np.stack((self.pp_prf_est_dict[val][vertex] for val in self.pRF_keys))
        
        # define spm hrf
        spm_hrf = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].create_hrf(hrf_params = [1, 1, 0],
                                                                                                                                                            onset=self.pRFModelObj.hrf_onset)

        if self.pRFModelObj.fit_hrf:
            hrf = self.pp_prf_models[ 'sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].create_hrf(hrf_params = [1.0,
                                                                                                                                estimates_arr[-3],
                                                                                                                                estimates_arr[-2]],
                                                                                                                    onset=self.pRFModelObj.hrf_onset)
        
            self.pp_prf_models['sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].hrf = hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].return_prediction(*list(estimates_arr[:-3]))
        
        else:
            self.pp_prf_models['sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].hrf = spm_hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)]['{name}_model'.format(name = self.pRFModelObj.model_type)].return_prediction(*list(estimates_arr[:-1]))
            
        return model_arr[0], estimates_arr[-1]


    def plot_prf_tc(self, axis, timecourse = None, plot_model = True):

        """
        plot pRF timecourse for model and data
        Parameters
        ----------
        timecourse : arr
            data time course
            
        """
        
        # plotting will be in seconds
        time_sec = np.linspace(0, len(timecourse) * self.MRIObj.TR,
                               num = len(timecourse)) 
        
        axis.plot(time_sec, timecourse,'k--', label = 'data')
        
        if plot_model:
            prediction, r2 = self.get_vertex_prf_model_tc(self.vertex)
            axis.plot(time_sec, prediction, c = 'red',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
            print('pRF model R$^2$ = %.2f'%r2)
            
        axis.set_xlabel('Time (s)')#,fontsize=20, labelpad=20)
        axis.set_ylabel('BOLD signal change (%)')#,fontsize=20, labelpad=10)
        axis.set_xlim(0, len(timecourse)*self.MRIObj.TR)
        #axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
        
        return axis


    def plot_soma_tc(self, axis, timecourse = None, plot_model = True):

        """
        plot soma timecourse for model and data
        Parameters
        ----------
        timecourse : arr
            data time course
            
        """
        
        # plotting will be in seconds
        time_sec = np.linspace(0, len(timecourse)*self.MRIObj.TR,
                               num = len(timecourse)) 
        
        axis.plot(time_sec, timecourse,'k--', label = 'data')
        
        if plot_model:
            prediction = self.soma_prediction[self.vertex]
            r2 = self.soma_r2[self.vertex]
            axis.plot(time_sec, prediction, c = 'blue',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
            print('Soma GLM model R$^2$ = %.2f'%r2)
            

        axis.set_xlabel('Time (s)')#,fontsize=20, labelpad=20)
        axis.set_ylabel('BOLD signal change (%)')#,fontsize=20, labelpad=10)
        axis.set_xlim(0, len(timecourse)*self.MRIObj.TR)
        #axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
        
        return axis


    def redraw_vertex_plots(self, vertex, refresh):

        """
        redraw vertex
            
        """
        
        self.vertex = vertex

        print(refresh)

        if refresh: # if we want to clean up timecourses
            if self.task2viz in ['both', 'prf', 'pRF']:
                self.prf_timecourse_ax.clear()
            
            if self.task2viz in ['both', 'soma']:
                self.soma_timecourse_ax.clear()
            
        if self.task2viz in ['both', 'prf', 'pRF']:

            self.prf_timecourse_ax = self.plot_prf_tc(self.prf_timecourse_ax, timecourse = self.pRF_data[vertex])

            prf = gauss2D_iso_cart(self.point_grid_2D[0],
                               self.point_grid_2D[1],
                               mu = (self.pp_prf_est_dict['x'][vertex], 
                                     self.pp_prf_est_dict['y'][vertex]),
                               sigma = self.pp_prf_est_dict['size'][vertex]) #, alpha=0.6)

            self.prf_ax.clear()
            self.prf_ax.imshow(prf, cmap='cubehelix')
            self.prf_ax.axvline(self.prf_dm.shape[0]/2, color='white', linestyle='dashed', lw=0.5)
            self.prf_ax.axhline(self.prf_dm.shape[1]/2, color='white', linestyle='dashed', lw=0.5)
            #prf_ax.set_title(f"x: {self.pp_prf_est_dict['x'][vertex]}, y: {self.pp_prf_est_dict['y'][vertex]}")

            # just to check if exponent values make sense
            if self.pRFModelObj.model_type == 'css':
                print('pRF exponent = %.2f'%self.pp_prf_est_dict['ns'][vertex])

        # plot soma data (and model if provided) 
        if self.task2viz in ['both', 'soma']:
            if self.SomaModelObj:
                self.soma_timecourse_ax = self.plot_soma_tc(self.soma_timecourse_ax, timecourse = self.soma_data[vertex])
            elif self.soma_data:
                self.soma_timecourse_ax = self.plot_soma_tc(self.soma_timecourse_ax, timecourse = self.soma_data[vertex], plot_model = False) 

            if self.somamodel_name == 'glm':
                self.betas_ax.clear()
                self.betas_ax.bar(np.arange(self.soma_betas.shape[-1]), self.soma_betas[vertex])
                self.betas_ax.set_xticks(np.arange(self.soma_betas.shape[-1]))
                self.betas_ax.set_xticklabels(self.soma_regressors, rotation=80)
            
            elif self.somamodel_name == 'somaRF':
                self.face_betas_ax.clear()
                self.RH_betas_ax.clear()
                self.LH_betas_ax.clear()

                # plot betas for each region
                # face
                self.face_betas_ax.bar(np.arange(len(self.RF_reg_inds['face'])), self.soma_betas[vertex][self.RF_reg_inds['face']])
                self.face_betas_ax.set_xticks(np.arange(len(self.RF_reg_inds['face'])))
                self.face_betas_ax.set_xticklabels(self.RF_reg_names['face'], rotation=80)
                self.face_betas_ax.set_title('face')
                # add predicted tc
                self.face_betas_ax.plot(self.SomaRF_ModelObj.return_prediction(mu = self.RF_estimates['face']['mu'][vertex], 
                                                                            size = self.RF_estimates['face']['size'][vertex], 
                                                                            slope = self.RF_estimates['face']['slope'][vertex], 
                                                                            nr_points = len(self.RF_reg_inds['face'])),
                                        c = 'red',lw=3)

                # RH
                self.RH_betas_ax.bar(np.arange(len(self.RF_reg_inds['RH'])), self.soma_betas[vertex][self.RF_reg_inds['RH']])
                self.RH_betas_ax.set_xticks(np.arange(len(self.RF_reg_inds['RH'])))
                self.RH_betas_ax.set_xticklabels(self.RF_reg_names['RH'], rotation=80)
                self.RH_betas_ax.set_title('RH')
                # add predicted tc
                self.RH_betas_ax.plot(self.SomaRF_ModelObj.return_prediction(mu = self.RF_estimates['RH']['mu'][vertex], 
                                                                            size = self.RF_estimates['RH']['size'][vertex], 
                                                                            slope = self.RF_estimates['RH']['slope'][vertex], 
                                                                            nr_points = len(self.RF_reg_inds['RH'])),
                                        c = 'red',lw=3)

                # LH
                self.LH_betas_ax.bar(np.arange(len(self.RF_reg_inds['LH'])), self.soma_betas[vertex][self.RF_reg_inds['LH']])
                self.LH_betas_ax.set_xticks(np.arange(len(self.RF_reg_inds['LH'])))
                self.LH_betas_ax.set_xticklabels(self.RF_reg_names['LH'], rotation=80)
                self.LH_betas_ax.set_title('LH')
                # add predicted tc
                self.LH_betas_ax.plot(self.SomaRF_ModelObj.return_prediction(mu = self.RF_estimates['LH']['mu'][vertex], 
                                                                            size = self.RF_estimates['LH']['size'][vertex], 
                                                                            slope = self.RF_estimates['LH']['slope'][vertex], 
                                                                            nr_points = len(self.RF_reg_inds['LH'])),
                                        c = 'red',lw=3)


    def onclick(self, event):
        print('you pressed', event.button, event.xdata, event.ydata)

        if  event.button is MouseButton.RIGHT:
            refresh_fig = True
        else:
            refresh_fig = False
        
        if event.inaxes == self.flatmap_ax:
            xmin, xmax = self.flatmap_ax.get_xbound()
            ax_xrange = xmax-xmin
            ymin, ymax = self.flatmap_ax.get_ybound()
            ax_yrange = ymax-ymin

            rel_x = int(self.mask.shape[0] * (event.xdata-xmin)/ax_xrange)
            rel_y = int(self.mask.shape[1] * (event.ydata-ymin)/ax_yrange)
            clicked_pixel = (rel_x, rel_y)

            clicked_vertex = self.vc[int(
                self.mask_index[clicked_pixel[0], clicked_pixel[1]])]

            print(clicked_vertex)
            self.redraw_vertex_plots(clicked_vertex.indices[0], refresh_fig)
            plt.draw()
        
    def onkey(self, event, with_rois = True, with_sulci=True):
        
        # clear flatmap axis
        self.flatmap_ax.clear()

        if self.task2viz in ['prf', 'pRF']:

            if event.key == '1':  # pRF rsq
                cortex.quickshow(self.images['pRF_rsq'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('pRF rsq')
            elif event.key == '2':  # pRF eccentricity
                cortex.quickshow(self.images['ecc'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('pRF eccentricity')
            elif event.key == '3':  # pRF Size
                cortex.quickshow(self.images['size_fwhmax'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('pRF size (FWHMax)')
            elif event.key == '4':  # pRF PA
                cortex.quickshow(self.images['PA'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('pRF PA')
            elif (event.key == '5') & (self.pRFmodel_name == 'css'):  # pRF exponent
                cortex.quickshow(self.images['ns'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('pRF exponent')   
            
        elif self.task2viz in ['soma']:

            if event.key == '1':  # soma rsq
                cortex.quickshow(self.images['Soma_rsq'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('Soma rsq')
            elif event.key == '2':  # soma face
                cortex.quickshow(self.images['face'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('Soma face')
            elif event.key == '3':  # soma RH
                cortex.quickshow(self.images['RH'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('Soma Right Hand')
            elif event.key == '4':  # soma LH
                cortex.quickshow(self.images['LH'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                self.flatmap_ax.set_title('Soma Left Hand')

            if self.somamodel_name == 'somaRF':
                if event.key == '5':
                    cortex.quickshow(self.images['face_size'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                    self.flatmap_ax.set_title('Soma face size')
                elif event.key == '6':
                    cortex.quickshow(self.images['RH_size'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                    self.flatmap_ax.set_title('Soma RH size')
                elif event.key == '7':
                    cortex.quickshow(self.images['LH_size'], with_rois = with_rois, with_curvature = True, with_sulci = with_sulci, with_labels = False,
                            fig = self.flatmap_ax, with_colorbar = False)
                    self.flatmap_ax.set_title('Soma LH size')

        plt.draw()