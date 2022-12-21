import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

from visuomotor_utils import normalize, add_alpha2colormap

import cortex

class somaViewer:

    def __init__(self, somaModelObj, outputdir = None, pysub = 'fsaverage'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        somaModelObj : soma Model object
            object from one of the classes defined in soma_model
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # set object to use later on
        self.somaModelObj = somaModelObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots')
        else:
            self.outputdir = outputdir

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

        # pycortex subject to use
        self.pysub = pysub

    def plot_glasser_rois(self):

        """
        plot glasser atlas with specific color scheme for each ROI
        """

        fig_pth = op.join(self.outputdir, 'glasser_atlas')
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # get ROI color map
        atlas_rgb_dict = self.somaModelObj.get_atlas_roi_df(return_RGBA = True)

        # plot flatmap
        glasser = cortex.VertexRGB(np.array(atlas_rgb_dict['R']),
                           np.array(atlas_rgb_dict['G']), 
                           np.array(atlas_rgb_dict['B']),
                           alpha = np.array(atlas_rgb_dict['A']),
                           subject = self.pysub)

        #cortex.quickshow(glasser,with_curvature=True,with_sulci=True,with_colorbar=False)
        filename = op.join(fig_pth, 'glasser_flatmap.png')
        print('saving %s' %filename)
        _ = cortex.quickflat.make_png(filename, glasser, recache=True,
                                        with_colorbar=False,with_curvature=True,with_sulci=True)

        # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
        cutout_name = 'zoom_roi_left'
        _ = cortex.quickflat.make_figure(glasser,
                                        with_curvature=True,
                                        with_sulci=True,
                                        with_roi=True,
                                        with_colorbar=False,
                                        cutout=cutout_name,height=2048)
        filename = op.join(fig_pth, cutout_name+'_glasser_flatmap.png')
        print('saving %s' %filename)
        _ = cortex.quickflat.make_png(filename, glasser, recache=True,
                                        with_colorbar=False,with_curvature=True,with_sulci=True)

        # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
        cutout_name = 'zoom_roi_right'
        _ = cortex.quickflat.make_figure(glasser,
                                        with_curvature=True,
                                        with_sulci=True,
                                        with_roi=True,
                                        with_colorbar=False,
                                        cutout=cutout_name,height=2048)
        filename = op.join(fig_pth, cutout_name+'_glasser_flatmap.png')
        print('saving %s' %filename)
        _ = cortex.quickflat.make_png(filename, glasser, recache=True,
                                        with_colorbar=False,with_curvature=True,with_sulci=True)

        # save inflated 3D screenshots 
        cortex.export.save_3d_views(glasser, 
                            base_name = op.join(fig_pth,'3D_glasser'),
                            list_angles = ['lateral_pivot', 'medial_pivot', 'left', 'right', 'top', 'bottom',
                                       'left'],
                            list_surfaces = ['inflated', 'inflated', 'inflated', 'inflated','inflated','inflated',
                                          'flatmap'],
                            viewer_params=dict(labels_visible=[],
                                               overlays_visible=['rois','sulci']),
                            size=(1024 * 4, 768 * 4), trim=True, sleep=60)


    def plot_glmsingle_tc(self, participant, vertex, psc_tc = True):

        """
        quick function to 
        plot vertex timeseries and model fit 
        """

        fig_pth = op.join(self.outputdir, 'glmsingle_fits', 'single_vertex', 
                                            'sub-{sj}'.format(sj = participant), 
                                            'hrf_{h}'.format(h = self.somaModelObj.glm_single_ops['hrf']))
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        ## load estimates and fitting options
        results_glmsingle, fit_opts = self.somaModelObj.load_results_dict(participant,
                                                                    load_opts = True)

        ## Load PSC data for all runs
        # and nuisance matrix
        data_psc = self.somaModelObj.get_denoised_data(participant,
                                                maxpolydeg = fit_opts['maxpolydeg'],
                                                pcregressors = results_glmsingle['typed']['pcregressors'],
                                                pcnum = results_glmsingle['typed']['pcnum'],
                                                run_ID = None,
                                                psc_tc = psc_tc)

        ## get hrf for all vertices
        hrf_surf = self.somaModelObj.get_hrf_tc(fit_opts['hrflibrary'], 
                                                results_glmsingle['typed']['HRFindex'])

        ## make design matrix
        all_dm = self.somaModelObj.get_dm_glmsing(nr_TRs = np.array(data_psc).shape[1], 
                                                    nr_runs = np.array(data_psc).shape[0])

        ## get average beta per condition
        avg_betas = self.somaModelObj.average_betas_per_cond(results_glmsingle['typed']['betasmd'])

        ## get nuisance matrix
        combinedmatrix = self.somaModelObj.get_nuisance_matrix(fit_opts['maxpolydeg'],
                                                            pcregressors = results_glmsingle['typed']['pcregressors'],
                                                            pcnum = results_glmsingle['typed']['pcnum'],
                                                            nr_TRs = np.array(data_psc).shape[1])[-1]

        ## get predicted timecourse for vertex 
        # for all runs
        predicted_tc_psc = [self.somaModelObj.get_prediction_tc(all_dm[run_ind],
                                hrf_surf[vertex],
                                avg_betas[run_ind][vertex],
                                psc_tc = psc_tc,
                                combinedmatrix = combinedmatrix[run_ind],
                                meanvol = results_glmsingle['typed']['meanvol'][vertex]) for run_ind in np.arange(np.array(data_psc).shape[0])]

        ## actually plot, for all runs
        time_sec = np.linspace(0, np.array(data_psc).shape[1] * self.somaModelObj.MRIObj.TR, 
                                num = np.array(data_psc).shape[1]) 

        data_color = '#035900'

        for run_ind in np.arange(np.array(data_psc).shape[0]):

            prediction = predicted_tc_psc[run_ind] 
            voxel = data_psc[run_ind][..., vertex] 

            r2 = np.nan_to_num(1 - (np.nansum((voxel - prediction)**2, axis=0)/ np.nansum((voxel**2), axis=0)))


            fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)

            # plot data with model
            axis.plot(time_sec, prediction, 
                        c = data_color, lw = 3, label = 'R$^2$ = %.2f'%r2, zorder = 1)
            axis.scatter(time_sec, voxel, 
                            marker = 'v', s = 15, c = data_color)
            axis.set_xlabel('Time (s)',fontsize = 20, labelpad = 20)
            axis.set_ylabel('BOLD signal change (%)', fontsize = 20, labelpad = 10)
            axis.tick_params(axis='both', labelsize=18)
            axis.set_xlim(0, np.array(data_psc).shape[1] * self.somaModelObj.MRIObj.TR)

            handles,labels = axis.axes.get_legend_handles_labels()
            axis.legend(handles,labels,loc='upper left',fontsize=15)   

            fig.savefig(op.join(fig_pth,'vertex-%i_runIndex-%i_timeseries.png'%(vertex, run_ind)), dpi=100,bbox_inches = 'tight')


    def plot_betas_roi_hexabins(self, betas_arr, roi_coords, roi_vertices = None,
                                    regs2plot = [], hemi_labels = ['LH', 'RH'], fig_name = '', vmin=-2, vmax=2):

        """
        plot ROI hexabins with beta values per regressor, 
        for a sanity check of how they look
        """
        
        fig_pth = op.join(self.outputdir, 'betas_ROI_hexabins')
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # indices for regressors for interest
        reg_ind = [np.where((self.somaModelObj.soma_cond_unique == name))[0][0] for name in regs2plot]

        # make figure
        # row is hemisphere, columns regressor
        fig, axs = plt.subplots(len(hemi_labels), len(reg_ind), sharey=True, figsize=(18,8))

        for hi, hemi in enumerate(hemi_labels):
            
            # get betas for vertices of that ROI and hemisphere
            if roi_vertices is not None:
                reg_betas = betas_arr[..., reg_ind][roi_vertices[hemi]]
            else:
                if isinstance(betas_arr,dict):
                    reg_betas = betas_arr[hemi][..., reg_ind]
                else:
                    reg_betas = betas_arr[..., reg_ind]

            for ind, reg_name in enumerate(regs2plot):
                
                if len(hemi_labels) > 1:
                    sc = axs[hi][ind].hexbin(roi_coords[hemi][0], 
                                            roi_coords[hemi][1], 
                                            C = reg_betas[...,ind], 
                                            reduce_C_function = np.mean,
                                            cmap = 'RdBu_r',
                                            gridsize = (50,50), vmin=vmin, vmax=vmax)
                    #axs[hi][ind].set_xlim((-20,20))
                    axs[hi][ind].axis('equal')
                    axs[hi][ind].set_title(reg_name)
                else:
                    sc = axs[ind].hexbin(roi_coords[hemi][0], 
                                            roi_coords[hemi][1], 
                                            C = reg_betas[...,ind], 
                                            reduce_C_function = np.mean,
                                            cmap = 'RdBu_r',
                                            gridsize = (50,50), vmin=vmin, vmax=vmax)
                    #axs[ind].set_xlim((-20,20))
                    axs[ind].axis('equal')
                    axs[ind].set_title(reg_name)

        fig.colorbar(sc, ax=axs[:, len(reg_ind)-1]) if len(hemi_labels) > 1 else fig.colorbar(sc)

        fig.savefig(op.join(fig_pth, fig_name), dpi=100,bbox_inches = 'tight')


    def plot_betas_roi_binned(self, betas_arr, roi_coords, roi_vertices = None, n_bins = 100, weight_arr = [],
                                    regs2plot = [], hemi_labels = ['LH', 'RH'], fig_name = '', vmin=-40, vmax=40):

        """
        plot weighted sum of betas for an ROI
        over binned y axis
        """
        
        fig_pth = op.join(self.outputdir, 'betas_ROI_ybins')
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # indices for regressors for interest
        reg_ind = [np.where((self.somaModelObj.soma_cond_unique == name))[0][0] for name in regs2plot]

        # make figure
        # row is hemisphere, columns regressor
        fig, axs = plt.subplots(1, len(hemi_labels), sharey=True, figsize=(18,8))

        for hi, hemi in enumerate(hemi_labels):

            # make y bins
            ybins = np.linspace(np.min(roi_coords[hemi][1]), 
                                np.max(roi_coords[hemi][1]), n_bins+1, endpoint=False)
            
            # get betas for vertices of that ROI and hemisphere
            if isinstance(betas_arr,dict):
                reg_betas = betas_arr[hemi][..., reg_ind]
            else:
                if roi_vertices is not None:
                    reg_betas = betas_arr[..., reg_ind][roi_vertices[hemi]]
                else:
                    reg_betas = betas_arr[..., reg_ind]

            # for each regressor, get summed betas for each bin
            betas_binned_arr = []
            for ind, reg_name in enumerate(regs2plot):

                betas_reg_bin = []
                for b_ind in np.arange(len(ybins)):
                
                    if b_ind == len(ybins) - 1:
                        vert_bin = np.where(((roi_coords[hemi][1] >= ybins[b_ind:][0])))[0]
                    else:
                        vert_bin = np.where(((roi_coords[hemi][1] >= ybins[b_ind:b_ind+2][0]) & \
                                            (roi_coords[hemi][1] < ybins[b_ind:b_ind+2][1])))[0]
                        
                    betas_reg_bin.append(reg_betas[vert_bin][...,ind].dot(weight_arr[roi_vertices[hemi]][vert_bin]/100))
                    
                betas_binned_arr.append(np.flip(betas_reg_bin)) # to then show with right orientation
                
            sc = axs[hi].imshow(np.vstack(betas_binned_arr).T, 
                            aspect='auto', cmap = 'RdBu_r', vmin = vmin, vmax = vmax)

            axs[hi].set_xticks(range(len(regs2plot)))
            axs[hi].set_xticklabels(regs2plot)
            axs[hi].set_title(hemi)

        fig.colorbar(sc)

        fig.savefig(op.join(fig_pth, fig_name), dpi=100,bbox_inches = 'tight')


    def plot_glmsingle_roi_betas(self, participant_list, all_rois = {'M1': ['4'], 'S1': ['3b']}):

        """
        Plot betas for an ROI, for one participant or group

        """

        ## load atlas ROI df
        self.somaModelObj.get_atlas_roi_df(return_RGBA = False)

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.somaModelObj.get_fs_coords(pysub = self.somaModelObj.MRIObj.params['processing']['space'], 
                                                                        merge = True)

        ## load betas per participant
        avg_betas_all = []
        r2_all = []

        for pp in participant_list:

            ## load estimates and fitting options
            results_glmsingle, fit_opts = self.somaModelObj.load_results_dict(pp, load_opts = True)

            # get run design matrix
            # (assumes all runs the same)
            run_dm = self.somaModelObj.get_dm_glmsing(nr_TRs = self.somaModelObj.MRIObj.params['soma']['n_TR'], 
                                                        nr_runs = 1)[0]

            ## get average beta per condition
            avg_betas_runs = self.somaModelObj.average_betas_per_cond(results_glmsingle['typed']['betasmd'])

            ## average across runs
            avg_betas = np.mean(avg_betas_runs, axis = 0)

            ## append for pp in list
            avg_betas_all.append(avg_betas[np.newaxis,...])

            # append r2
            r2_all.append(results_glmsingle['typed']['R2'][np.newaxis,...])

        avg_betas_all = np.vstack(avg_betas_all) if len(participant_list) > 1 else avg_betas_all[0]
        r2_all = np.mean(np.vstack(r2_all), axis = 0)

        ## make images for all ROIs
        for roi2plot in all_rois.keys():

            # for this ROI, 
            # get vertices for each hemisphere
            roi_vertices = {}
            roi_coord_transformed = {}

            for hemi in ['LH', 'RH']:
                
                roi_vertices[hemi] = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                                    roi_list = all_rois[roi2plot],
                                                                    hemi = hemi)
                ## get FS coordinates for each ROI vertex
                roi_coord_transformed[hemi] = self.somaModelObj.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                        y_coord_surf[roi_vertices[hemi]])), 
                                                                                        fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                                        roi_name = roi2plot+'_'+hemi)

            if len(participant_list) > 1:
                # if looking at group data, z-score and average per hemi
                avg_z_betas_dict = {}
                for hemi in ['LH', 'RH']:
                    group_z_betas = [self.somaModelObj.zscore_reg_betas(avg_betas_all[s][roi_vertices[hemi]])[np.newaxis,...] for s in np.arange(len(participant_list))]
                    group_z_betas = np.vstack(group_z_betas)

                    avg_z_betas_dict[hemi] = np.nanmean(group_z_betas, axis = 0)

                ## plot betas in hexabin
                # for each region and both hemispheres
                for region in ['face', 'right_hand', 'left_hand', 'both_hand']:
                    
                    self.plot_betas_roi_hexabins(avg_z_betas_dict, roi_coord_transformed, 
                                                regs2plot = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region],
                                                hemi_labels = ['LH', 'RH'], vmin = -2, vmax = 2,
                                                fig_name = 'sub-group_ROI-{r}_regs-{rg}_glmsinglebetas_zscore.png'.format(r = roi2plot,
                                                                                                                        rg = region))

                    self.plot_betas_roi_binned(avg_z_betas_dict, roi_coord_transformed, roi_vertices = roi_vertices,
                                                n_bins = 100, weight_arr = r2_all,
                                                regs2plot = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region],
                                                hemi_labels = ['LH', 'RH'], vmin = -20, vmax = 20,
                                                fig_name = 'sub-group_ROI-{r}_regs-{rg}_binned_glmsinglebetas_zscore.png'.format(r = roi2plot,
                                                                                                                                rg = region))
            else:
                ## plot betas in hexabin
                # for each region and both hemispheres
                for region in ['face', 'right_hand', 'left_hand', 'both_hand']:
                    
                    self.plot_betas_roi_hexabins(avg_betas, roi_coord_transformed, roi_vertices = roi_vertices,
                                                regs2plot = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region],
                                                hemi_labels = ['LH', 'RH'], vmin = -2, vmax = 2,
                                                fig_name = 'sub-{s}_ROI-{r}_regs-{rg}_glmsinglebetas.png'.format(s = pp,
                                                                                                                r = roi2plot,
                                                                                                                rg = region))

    
    def plot_flatmap(self, data_arr, verts, vmin = 0, vmax = 80, cmap='hot', fig_abs_name = None):

        """
        plot flatmap of data (1D)
        only show select vertices
        """

        surface_arr = np.zeros(data_arr.shape[0])
        surface_arr[:] = np.nan
        surface_arr[verts] = data_arr[verts]

        flatmap = cortex.Vertex(surface_arr, 
                        self.pysub,
                        vmin = vmin, vmax = vmax, 
                        cmap = cmap)
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        # if we provide absolute name for figure, then save there
        if fig_abs_name is not None:

            fig_pth = op.split(fig_abs_name)[0]
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            print('saving %s' %fig_abs_name)
            _ = cortex.quickflat.make_png(fig_abs_name, flatmap, recache=True,
                                        with_colorbar=False,with_curvature=True,with_sulci=True)


    def plot_COM_maps(self, participant, region = 'face', n_bins = 256):

        """
        plot COM maps from GLM betas
        """

        fig_pth = op.join(self.outputdir, 'glm_COM_maps',
                                            'sub-{sj}'.format(sj = participant))
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # load GLM estimates, and get betas and prediction
        soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 
                                        'sub-{sj}'.format(sj = participant), 
                                        'mean_run', 'estimates_run-mean.npy'), allow_pickle=True).item()
        r2 = soma_estimates['r2']

        # normalize the distribution, for better visualization
        region_mask_alpha = normalize(np.clip(r2,.2,.6)) 

        # call COM function
        self.somaModelObj.make_COM_maps(participant, region = region)

        ## load COM values and plot
        if region == 'face':

            com_betas_dir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), 'COM_reg-face.npy')

            COM_region = np.load(com_betas_dir, allow_pickle = True)
            
            # create costume colormp J4
            n_bins = 256
            col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = ['navy','forestgreen','darkorange','purple'],
                                                                bins = n_bins, cmap_name = 'costum_face'))[-1])[0]
            print('created costum colormap %s'%col2D_name)
            
            
            # vertex for face vs all others
            flatmap = cortex.Vertex2D(COM_region, 
                                    region_mask_alpha,
                                    subject = self.pysub,
                                    vmin=0, vmax=3,
                                    vmin2 = 0, vmax2 = 1,
                                    cmap = col2D_name)

            cortex.quickshow(flatmap,with_curvature=True,with_sulci=True,with_colorbar=True, 
                                curvature_brightness = 0.4, curvature_contrast = 0.1)
            
            filename = op.join(fig_pth, 'COM_flatmap_region-face.png')
            print('saving %s' %filename)
            _ = cortex.quickflat.make_png(filename, flatmap, recache=False,with_colorbar=True,
                                                with_curvature=True,with_sulci=True,with_labels=False,
                                                curvature_brightness = 0.4, curvature_contrast = 0.1)
                
        else:
            for side in ['L', 'R']:
                
                com_betas_dir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), 'COM_reg-upper_limb_{s}.npy'.format(s=side))

                COM_region = np.load(com_betas_dir, allow_pickle = True)
                
                # create costume colormp J4
                n_bins = 256
                col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = 'rainbow_r',
                                                                    bins = n_bins, 
                                                                    cmap_name = 'rainbow_r'))[-1])[0]
                print('created costum colormap %s'%col2D_name)


                # vertex for face vs all others
                flatmap = cortex.Vertex2D(COM_region, 
                                        region_mask_alpha,
                                        subject = self.pysub,
                                        vmin=0, vmax=4,
                                        vmin2 = 0, vmax2 = 1,
                                        cmap = col2D_name)

                cortex.quickshow(flatmap,with_curvature=True,with_sulci=True,with_colorbar=True, 
                                curvature_brightness = 0.4, curvature_contrast = 0.1)
            
                filename = op.join(fig_pth, 'COM_flatmap_region-upper_limb_{s}hand.png'.format(s=side))
                print('saving %s' %filename)
                _ = cortex.quickflat.make_png(filename, flatmap, recache=False,with_colorbar=True,
                                                    with_curvature=True,with_sulci=True,with_labels=False,
                                                    curvature_brightness = 0.4, curvature_contrast = 0.1)




