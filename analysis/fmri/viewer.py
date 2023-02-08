import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import yaml

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

from visuomotor_utils import normalize, add_alpha2colormap, make_raw_vertex_image, make_colormap, COM

import cortex
import click_viewer

import scipy

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

    
    def open_click_viewer(self, participant, custom_dm = True, model2plot = 'glm', data_RFmodel = None,
                                            fit_type = 'mean_run', keep_b_evs = False, fixed_effects = True):

        """
        Opens viewer with flatmap, timeseries and beta estimates
        of GLM model fit or soma RF fitting
        """

        # get list with gii files
        gii_filenames = self.somaModelObj.get_soma_file_list(participant, 
                                    file_ext = self.somaModelObj.MRIObj.params['fitting']['soma']['extension'])

        # load data of all runs
        all_data = self.somaModelObj.load_data4fitting(gii_filenames) # [runs, vertex, TR]

        # average runs
        data2fit = np.nanmean(all_data, axis = 0)

        # if we want to used loo betas, and fixed effects t-stat
        if (fit_type == 'loo_run') and (fixed_effects == True): 
            # get all run lists
            run_loo_list = self.somaModelObj.get_run_list(gii_filenames)

            ## get average beta values 
            _, r2 = self.somaModelObj.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)

            ## COM map dir
            com_betas_dir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)

        else:
            # load GLM estimates, and get betas and prediction
            soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                            allow_pickle=True).item()
            r2 = soma_estimates['r2']

            ## COM map dir
            com_betas_dir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), fit_type)

        ## Load click viewer plotted object
        click_plotter = click_viewer.visualize_on_click(self.somaModelObj.MRIObj, 
                                                        SomaModelObj = self.somaModelObj,
                                                        SomaRF_ModelObj = data_RFmodel,
                                                        soma_data = data2fit,
                                                        pysub = self.pysub)

        ## set figure, and also load estimates and models
        click_plotter.set_figure(participant, custom_dm = custom_dm, keep_b_evs = keep_b_evs,
                                            task2viz = 'soma', soma_run_type = fit_type,
                                            somamodel_name = model2plot)

        # normalize the distribution, for better visualization
        region_mask_alpha = normalize(np.clip(r2,0,.6)) 
        
        # if model is GLM, load COM maps 
        if model2plot == 'glm':

            ########## load face plots ##########
            
            COM_face = np.load(op.join(com_betas_dir, 'COM_reg-face.npy'), allow_pickle = True)

            # create costume colormp J4
            n_bins = 256
            col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = ['navy','forestgreen','darkorange','purple'],
                                                                bins = n_bins, cmap_name = 'costum_face'))[-1])[0]

            click_plotter.images['face'] = cortex.Vertex2D(COM_face, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=3,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = col2D_name)

            ########## load right hand plots ##########
            
            COM_RH = np.load(op.join(com_betas_dir, 'COM_reg-upper_limb_R.npy'), allow_pickle = True)

            col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = 'rainbow_r',
                                                                    bins = n_bins, 
                                                                    cmap_name = 'rainbow_r'))[-1])[0]

            click_plotter.images['RH'] = cortex.Vertex2D(COM_RH, 
                                                        region_mask_alpha,
                                                        subject = self.pysub,
                                                        vmin=0, vmax=4,
                                                        vmin2 = 0, vmax2 = 1,
                                                        cmap = col2D_name)

            ########## load left hand plots ##########
            
            COM_LH = np.load(op.join(com_betas_dir, 'COM_reg-upper_limb_L.npy'), allow_pickle = True)

            click_plotter.images['LH'] = cortex.Vertex2D(COM_LH, 
                                                        region_mask_alpha,
                                                        subject = self.pysub,
                                                        vmin=0, vmax=4,
                                                        vmin2 = 0, vmax2 = 1,
                                                        cmap = col2D_name)

            if keep_b_evs:
                ########## load both hand plots ##########
                
                COM_BH = np.load(op.join(com_betas_dir, 'COM_reg-upper_limb_B.npy'), allow_pickle = True)

                click_plotter.images['BH'] = cortex.Vertex2D(COM_BH, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=4,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = col2D_name)

        elif model2plot == 'somaRF':
            
            ########## load face plots ##########

            face_mask = np.load(op.join(com_betas_dir, 'zmask_reg-face.npy'), allow_pickle=True)
            face_center = np.array(click_plotter.RF_estimates['face']['mu'])#.copy()
            face_center[np.isnan(face_mask)] = np.nan

            region_mask_alpha = np.array(click_plotter.RF_estimates['face']['r2']) # use RF model R2 as mask

            # create costume colormp J4
            n_bins = 256
            col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = ['navy','forestgreen','darkorange','purple'],
                                                                bins = n_bins, cmap_name = 'costum_face'))[-1])[0]

            click_plotter.images['face'] = cortex.Vertex2D(face_center, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=3,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = col2D_name)

            # face size plots
            face_size = np.array(click_plotter.RF_estimates['face']['size'])#.copy()
            face_size[np.isnan(face_mask)] = np.nan

            click_plotter.images['face_size'] = cortex.Vertex2D(face_size, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=3,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = 'hot_alpha')

            ########## load right hand plots ##########

            RH_mask = np.load(op.join(com_betas_dir, 'zmask_reg-upper_limb_R.npy'), allow_pickle=True)
            RH_center = np.array(click_plotter.RF_estimates['RH']['mu'])#.copy()
            RH_center[np.isnan(RH_mask)] = np.nan

            region_mask_alpha = np.array(click_plotter.RF_estimates['RH']['r2']) # use RF model R2 as mask

            col2D_name = op.splitext(op.split(add_alpha2colormap(colormap = 'rainbow_r',
                                                                    bins = n_bins, 
                                                                    cmap_name = 'rainbow_r'))[-1])[0]

            click_plotter.images['RH'] = cortex.Vertex2D(RH_center, 
                                                        region_mask_alpha,
                                                        subject = self.pysub,
                                                        vmin=0, vmax=4,
                                                        vmin2 = 0, vmax2 = 1,
                                                        cmap = col2D_name)

            # RH size plots
            RH_size = np.array(click_plotter.RF_estimates['RH']['size'])#.copy()
            RH_size[np.isnan(RH_mask)] = np.nan

            click_plotter.images['RH_size'] = cortex.Vertex2D(RH_size, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=4,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = 'hot_alpha')

            ########## load left hand plots ##########

            LH_mask = np.load(op.join(com_betas_dir, 'zmask_reg-upper_limb_L.npy'), allow_pickle=True)
            LH_center = np.array(click_plotter.RF_estimates['LH']['mu'])#.copy()
            LH_center[np.isnan(LH_mask)] = np.nan

            region_mask_alpha = np.array(click_plotter.RF_estimates['LH']['r2']) # use RF model R2 as mask

            click_plotter.images['LH'] = cortex.Vertex2D(LH_center, 
                                                        region_mask_alpha,
                                                        subject = self.pysub,
                                                        vmin=0, vmax=4,
                                                        vmin2 = 0, vmax2 = 1,
                                                        cmap = col2D_name)

            # LH size plots
            LH_size = np.array(click_plotter.RF_estimates['LH']['size'])#.copy()
            LH_size[np.isnan(LH_mask)] = np.nan

            click_plotter.images['LH_size'] = cortex.Vertex2D(LH_size, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=4,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = 'hot_alpha')

            if keep_b_evs:
                
                ########## load both hand plots ##########
                
                BH_mask = np.load(op.join(com_betas_dir, 'zmask_reg-upper_limb_B.npy'), allow_pickle=True)
                BH_center = np.array(click_plotter.RF_estimates['BH']['mu'])#.copy()
                BH_center[np.isnan(BH_mask)] = np.nan

                region_mask_alpha = np.array(click_plotter.RF_estimates['BH']['r2']) # use RF model R2 as mask

                click_plotter.images['BH'] = cortex.Vertex2D(BH_center, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=4,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = col2D_name)

                # BH size plots
                BH_size = np.array(click_plotter.RF_estimates['BH']['size'])#.copy()
                BH_size[np.isnan(BH_mask)] = np.nan

                click_plotter.images['BH_size'] = cortex.Vertex2D(BH_size, 
                                                                region_mask_alpha,
                                                                subject = self.pysub,
                                                                vmin=0, vmax=4,
                                                                vmin2 = 0, vmax2 = 1,
                                                                cmap = 'hot_alpha')

        ## soma rsq
        click_plotter.images['Soma_rsq'] = cortex.Vertex(r2, 
                                                        'fsaverage',
                                                        vmin = 0, vmax = 1, 
                                                        cmap = 'Greens')

        cortex.quickshow(click_plotter.images['Soma_rsq'], fig = click_plotter.flatmap_ax,
                                with_rois = False, with_curvature = True, with_colorbar=False, 
                                with_sulci = True, with_labels = False)

        click_plotter.full_fig.canvas.mpl_connect('button_press_event', click_plotter.onclick)
        click_plotter.full_fig.canvas.mpl_connect('key_press_event', click_plotter.onkey)

        plt.show()

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
            _ = cortex.quickflat.make_png(fig_abs_name, flatmap, recache=False,with_colorbar=True,
                                                with_curvature=True,with_sulci=True,with_labels=False,
                                                curvature_brightness = 0.4, curvature_contrast = 0.1)

    def plot_rsq(self, participant_list, fit_type = 'loo_run', 
                                all_rois = {'M1': ['4'], 'S1': ['3b'], #['3a', '3b', '1', '2'],
                                            'S2': ['OP1'],'SMA': ['6mp', '6ma', 'SCEF'],
                                            'sPMC': ['6d', '6a'],'iPMC': ['6v', '6r']}):
                                            
        """
        plot flatmap of data (1D)
        with R2 estimates for subject (or group)
        will also plot flatmaps for rois of interest
        """

        ## load atlas ROI df
        self.somaModelObj.get_atlas_roi_df(return_RGBA = False)

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.somaModelObj.get_fs_coords(pysub = self.somaModelObj.MRIObj.params['processing']['space'], 
                                                                        merge = True)

        # loop over participant list
        r2_all = []
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get list with gii files
                gii_filenames = self.somaModelObj.get_soma_file_list(pp, 
                                                    file_ext = self.somaModelObj.MRIObj.params['fitting']['soma']['extension'])
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(gii_filenames)

                ## get average beta values (all used in GLM)
                _, r2 = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)
            else:
                # load GLM estimates, and get betas and prediction
                soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 
                                                'sub-{sj}'.format(sj = pp), 
                                                fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                                allow_pickle=True).item()
                r2 = soma_estimates['r2']

            # append r2
            r2_all.append(r2[np.newaxis,...])
        r2_all = np.nanmean(np.vstack(r2_all), axis = 0)

        ## plot flatmap whole suface

        if len(participant_list) == 1: # if one participant
            fig_name = op.join(self.outputdir, 'glm_r2',
                                                'sub-{sj}'.format(sj = participant_list[0]), 
                                                fit_type, 'r2_flatmap.png')
        else:
            fig_name = op.join(self.outputdir, 'glm_r2',
                                                'group_mean_r2_flatmap_{l}.png'.format(l = fit_type))

        self.plot_flatmap(r2_all, np.arange(r2_all.shape[0]), 
                                vmin = 0, vmax = .6, cmap='hot', fig_abs_name = fig_name)

        ## plot flatmap for each region
        for region in all_rois.keys():
            
            # get roi vertices for BH
            roi_vertices_BH = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                    roi_list = all_rois[region],
                                                    hemi = 'BH')
            
            self.plot_flatmap(r2_all, roi_vertices_BH, 
                                    vmin = 0, vmax = .6, cmap='hot', 
                                    fig_abs_name = fig_name.replace('.png', '_{r}.png'.format(r=region))) 

    def plot_betas_over_y(self, participant_list, fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 150,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'],
                                            'S2': ['OP1'],'SMA': ['6mp', '6ma', 'SCEF'],
                                            'sPMC': ['6d', '6a'],'iPMC': ['6v', '6r']}):
                                            
        """
        plot imshow
        showing beta distribution over y axis,
        for each regressor of interest
        and for selected ROIs
        """
        # hemisphere labels
        hemi_labels = ['LH', 'RH']

        ## load atlas ROI df
        self.somaModelObj.get_atlas_roi_df(return_RGBA = False)

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.somaModelObj.get_fs_coords(pysub = self.somaModelObj.MRIObj.params['processing']['space'], 
                                                                        merge = True)

        # loop over participant list
        betas = []
        r2 = []
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get list with gii files
                gii_filenames = self.somaModelObj.get_soma_file_list(pp, 
                                                    file_ext = self.somaModelObj.MRIObj.params['fitting']['soma']['extension'])
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(gii_filenames)

                ## get average beta values (all used in GLM)
                betas_pp, r2_pp = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)
            else:
                # load GLM estimates, and get betas and prediction
                soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 
                                                'sub-{sj}'.format(sj = pp), 
                                                fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                                allow_pickle=True).item()
                r2_pp = soma_estimates['r2']

            # append r2
            r2.append(r2_pp[np.newaxis,...])
            betas.append(betas_pp[np.newaxis,...])
        r2 = np.nanmean(np.vstack(r2), axis = 0)
        betas = np.nanmean(np.vstack(betas), axis = 0)

        if len(participant_list) == 1: # if one participant
            fig_name = op.join(self.outputdir, 'betas_vs_coord',
                                                'sub-{sj}'.format(sj = participant_list[0]), 
                                                fit_type, 'betas_binned_all_regressors.png')
        else:
            fig_name = op.join(self.outputdir, 'betas_vs_coord',
                                                'group_mean_betas_binned_all_regressors_{l}.png'.format(l = fit_type))

        # if output path doesn't exist, create it
        os.makedirs(op.split(fig_name)[0], exist_ok = True)

        # make average event file for pp, based on events file
        events_avg = self.somaModelObj.get_avg_events(pp, keep_b_evs = keep_b_evs)

        design_matrix = self.somaModelObj.make_custom_dm(events_avg, 
                                        osf = 100, data_len_TR = nr_TRs, 
                                        TR = self.somaModelObj.MRIObj.TR, 
                                        hrf_params = self.somaModelObj.MRIObj.params['fitting']['soma']['hrf_params'], 
                                        hrf_onset = self.somaModelObj.MRIObj.params['fitting']['soma']['hrf_onset'])

        ## set beta values and reg names in dict
        ## for all relevant regions
        region_regs_dict = {}
        region_betas_dict = {}

        # also get list of all individual regressor names of interest (might not actually use it, but good to have)
        reg_list = []
        for region in all_regions:

            region_regs_dict[region] = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region]
            region_betas_dict[region] = [betas[..., np.where((design_matrix.columns == reg))[0][0]] for reg in region_regs_dict[region]]
            region_betas_dict[region] = np.vstack(region_betas_dict[region])

            reg_list += region_regs_dict[region]

        # make array of weights to use in bin
        weight_arr = r2.copy()
        weight_arr[weight_arr<=0] = 0 # to not use negative weights

        # for each roi, make plot
        for roi2plot in roi2plot_list:

            # get vertices for each hemisphere
            roi_vertices = {}
            roi_coords = {}

            for hemi in ['LH', 'RH']:
                
                roi_vertices[hemi] = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                    roi_list = all_rois[roi2plot],
                                                    hemi = hemi)

                ## get FS coordinates for each ROI vertex
                roi_coords[hemi] = self.somaModelObj.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                        y_coord_surf[roi_vertices[hemi]])), 
                                                                    fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                    roi_name = roi2plot+'_'+hemi)

            # make figure
            # row is hemisphere, columns regressor
            fig, axs = plt.subplots(1, len(hemi_labels), sharey=True, figsize=(18,8))

            for hi, hemi in enumerate(hemi_labels):

                # make y bins
                ybins = np.linspace(np.min(roi_coords[hemi][1]), 
                                    np.max(roi_coords[hemi][1]), n_bins+1, endpoint=False)
                
                # for each regressor, get median beta val for each bin
                betas_binned_arr = []
                for region in all_regions:
                    for ind in np.arange(region_betas_dict[region].shape[0]):

                        betas_reg_bin = []
                        for b_ind in np.arange(len(ybins)):

                            if b_ind == len(ybins) - 1:
                                vert_bin = np.where(((roi_coords[hemi][1] >= ybins[b_ind:][0])))[0]
                            else:
                                vert_bin = np.where(((roi_coords[hemi][1] >= ybins[b_ind:b_ind+2][0]) & \
                                                    (roi_coords[hemi][1] < ybins[b_ind:b_ind+2][1])))[0]

                            if sum(weight_arr[roi_vertices[hemi]][vert_bin]) == 0:
                                betas_reg_bin.append(0)
                            else:
                                betas_reg_bin.append(np.average(region_betas_dict[region][ind][roi_vertices[hemi]][vert_bin], 
                                                            weights = weight_arr[roi_vertices[hemi]][vert_bin]))

                        betas_binned_arr.append(np.flip(betas_reg_bin)) # to then show with right orientation
                
                sc = axs[hi].imshow(np.vstack(betas_binned_arr).T, #interpolation = 'spline36',
                                        extent=[-.5,len(reg_list)-.5,
                                                np.min(roi_coords[hemi][1]), np.max(roi_coords[hemi][1])],
                                        aspect='auto', cmap = 'RdBu_r', vmin = -1, vmax = 1) #vmin = -1.7, vmax = 1.7)
                axs[hi].set_xticks(range(len(reg_list)))
                axs[hi].set_xticklabels(reg_list, rotation=90, fontsize=15)
                axs[hi].set_ylabel('y coordinates (a.u.)', fontsize=20, labelpad=10)
                axs[hi].set_title('Left Hemisphere', fontsize=20) if hemi == 'LH' else axs[hi].set_title('Right Hemisphere', fontsize=20)
            fig.colorbar(sc)

            fig.savefig(fig_name.replace('.png', '_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

    def plot_COM_over_y(self, participant, fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 15,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                'BA43': ['43', 'OP4'], 'S2': ['OP1'],
            'Insula': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3','MI', 'AVI', 'AAIC']
            }):
                                            
        """
        plot scatter plot and binned average of COM values over y axis,
        for each regressor of interest
        and for selected ROIs
        """
        # hemisphere labels
        hemi_labels = ['LH', 'RH']
        # z-score threshold for region localizer 
        z_threshold = 3.1

        # make costum colormap for face
        colormap = ['navy','forestgreen','darkorange','purple']

        cvals  = np.arange(len(colormap))
        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colormap))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        ## load atlas ROI df
        self.somaModelObj.get_atlas_roi_df(return_RGBA = False)

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.somaModelObj.get_fs_coords(pysub = self.somaModelObj.MRIObj.params['processing']['space'], 
                                                                        merge = True)
            
        ## LOAD R2
        if fit_type == 'loo_run':
            # get list with gii files
            gii_filenames = self.somaModelObj.get_soma_file_list(participant, 
                                                file_ext = self.somaModelObj.MRIObj.params['fitting']['soma']['extension'])
            # get all run lists
            run_loo_list = self.somaModelObj.get_run_list(gii_filenames)

            ## get average beta values (all used in GLM)
            betas, r2 = self.somaModelObj.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)

            # path where Region contrasts were stored
            stats_dir = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_stats', 
                                                    'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)

            ## get gross region localizer areas (from contrasts)
            region_mask = {}
            region_mask['upper_limb'] = np.load(op.join(stats_dir, 'fixed_effects_T_upper_limb_contrast.npy'), 
                                                allow_pickle=True)
            region_mask['upper_limb'][region_mask['upper_limb'] < z_threshold] = np.nan
            region_mask['face'] = np.load(op.join(stats_dir, 'fixed_effects_T_face_contrast.npy'), 
                                                allow_pickle=True)
            region_mask['face'][region_mask['face'] < z_threshold] = np.nan
            
            ## get positive and relevant r2
            r2_mask = np.zeros(r2.shape)
            r2_mask[r2 >=0] = 1
        else:
            # load GLM estimates, and get betas and prediction
            soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 
                                            'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                            allow_pickle=True).item()
            r2 = soma_estimates['r2']

        # set fig name
        fig_name = op.join(self.outputdir, 'COM_vs_coord',
                                            'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'COM.png')

        # if output path doesn't exist, create it
        os.makedirs(op.split(fig_name)[0], exist_ok = True)

        # make average event file for pp, based on events file
        events_avg = self.somaModelObj.get_avg_events(participant, keep_b_evs = keep_b_evs)

        design_matrix = self.somaModelObj.make_custom_dm(events_avg, 
                                        osf = 100, data_len_TR = nr_TRs, 
                                        TR = self.somaModelObj.MRIObj.TR, 
                                        hrf_params = self.somaModelObj.MRIObj.params['fitting']['soma']['hrf_params'], 
                                        hrf_onset = self.somaModelObj.MRIObj.params['fitting']['soma']['hrf_onset'])

        ## set beta values and reg names in dict
        ## for all relevant regions
        region_regs_dict = {}
        region_betas_dict = {}

        # also get list of all individual regressor names of interest (might not actually use it, but good to have)
        reg_list = []
        for region in all_regions:

            region_regs_dict[region] = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region]
            region_betas_dict[region] = [betas[..., np.where((design_matrix.columns == reg))[0][0]] for reg in region_regs_dict[region]]
            region_betas_dict[region] = np.vstack(region_betas_dict[region])

            reg_list += region_regs_dict[region]
    
        ## use CS as major axis for ROI coordinate rotation
        roi_ref = 'CS'
        ref_theta = {}
        ref_roi_vert = {}
        for hemi in hemi_labels:
            ref_roi_vert[hemi] = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                roi_list = all_rois[roi_ref],
                                                hemi = hemi)
            ref_theta[hemi] = self.somaModelObj.get_rotation_angle(np.vstack((x_coord_surf[ref_roi_vert[hemi]], 
                                                                y_coord_surf[ref_roi_vert[hemi]])))

        # for each roi, make plot
        for roi2plot in roi2plot_list:

            # get vertices for each hemisphere
            roi_vertices = {}
            roi_coords = {}
            for hemi in hemi_labels:
                roi_vertices[hemi] = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                    roi_list = all_rois[roi2plot],
                                                    hemi = hemi)
                ## get FS coordinates for each ROI vertex
                roi_coords[hemi] = self.somaModelObj.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                        y_coord_surf[roi_vertices[hemi]])), 
                                                                    fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                    theta = ref_theta[hemi],
                                                                    roi_name = roi2plot+'_'+hemi)
                ## fixed effects mask * positive CV-r2
                mask_bool = ((~np.isnan(region_mask['upper_limb'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
                
                ### FOR LEFT HEMISPHERE, plot COM for appropriate hand
                if hemi == 'LH' :
                    ##### plot figure - scatter
                    fig, axs = plt.subplots(1, 2, figsize=(12,4))

                    region = 'right_hand'
                    axs[0].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                                roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                            c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool] ,cmap = 'rainbow_r')
                    axs[0].set_ylim(-20,20) #axs[0].set_ylim(-15,20) #axs[0].set_ylim(-20,15)
                    axs[0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[0].set_title('Right Hand', fontsize=20)

                    region = 'both_hand'
                    axs[1].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                                roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                            c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool] ,cmap = 'rainbow_r')
                    axs[1].set_ylim(-20,20) #axs[1].set_ylim(-15,20) #axs[1].set_ylim(-20,15)
                    axs[1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)
                    fig.savefig(fig_name.replace('.png', '_scatter_hands_LH_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

                    ##### plot figure - binned
                    fig, axs = plt.subplots(1, 2, figsize=(12,4))

                    region = 'right_hand'
                    binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                        COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                        statistic=lambda y: np.nanmean(y), 
                                                        bins=np.linspace(-20, 20, n_bins+1, endpoint=False))[:2]
                    axs[0].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = 'rainbow_r')
                    axs[0].set_xlim(-20,20) #axs[0].set_ylim(-15,20) #axs[0].set_ylim(-20,15)
                    axs[0].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[0].set_title('Right Hand', fontsize=20)

                    region = 'both_hand'
                    binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                        COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                        statistic=lambda y: np.nanmean(y), 
                                                        bins=np.linspace(-20, 20, n_bins+1, endpoint=False))[:2]
                    axs[1].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = 'rainbow_r')
                    axs[1].set_xlim(-20,20) #axs[1].set_ylim(-15,20) #axs[1].set_ylim(-20,15)
                    axs[1].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)
                    fig.savefig(fig_name.replace('.png', '_binned_hands_LH_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

                ### FOR RIGHT HEMISPHERE, plot COM for appropriate hand
                else:     
                    ##### plot figure - scatter
                    fig, axs = plt.subplots(1, 2, figsize=(12,4))

                    region = 'left_hand'
                    axs[0].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                                roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                            c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool] ,cmap = 'rainbow_r')
                    axs[0].set_ylim(-20,20) #axs[0].set_ylim(-15,20) #axs[0].set_ylim(-20,15)
                    axs[0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[0].set_title('Left Hand', fontsize=20)

                    region = 'both_hand'
                    axs[1].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                                roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                            c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool] ,cmap = 'rainbow_r')
                    axs[1].set_ylim(-20,20) #axs[1].set_ylim(-15,20) #axs[1].set_ylim(-20,15)
                    axs[1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)
                    fig.savefig(fig_name.replace('.png', '_scatter_hands_RH_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

                    ##### plot figure - binned
                    fig, axs = plt.subplots(1, 2, figsize=(12,4))

                    region = 'left_hand'
                    binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                        COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                        statistic=lambda y: np.nanmean(y), 
                                                        bins=np.linspace(-20, 20, n_bins+1, endpoint=False))[:2]
                    axs[0].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = 'rainbow_r')
                    axs[0].set_xlim(-20,20) #axs[0].set_ylim(-15,20) #axs[0].set_ylim(-20,15)
                    axs[0].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[0].set_title('Left Hand', fontsize=20)

                    region = 'both_hand'
                    binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                        COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                        statistic=lambda y: np.nanmean(y), 
                                                        bins=np.linspace(-20, 20, n_bins+1, endpoint=False))[:2]
                    axs[1].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = 'rainbow_r')
                    axs[1].set_xlim(-20,20) #axs[1].set_ylim(-15,20) #axs[1].set_ylim(-20,15)
                    axs[1].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)
                    fig.savefig(fig_name.replace('.png', '_binned_hands_RH_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

            ### FOR FACE - BOTH HEMISPHERES

            ##### plot figure - scatter
            fig, axs = plt.subplots(1, 2, figsize=(12,4))

            region = 'face'
            hemi = 'LH'
            ## fixed effects mask * positive CV-r2
            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
            axs[0].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                        roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                    c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool] , cmap = cmap)
            axs[0].set_ylim(-50,0) #axs[0].set_ylim(-20,15)
            axs[0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
            axs[0].set_xlabel('Betas COM', fontsize=15, labelpad=10)
            axs[0].set_title('Face Left Hemisphere', fontsize=20)

            hemi = 'RH'
            ## fixed effects mask * positive CV-r2
            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
            axs[1].scatter(COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool],
                        roi_coords[hemi][1][mask_bool], alpha=r2[roi_vertices[hemi]][mask_bool],
                    c = COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], cmap = cmap)
            axs[1].set_ylim(-50,0) #axs[0].set_ylim(-20,15)
            axs[1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
            axs[1].set_xlabel('Betas COM', fontsize=15, labelpad=10)
            axs[1].set_title('Face Right Hemisphere', fontsize=20)
            fig.savefig(fig_name.replace('.png', '_scatter_face_{roi_name}.png'.format(roi_name = roi2plot)), 
                        dpi=100,bbox_inches = 'tight')

            ##### plot figure - binned
            fig, axs = plt.subplots(1, 2, figsize=(12,4))

            hemi = 'LH'
            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
            binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                statistic=lambda y: np.nanmean(y), 
                                                bins=np.linspace(-50, 0, n_bins+1, endpoint=False))[:2]
            axs[0].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = cmap)
            axs[0].set_xlim(-50,0) #axs[0].set_ylim(-15,20) #axs[0].set_ylim(-20,15)
            axs[0].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
            axs[0].set_ylabel('Betas COM', fontsize=15, labelpad=10)
            axs[0].set_title('Face Left Hemisphere', fontsize=20)

            hemi = 'RH'
            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
            binned_x, binned_y = scipy.stats.binned_statistic(roi_coords[hemi][1][mask_bool], 
                                                COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool], 
                                                statistic=lambda y: np.nanmean(y), 
                                                bins=np.linspace(-50, 0, n_bins+1, endpoint=False))[:2]
            axs[1].scatter(binned_y[:-1],binned_x, c=binned_x,cmap = cmap)
            axs[1].set_xlim(-50,0) #axs[1].set_ylim(-15,20) #axs[1].set_ylim(-20,15)
            axs[1].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
            axs[1].set_ylabel('Betas COM', fontsize=15, labelpad=10)
            axs[1].set_title('Face Right Hemisphere', fontsize=20)
            fig.savefig(fig_name.replace('.png', '_binned_face_{roi_name}.png'.format(roi_name = roi2plot)), 
                        dpi=100,bbox_inches = 'tight')

    def plot_COM_maps(self, participant, region = 'face', fit_type = 'mean_run', fixed_effects = True,
                                    n_bins = 256, plot_cuttout = False, custom_dm = True, keep_b_evs = False):

        """
        plot COM maps from GLM betas
        """

        # if we want to used loo betas, and fixed effects t-stat
        if (fit_type == 'loo_run') and (fixed_effects == True): 

            fig_pth = op.join(self.outputdir, 'glm_COM_maps',
                                                'sub-{sj}'.format(sj = participant), 
                                                'fixed_effects', fit_type)
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            # get list with gii files
            gii_filenames = self.somaModelObj.get_soma_file_list(participant, 
                                                file_ext = self.somaModelObj.MRIObj.params['fitting']['soma']['extension'])
            # get all run lists
            run_loo_list = self.somaModelObj.get_run_list(gii_filenames)

            ## get average beta values 
            _, r2 = self.somaModelObj.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)

            # path to COM betas
            com_filepath = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), 
                                                    'fixed_effects', fit_type)

        else:
            fig_pth = op.join(self.outputdir, 'glm_COM_maps',
                                                'sub-{sj}'.format(sj = participant), fit_type)
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            # load GLM estimates, and get betas and prediction
            soma_estimates = np.load(op.join(self.somaModelObj.outputdir, 
                                            'sub-{sj}'.format(sj = participant), 
                                            fit_type, 'estimates_run-{rt}.npy'.format(rt = fit_type.split('_')[0])), 
                                            allow_pickle=True).item()
            r2 = soma_estimates['r2']

            # path to COM betas
            com_filepath = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'glm_COM', 
                                                    'sub-{sj}'.format(sj = participant), 
                                                    fit_type)

        # normalize the distribution, for better visualization
        region_mask_alpha = normalize(np.clip(r2, 0, .6)) 

        # call COM function
        self.somaModelObj.make_COM_maps(participant, region = region, fit_type = fit_type, fixed_effects = fixed_effects,
                                                    custom_dm = custom_dm, keep_b_evs = keep_b_evs)

        sides_list = ['L', 'R'] if keep_b_evs == False else ['L', 'R', 'B']

        ## load atlas ROI df
        self.somaModelObj.get_atlas_roi_df(return_RGBA = False)

        ## load COM values and plot
        if region == 'face':

            com_betas_dir = op.join(com_filepath, 'COM_reg-face.npy')

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

            ## save same plot but for a few glasser ROIs
            for region, region_label in {'M1': ['4'], 'S1': ['3b']}.items():
                
                # get roi vertices for BH
                roi_vertices_BH = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                        roi_list = region_label,
                                                        hemi = 'BH')
                
                region_com_arr = np.zeros(COM_region.shape[0]); region_com_arr[:] = np.nan
                region_com_arr[roi_vertices_BH] = COM_region[roi_vertices_BH]

                # vertex for face vs all others
                flatmap = cortex.Vertex2D(region_com_arr, 
                                        region_mask_alpha,
                                        subject = self.pysub,
                                        vmin=0, vmax=3,
                                        vmin2 = 0, vmax2 = 1,
                                        cmap = col2D_name)
                
                filename = op.join(fig_pth, 'COM_flatmap_region-face_{r}.png'.format(r = region))
                print('saving %s' %filename)
                _ = cortex.quickflat.make_png(filename, flatmap, recache=False,with_colorbar=True,
                                                    with_curvature=True,with_sulci=True,with_labels=False,
                                                    curvature_brightness = 0.4, curvature_contrast = 0.1)
                

            if plot_cuttout:
                # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
                # left hemi
                cutout_name = 'zoom_roi_left'
                _ = cortex.quickflat.make_figure(flatmap,
                                                with_curvature=True, with_sulci=True, with_roi=False,with_colorbar=False,
                                                cutout=cutout_name,height=2048)

                filename = op.join(fig_pth, 'COM_cutout_region-face_LH.png')
                print('saving %s' %filename)
                _ = cortex.quickflat.make_png(filename, flatmap, recache=True,with_colorbar=False,with_labels=False,
                                            cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                                curvature_brightness = 0.4, curvature_contrast = 0.1)
                # right hemi
                cutout_name = 'zoom_roi_right'
                _ = cortex.quickflat.make_figure(flatmap,
                                                with_curvature=True, with_sulci=True, with_roi=False,with_colorbar=False,
                                                cutout=cutout_name,height=2048)

                filename = op.join(fig_pth, 'COM_cutout_region-face_RH.png')
                print('saving %s' %filename)
                _ = cortex.quickflat.make_png(filename, flatmap, recache=True,with_colorbar=False,with_labels=False,
                                            cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                                curvature_brightness = 0.4, curvature_contrast = 0.1)

                
        else:
            for side in sides_list:
                
                com_betas_dir = op.join(com_filepath, 'COM_reg-upper_limb_{s}.npy'.format(s=side))

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

                ## save same plot but for a few glasser ROIs
                for region, region_label in {'M1': ['4'], 'S1': ['3b']}.items():
                    
                    # get roi vertices for BH
                    roi_vertices_BH = self.somaModelObj.get_roi_vert(self.somaModelObj.atlas_df, 
                                                            roi_list = region_label,
                                                            hemi = 'BH')
                    
                    region_com_arr = np.zeros(COM_region.shape[0]); region_com_arr[:] = np.nan
                    region_com_arr[roi_vertices_BH] = COM_region[roi_vertices_BH]

                    # vertex for face vs all others
                    flatmap = cortex.Vertex2D(region_com_arr, 
                                            region_mask_alpha,
                                            subject = self.pysub,
                                            vmin=0, vmax=4,
                                            vmin2 = 0, vmax2 = 1,
                                            cmap = col2D_name)
                    
                    filename = op.join(fig_pth, 'COM_flatmap_region-upper_limb_{s}hand_{r}.png'.format(s=side,r = region))
                    print('saving %s' %filename)
                    _ = cortex.quickflat.make_png(filename, flatmap, recache=False,with_colorbar=True,
                                                        with_curvature=True,with_sulci=True,with_labels=False,
                                                        curvature_brightness = 0.4, curvature_contrast = 0.1)

                if plot_cuttout:
                    # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
                    # left hemi
                    cutout_name = 'zoom_roi_left'
                    _ = cortex.quickflat.make_figure(flatmap,
                                                    with_curvature=True, with_sulci=True, with_roi=False,with_colorbar=False,
                                                    cutout=cutout_name,height=2048)

                    filename = op.join(fig_pth, 'COM_cutout_region-upper_limb_{s}hand_LH.png'.format(s=side))
                    print('saving %s' %filename)
                    _ = cortex.quickflat.make_png(filename, flatmap, recache=True,with_colorbar=False,with_labels=False,
                                                cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                                    curvature_brightness = 0.4, curvature_contrast = 0.1)
                    # right hemi
                    cutout_name = 'zoom_roi_right'
                    _ = cortex.quickflat.make_figure(flatmap,
                                                    with_curvature=True, with_sulci=True, with_roi=False,with_colorbar=False,
                                                    cutout=cutout_name,height=2048)

                    filename = op.join(fig_pth, 'COM_cutout_region-upper_limb_{s}hand_RH.png'.format(s=side))
                    print('saving %s' %filename)
                    _ = cortex.quickflat.make_png(filename, flatmap, recache=True,with_colorbar=False,with_labels=False,
                                                cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                                    curvature_brightness = 0.4, curvature_contrast = 0.1)


class pRFViewer:

    def __init__(self, pRFModelObj, outputdir = None, pysub = 'fsaverage'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        pRFModelObj : soma Model object
            object from one of the classes defined in soma_model
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # set object to use later on
        self.pRFModelObj = pRFModelObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.pRFModelObj.MRIObj.derivatives_pth, 'plots')
        else:
            self.outputdir = outputdir

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

        # pycortex subject to use
        self.pysub = pysub

    def plot_vertex_tc(self, participant, vertex = None, run_type = 'mean_run', fit_now = True,
                            model2fit = 'gauss', chunk_num = None, ROI = None, fit_hrf = False):

        """
        quick function to 
        plot vertex timeseries and model fit 
        """

        fig_pth = op.join(self.outputdir, 'prf_single_vertex', 
                                            'sub-{sj}'.format(sj = participant), run_type)
        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # get participant models, which also will load 
        # DM and mask it according to participants behavior
        pp_prf_models = self.pRFModelObj.set_models(participant_list = self.pRFModelObj.MRIObj.sj_num)

        if fit_now:

            if fit_hrf:
                self.pRFModelObj.fit_hrf = True
            else:
                self.pRFModelObj.fit_hrf = False

            estimates, chunk2fit =  self.pRFModelObj.fit_data(participant, pp_prf_models = pp_prf_models, 
                                                fit_type = run_type, 
                                                chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                                model2fit = model2fit, save_estimates = False,
                                                xtol = 1e-3, ftol = 1e-4, n_jobs = 8)

            if fit_hrf:
                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)]['{name}_model'.format(name = model2fit)].return_prediction(*list(estimates['it_{name}'.format(name = model2fit)][0, :-3]))
            else:
                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)]['{name}_model'.format(name = model2fit)].return_prediction(*list(estimates['it_{name}'.format(name = model2fit)][0, :-1]))

            r2 = estimates['it_{name}'.format(name = model2fit)][0][-1]

        ## actually plot, for all runs
        time_sec = np.linspace(0, chunk2fit.shape[-1] * self.pRFModelObj.MRIObj.TR, 
                                num = chunk2fit.shape[-1]) 

        data_color = '#db3050'

        fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)

        # plot data with model
        axis.plot(time_sec, model_arr[0], 
                    c = data_color, lw = 3, label = 'R$^2$ = %.2f'%r2, zorder = 1)
        axis.scatter(time_sec, chunk2fit[0], 
                        marker = 'v', s = 15, c = data_color)
        axis.set_xlabel('Time (s)',fontsize = 20, labelpad = 20)
        axis.set_ylabel('BOLD signal change (%)', fontsize = 20, labelpad = 10)
        axis.tick_params(axis='both', labelsize=18)
        axis.set_xlim(0, chunk2fit.shape[-1] * self.pRFModelObj.MRIObj.TR)

        handles,labels = axis.axes.get_legend_handles_labels()
        axis.legend(handles,labels,loc='upper left',fontsize=15)   

        fig.savefig(op.join(fig_pth,'vertex-%i_model-%s_timeseries.png'%(vertex, model2fit)), dpi=100,bbox_inches = 'tight')



    def plot_prf_results(self, participant_list = [], 
                                fit_type = 'mean_run', prf_model_name = 'gauss', max_ecc_ext = 5,
                                mask_arr = True, rsq_threshold = .1, iterative = True, figures_pth = None):


        ## stores estimates for all participants in dict, for ease of access
        group_estimates = {}
  
        for pp in participant_list:

            print('Loading iterative estimates')

            ## load estimates
            pp_prf_est_dict, pp_prf_models = self.pRFModelObj.load_pRF_model_estimates(pp, 
                                                                                fit_type = fit_type, 
                                                                                model_name = prf_model_name, 
                                                                                iterative = iterative, 
                                                                                fit_hrf = self.pRFModelObj.fit_hrf)

            ## mask the estimates, if such is the case
            if mask_arr:
                print('masking estimates')

                # get estimate keys
                keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

                # get screen limits
                max_ecc_ext = pp_prf_models['sub-{sj}'.format(sj = pp)]['prf_stim'].screen_size_degrees/2

                group_estimates['sub-{sj}'.format(sj = pp)] = self.pRFModelObj.mask_pRF_model_estimates(pp_prf_est_dict, 
                                                                            ROI = None,
                                                                            estimate_keys = keys,
                                                                            x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            rsq_threshold = rsq_threshold,
                                                                            pysub = self.pysub
                                                                            )

            else:
                group_estimates['sub-{sj}'.format(sj = pp)] = pp_prf_est_dict


        ## Now actually plot results
        # 
        ### RSQ ###
        # self.plot_rsq(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
        #                                     model_name = prf_model_name, figures_pth = figures_pth)


    def open_click_viewer(self, participant, fit_type = 'mean_run', prf_model_name = 'gauss', 
                            max_ecc_ext = 5, mask_arr = True, rsq_threshold = .1):

        """
        quick function to open click viewer, need to re furbish later
        """

        ## load estimates and model for participant
        pp_prf_est_dict, pp_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant, 
                                                                                fit_type = fit_type, 
                                                                                model_name = prf_model_name, 
                                                                                iterative = True, 
                                                                                fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

            # get screen limits
            max_ecc_ext = pp_prf_models['sub-{sj}'.format(sj = participant)]['prf_stim'].screen_size_degrees/2

            estimates_dict = self.pRFModelObj.mask_pRF_model_estimates(pp_prf_est_dict, 
                                                                            ROI = None,
                                                                            estimate_keys = keys,
                                                                            x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            rsq_threshold = rsq_threshold,
                                                                            pysub = self.pysub
                                                                            )
        else:
            estimates_dict = pp_prf_est_dict

        # Load pRF data
        # get list with gii files
        gii_filenames = self.pRFModelObj.get_prf_file_list(participant, 
                                            file_ext = self.pRFModelObj.MRIObj.params['fitting']['prf']['extension'])

        if fit_type == 'mean_run':         
            # load data of all runs
            all_data = self.pRFModelObj.load_data4fitting(gii_filenames) # [runs, vertex, TR]

            # average runs
            data2fit = np.nanmean(all_data, axis = 0)


        ## Load click viewer plotted object
        click_plotter = click_viewer.visualize_on_click(self.pRFModelObj.MRIObj, 
                                        pRFModelObj = self.pRFModelObj,
                                        pRF_data = data2fit,
                                        prf_dm = pp_prf_models['sub-{sj}'.format(sj = participant)]['prf_stim'].design_matrix,
                                        pysub = self.pysub)

        ## set figure, and also load estimates and models
        click_plotter.set_figure(participant, task2viz = 'prf', pp_prf_est_dict = estimates_dict,
                                        pp_prf_models = pp_prf_models,
                                        prf_run_type = fit_type,  pRFmodel_name = prf_model_name)

        ## calculate pa + ecc + size
        nan_mask = np.where((np.isnan(estimates_dict['r2'])) | (estimates_dict['r2'] < rsq_threshold))[0]
        
        complex_location = estimates_dict['x'] + estimates_dict['y'] * 1j # calculate eccentricity values

        polar_angle = np.angle(complex_location)
        polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0))
        polar_angle_norm[nan_mask] = np.nan

        eccentricity = np.abs(complex_location)
        eccentricity[nan_mask] = np.nan
        
        if prf_model_name in ['dn', 'dog']:
            size_fwhmax, fwatmin = self.pRFModelObj.fwhmax_fwatmin(prf_model_name, estimates_dict)
        else: 
            size_fwhmax = self.pRFModelObj.fwhmax_fwatmin(prf_model_name, estimates_dict)

        size_fwhmax[nan_mask] = np.nan

        ## make alpha mask
        alpha_level = normalize(np.clip(estimates_dict['r2'], rsq_threshold, .5)) # normalize 
        alpha_level[nan_mask] = np.nan
        
        ## pRF rsq
        click_plotter.images['pRF_rsq'] = self.get_flatmaps(estimates_dict['r2'], 
                                                            vmin1 = 0, vmax1 = .8,
                                                            cmap = 'Reds')

        ## pRF Eccentricity

        # make costum colormap
        ecc_cmap = make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = 256, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['ecc'] = make_raw_vertex_image(eccentricity, 
                                                            cmap = ecc_cmap, 
                                                            vmin = 0, vmax = 6, 
                                                            data2 = alpha_level, 
                                                            vmin2 = 0, vmax2 = 1, 
                                                            subject = self.pysub, data2D = True)

        ## pRF Size
        click_plotter.images['size_fwhmax'] = make_raw_vertex_image(size_fwhmax, 
                                                    cmap = 'hot', vmin = 0, vmax = 14, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub, data2D = True)

        ## pRF Polar Angle
        # get matplotlib color map from segmented colors
        PA_cmap = make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = 256, 
                                                    cmap_name = 'PA_mackey_costum',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['PA'] = make_raw_vertex_image(polar_angle_norm, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub, data2D = True)

        ## pRF Exponent 
        if prf_model_name == 'css':
            
            ns = estimates_dict['ns']
            ns[nan_mask] = np.nan

            click_plotter.images['ns'] = self.get_flatmaps(ns, 
                                                        vmin1 = 0, vmax1 = 1,
                                                        cmap = 'plasma')

        ## open figure 

        cortex.quickshow(click_plotter.images['pRF_rsq'], fig = click_plotter.flatmap_ax,
                                with_rois = False, with_curvature = True, with_colorbar=False, 
                                with_sulci = True, with_labels = False)

        click_plotter.full_fig.canvas.mpl_connect('button_press_event', click_plotter.onclick)
        click_plotter.full_fig.canvas.mpl_connect('key_press_event', click_plotter.onkey)

        plt.show()


    def get_flatmaps(self, est_arr1, est_arr2 = None, 
                            vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None,
                            cmap = 'BuBkRd'):

        """
        Helper function to set and return flatmap  
        Parameters
        ----------
        est_arr1 : array
            data array
        cmap : str
            string with colormap name
        vmin: int/float
            minimum value
        vmax: int/float 
            maximum value
        subject: str
            overlay subject name to use
        """

        # if two arrays provided, then fig is 2D
        if est_arr2:
            flatmap = cortex.Vertex2D(est_arr1, est_arr2,
                                    self.pysub,
                                    vmin = vmin1, vmax = vmax1,
                                    vmin2 = vmin2, vmax2 = vmax2,
                                    cmap = cmap)
        else:
            flatmap = cortex.Vertex(est_arr1, 
                                    self.pysub,
                                    vmin = vmin1, vmax = vmax1,
                                    cmap = cmap)

        return flatmap
