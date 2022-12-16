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
