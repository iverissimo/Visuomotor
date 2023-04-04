import numpy as np
import os
import os.path as op
import pandas as pd
import yaml
import re

import ptitprince as pt # raincloud plots

import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D

import seaborn as sns

import cortex
import click_viewer

import scipy
from statsmodels.stats import weightstats

import nibabel as nib

class Viewer:

    def __init__(self, pysub = 'fsaverage', derivatives_pth = None, MRIObj = None, curr_system = 'local'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs, sulci etc
        derivatives_pth: str
            absolute path to derivatives folder
        MRIObj: MRIData object
            object from one of the classes defined in preproc_mridata
        curr_system: str
            current system we are working in (default local machine)
        """

        # pycortex subject to use
        self.pysub = pysub

        # MRI data object
        self.MRIObj = MRIObj

        # current system, useful for paths
        self.curr_system = curr_system

        # derivatives path
        self.derivatives_pth = derivatives_pth

        # set other relevant paths
        self.atlas_annot_pth = op.join(self.derivatives_pth, self.MRIObj.params['general']['paths'][self.curr_system]['atlas']) # absolute path to atlas annotation files

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

        # hemisphere labels
        self.hemi_labels = ['LH', 'RH']

    def add_data2overlay(self, flatmap = None, name = ''):

        """
        Helper func to add data to overlay.
        Useful for ROI drawing
        
        Parameters
        ----------
        flatmap: pycortex data object
            XD vertex data
        name: str
            name for data layer that will be added
        """

        # ADD ROI TO OVERLAY
        cortex.utils.add_roi(flatmap, name = name, open_inkscape=False)

    def get_atlas_roi_df(self, annot_pth = None, base_str = 'HCP-MMP1.annot', return_RGBA = False):

        """
        Get all atlas ROI labels and vertex indices, for both hemispheres
        and return it as pandas DF

        Assumes Glasser atlas (2016), might generalize to others in the future
        
        Parameters
        ----------
        annot_pth: str
            absolute path to atlas annotation files
        base_str: str
            base name for annotation file 
        return_RGBA: bool
            if we want to return rgbt(a) as used in Glasser atlas figures, for each vertex
        """

        # number of vertices in one hemisphere (for bookeeping) 
        hemi_vert_num = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

        # path to get annotations
        if annot_pth is None:
            annot_pth = self.atlas_annot_pth

        # make empty rgb dict (although we might not use it)
        atlas_rgb_dict = {'R': [], 'G': [], 'B': [], 'A': []}

        # fill atlas dataframe per hemi
        atlas_df = pd.DataFrame({'ROI': [], 'hemi_vertex': [], 'merge_vertex': [], 'hemisphere': []})

        # iterate per hemifield
        for hemi in self.hemi_labels:
            
            # get annotation file for hemisphere
            annotfile = [op.join(annot_pth,x) for x in os.listdir(annot_pth) if base_str in x and hemi.lower() in x][0]
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
                surf_roi_verts = hemi_roi_verts + hemi_vert_num if hemi[0] == 'R' else hemi_roi_verts
                
                atlas_df = pd.concat((atlas_df,
                                    pd.DataFrame({'ROI': np.tile(hemi_roi[-1], len(hemi_roi_verts)), 
                                                'hemi_vertex': hemi_roi_verts, 
                                                'merge_vertex': surf_roi_verts, 
                                                'hemisphere': np.tile(hemi[0], len(hemi_roi_verts))})
                                    
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

    def get_atlas_roi_vert(self, roi_list = [], hemi = 'BH'):

        """
        get vertex indices for an atlas ROI (or several)
        as defined by the list of labels
        for a specific hemisphere (or both)
        
        Parameters
        ----------
        roi_list: list
            list of strings with ROI labels to load
        hemi: str
            which hemisphere (LH, RH or BH - both)
        """

        roi_vert = []

        for roi2plot in roi_list:
            if hemi == 'BH':
                roi_vert += list(self.atlas_df[self.atlas_df['ROI'] == roi2plot].merge_vertex.values.astype(int))
            else:
                roi_vert += list(self.atlas_df[(self.atlas_df['ROI'] == roi2plot) & \
                                (self.atlas_df['hemisphere'] == hemi[0])].merge_vertex.values.astype(int))

        return np.array(roi_vert)
    
    def get_fs_coords(self, merge = True):

        """
        get freesurfer surface mesh coordinates

        Parameters
        ----------
        merge: bool
            if we are merging both hemispheres, and hence getting coordinates for both combined or separate
        """

        ## FreeSurfer surface file format: Contains a brain surface mesh in a binary format
        # Such a mesh is defined by a list of vertices (each vertex is given by its x,y,z coords) 
        # and a list of faces (each face is given by three vertex indices)

        if merge:
            pts, polys = cortex.db.get_surf(self.pysub, 'flat', merge=True)

            return pts[:,0], pts[:,1], pts[:,2] # [vertex, axis] --> x, y, z
        else:
            left, right = cortex.db.get_surf(self.pysub, 'flat', merge=False)

            return {'LH': [left[0][:,0], left[0][:,1], left[0][:,2]],
                    'RH': [right[0][:,0], right[0][:,1], right[0][:,2]]} # [vertex, axis] --> x, y, z

    def get_rotation_angle(self, roi_list = []):

        """
        given a reference ROI, use PCA to find major axis (usually y),
        and get theta angle value, that is used to build rotation matrix
        
        Parameters
        ----------
        roi_list: list
            list of strings with ROI labels to load
        """
        ## get surface x and y coordinates, for each hemisphere
        x_coord_surf, y_coord_surf, _ = self.get_fs_coords(merge = True)

        ref_theta = {}
        for hemi in self.hemi_labels:
            # get vertex indices for selected ROI and hemisphere
            ref_roi_vert = self.get_atlas_roi_vert(roi_list = roi_list, hemi = hemi)

            # x,y coordinate array for ROI [2, vertex]
            orig_coords = np.vstack((x_coord_surf[ref_roi_vert], 
                                      y_coord_surf[ref_roi_vert]))

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

            # calculate theta
            ref_theta[hemi] = np.arctan((x_v1)/(y_v1))  

        return ref_theta
    
    def transform_roi_coords(self, orig_coords, fig_pth = None, roi_name = '', theta = None):

        """
        Use PCA to rotate x,y ROI coordinates along major axis (usually y)
        NOTE - Assumes we are providing ROI from 1 hemisphere only
        
        Parameters
        ----------
        orig_coords: arr
            x,y coordinate array for ROI [2, vertex]
        fig_pth: str
            if provided, will plot some sanity check plots and save in absolute dir
        roi_name: str
            roi name, used when fig_pth is not None
        theta: float
            rotation angle, if None will calculate relative to major component axis
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
        if theta is None:
            theta = np.arctan((x_v1)/(y_v1))  
        else:
            print('using input theta value')
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation_mat * roi_coord_zeromean

        # get transformed coordenates
        x_transformed, y_transformed = transformed_mat.A
        roi_coord_transformed = np.vstack((x_transformed, y_transformed))

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

        if fig_pth is not None:
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)
            fig.savefig(op.join(fig_pth, 'ROI_PCA_%s.png'%roi_name))
        else:
            plt.show()

        return roi_coord_transformed

    def get_flatmaps(self, est_arr1, est_arr2 = None, 
                            vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None,
                            cmap = 'BuBkRd'):

        """
        Helper function to set and return flatmap  

        Parameters
        ----------
        est_arr1 : array
            data array
        est_arr2 : array
            data array
        cmap : str
            string with colormap name
        vmin1: int/float
            minimum value est_arr1
        vmin2: int/float
            minimum value est_arr2
        vmax1: int/float 
            maximum value est_arr1
        vmax2: int/float 
            maximum value est_arr2
        """

        # if two arrays provided, then fig is 2D
        if est_arr2 is not None:
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

    def plot_flatmap(self, est_arr1, est_arr2 = None, verts = None, 
                            vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None, 
                            cmap='hot', fig_abs_name = None, recache = False, with_colorbar = True,
                            with_curvature = True, with_sulci = True, with_labels=False,
                            curvature_brightness = 0.4, curvature_contrast = 0.1, with_rois = True,
                            zoom2ROI = None, hemi_list = ['left', 'right'], figsize=(15,5), dpi=300):

        """
        plot flatmap of data (1D)
        with option to only show select vertices

        Parameters
        ----------
        est_arr1 : array
            data array
        est_arr2 : array
            data array
        verts: array
            list of vertices to select
        cmap : str
            string with colormap name
        vmin1: int/float
            minimum value est_arr1
        vmin2: int/float
            minimum value est_arr2
        vmax1: int/float 
            maximum value est_arr1
        vmax2: int/float 
            maximum value est_arr2
        fig_abs_name: str
            if provided, will save figure with this absolute name
        zoom2ROI: str
            if we want to zoom into an ROI, provide ROI name
        hemi_list: list/arr
            when zooming, which hemisphere to look at (can also be both)
        """

        # subselect vertices, if provided
        if verts is not None:
            surface_arr1 = np.zeros(est_arr1.shape[0])
            surface_arr1[:] = np.nan
            surface_arr1[verts] = est_arr1[verts]
            if est_arr2 is not None:
                surface_arr2 = np.zeros(est_arr2.shape[0])
                surface_arr2[:] = np.nan
                surface_arr2[verts] = est_arr2[verts]
            else:
                surface_arr2 = None
        else:
            surface_arr1 = est_arr1
            surface_arr2 = est_arr2

        if isinstance(cmap, str):
            flatmap = self.get_flatmaps(surface_arr1, est_arr2 = surface_arr2, 
                                vmin1 = vmin1, vmax1 = vmax1, vmin2 = vmin2, vmax2 = vmax2,
                                cmap = cmap)
        else:
            if surface_arr2 is None:
                data2D = False
            else:
                data2D = True
            flatmap = self.make_raw_vertex_image(surface_arr1, 
                                                cmap = cmap, 
                                                vmin = vmin1, vmax = vmax1, 
                                                data2 = surface_arr2, 
                                                vmin2 = vmin2, vmax2 = vmax2, 
                                                subject = self.pysub, data2D = data2D)
        
        # if we provide absolute name for figure, then save there
        if fig_abs_name is not None:

            fig_pth = op.split(fig_abs_name)[0]
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            print('saving %s' %fig_abs_name)
            _ = cortex.quickflat.make_png(fig_abs_name, flatmap, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                                with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                                curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
        else:
            if len(hemi_list)>1 and zoom2ROI is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize, dpi = dpi)
            else:
                fig, ax1 =  plt.subplots(1, figsize = figsize, dpi = dpi)

            cortex.quickshow(flatmap, fig = ax1, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                    with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                    curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
            
            if zoom2ROI is not None:
                # Zoom on just one hemisphere
                self.zoom_to_roi(self.pysub, zoom2ROI, hemi_list[0], ax=ax1)

                if len(hemi_list)>1:
                    cortex.quickshow(flatmap, fig = ax2, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                    with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                    curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
                    # Zoom on just one region
                    self.zoom_to_roi(self.pysub, zoom2ROI, hemi_list[1], ax=ax2)
            
            return flatmap
        
    def zoom_to_roi(self, subject, roi = None, hem = 'left', margin=10.0, ax=None):

        """
        Plot zoomed in view of flatmap, around a given ROI.
        need to give it the flatmap axis as ref, so it know what to do

        Parameters
        ----------
        subject : str
            Name of the pycortex subject
        roi: str
            name of the ROI to zoom into
        hem: str
            left or right hemisphere
        margin: float
            margin around ROI - will add/subtract to axis max and min
        ax: figure axis
            where to plot (needs to be an axis where a flatmap is already plotted)
        """

        roi_verts = cortex.get_roi_verts(subject, roi)[roi]
        roi_map = cortex.Vertex.empty(subject)
        roi_map.data[roi_verts] = 1

        (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                    nudge=True)
        sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
        roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

        xmin, ymin = roi_pts.min(0) - margin
        xmax, ymax = roi_pts.max(0) + margin
        
        ax.axis([xmin, xmax, ymin, ymax])
            
    def plot_RGBflatmap(self, rgb_arr = [], alpha_arr = None, fig_abs_name = None,
                            recache = False, with_colorbar = True,
                            with_curvature = True, with_sulci = True, with_labels=False,
                            curvature_brightness = 0.4, curvature_contrast = 0.1, with_rois = True,
                            figsize=(15,5), dpi=300):

        """
        plot RGB flatmap

        Parameters
        ----------
        rgb_arr : array
            data array [vertex, 3]
        alpha_arr: array
            alpha array
        fig_abs_name: str
            if provided, will save figure with this absolute name
        """
        if alpha_arr is None:
            alpha_arr = np.ones(rgb_arr.shape[0])

        flatmap = cortex.VertexRGB(rgb_arr[:, 0], rgb_arr[:, 1], rgb_arr[:, 2],
                                alpha = alpha_arr, 
                                subject = self.pysub)

        # if we provide absolute name for figure, then save there
        if fig_abs_name is not None:

            fig_pth = op.split(fig_abs_name)[0]
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            print('saving %s' %fig_abs_name)
            _ = cortex.quickflat.make_png(fig_abs_name, flatmap, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                                with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                                curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
        else:
            fig, ax1 =  plt.subplots(1, figsize = figsize, dpi = dpi)

            cortex.quickshow(flatmap, fig = ax1, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                    with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                    curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
            
            return flatmap
            
    def make_raw_vertex_image(self, data1, cmap = 'hot', vmin = 0, vmax = 1, 
                          data2 = [], vmin2 = 0, vmax2 = 1, subject = 'fsaverage', data2D = False):  
    
        """ function to fix web browser bug in pycortex
            allows masking of data with nans
        
        Parameters
        ----------
        data1 : array
            data array
        data2 : array
            alpha array
        cmap : str
            string with colormap name (not the alpha version)
        vmin: int/float
            minimum value
        vmax: int/float 
            maximum value
        vmin2: int/float
            minimum value
        vmax2: int/float 
            maximum value
        subject: str
            overlay subject name to use
        data2D: bool
            if we want to add alpha or not
        
        Outputs
        -------
        vx_fin : VertexRGB
            vertex object to call in webgl
        
        """
        
        # Get curvature
        curv = cortex.db.get_surfinfo(subject, type = 'curvature', recache=False)#,smooth=1)
        # Adjust curvature contrast / color. Alternately, you could work
        # with curv.data, maybe threshold it, and apply a color map.     
        curv.data[curv.data>0] = .1
        curv.data[curv.data<=0] = -.1
        #curv.data = np.sign(curv.data.data) * .25
        
        curv.vmin = -1
        curv.vmax = 1
        curv.cmap = 'gray'
        
        # Create display data 
        vx = cortex.Vertex(data1, subject, cmap = cmap, vmin = vmin, vmax = vmax)
        
        # Pick an arbitrary region to mask out
        # (in your case you could use np.isnan on your data in similar fashion)
        if data2D:
            data2[np.isnan(data2)] = vmin2
            norm2 = colors.Normalize(vmin2, vmax2)  
            alpha = np.clip(norm2(data2), 0, 1)
        else:
            alpha = ~np.isnan(data1) #(data < 0.2) | (data > 0.4)
        alpha = alpha.astype(np.float)
        
        # Map to RGB
        vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
        vx_rgb[:,alpha>0] = vx_rgb[:,alpha>0] * alpha[alpha>0]
        
        curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])
        # do this to avoid artifacts where curvature gets color of 0 valur of colormap
        curv_rgb[:,np.where((vx_rgb > 0))[-1]] = curv_rgb[:,np.where((vx_rgb > 0))[-1]] * (1-alpha)[np.where((vx_rgb > 0))[-1]]

        # Alpha mask
        display_data = curv_rgb + vx_rgb 

        # Create vertex RGB object out of R, G, B channels
        vx_fin = cortex.VertexRGB(*display_data, subject, curvature_brightness = 0.4, curvature_contrast = 0.1)

        return vx_fin

    def make_colormap(self, colormap = 'rainbow_r', bins = 256, add_alpha = True, invert_alpha = False, cmap_name = 'custom',
                      discrete = False, return_cmap = False):

        """ make custom colormap
        can add alpha channel to colormap,
        and save to pycortex filestore
        
        Parameters
        ----------
        colormap : str or List/arr
            if string then has to be a matplolib existent colormap
            if list/array then contains strings with color names, to create linear segmented cmap
        bins : int
            number of bins for colormap
        add_alpha: bool
            if we want to add an alpha channel
        invert_alpha : bool
            if we want to invert direction of alpha channel
            (y can be from 0 to 1 or 1 to 0)
        cmap_name : str
            new cmap filename, final one will have _alpha_#-bins added to it
        discrete : bool
            if we want a discrete colormap or not (then will be continuous)
        return_cmap: bool
            if we want to return the cmap itself or the absolute path to new colormap
        """
        
        if isinstance(colormap, str): # if input is string (so existent colormap)

            # get colormap
            cmap = cm.get_cmap(colormap)

        else: # is list of strings
            cvals  = np.arange(len(colormap))
            norm = plt.Normalize(min(cvals),max(cvals))
            tuples = list(zip(map(norm,cvals), colormap))
            cmap = colors.LinearSegmentedColormap.from_list("", tuples)
            
            if discrete == True: # if we want a discrete colormap from list
                cmap = colors.ListedColormap(colormap)
                bins = int(len(colormap))

        # convert into array
        cmap_array = cmap(range(bins))

        # reshape array for map
        new_map = []
        for i in range(cmap_array.shape[-1]):
            new_map.append(np.tile(cmap_array[...,i],(bins,1)))

        new_map = np.moveaxis(np.array(new_map), 0, -1)
        
        if add_alpha: 
            # make alpha array
            if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
                _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
            else:
                _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))

            # add alpha channel
            new_map[...,-1] = alpha
            cmap_ext = (0,1,0,1)
        else:
            new_map = new_map[:1,...].copy() 
            cmap_ext = (0,100,0,1)
        
        fig = plt.figure(figsize=(1,1))
        ax = fig.add_axes([0,0,1,1])
        # plot 
        plt.imshow(new_map,
        extent = cmap_ext,
        origin = 'lower')
        ax.axis('off')

        if add_alpha: 
            rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)
        else:
            rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', cmap_name+'_bins_%d.png'%bins)
        #misc.imsave(rgb_fn, new_map)
        plt.savefig(rgb_fn, dpi = 200,transparent=True)

        if return_cmap:
            return cmap
        else:
            return rgb_fn 

    def make_2D_colormap(self, rgb_color = '101', bins = 50, scale=[1,1]):
        
        """
        generate 2D basic colormap, from RGB combination,
        and save to pycortex filestore

        Parameters
        ----------
        rgb_color: str
            combination of rgb values (ex: 101 means it will use red and blue)
        bins: int
            number of color bins between min and max value
        scale: arr/list
            int/float with how much to scale each color (ex: 1 == full red)
        
        """
        
        ##generating grid of x bins
        x,y = np.meshgrid(
            np.linspace(0,1*scale[0],bins),
            np.linspace(0,1*scale[1],bins)) 
        
        # define color combination for plot
        if rgb_color=='101': #red blue
            col_grid = np.dstack((x,np.zeros_like(x), y))
            name='RB'
        elif rgb_color=='110': # red green
            col_grid = np.dstack((x, y,np.zeros_like(x)))
            name='RG'
        elif rgb_color=='011': # green blue
            col_grid = np.dstack((np.zeros_like(x),x, y))
            name='GB'
        
        fig = plt.figure(figsize=(1,1))
        ax = fig.add_axes([0,0,1,1])
        # plot 
        plt.imshow(col_grid,
        extent = (0,1,0,1),
        origin = 'lower')
        ax.axis('off')

        rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', 'custom2D_'+name+'_bins_%d.png'%bins)

        plt.savefig(rgb_fn, dpi = 200)
        
        return rgb_fn  

    def get_weighted_mean_bins(self, data_df, x_key = 'ecc', y_key = 'size', 
                           sort_key = 'ecc', weight_key = 'rsq', n_bins = 10):

        """ 
        Get weighted average bins from dataframe, sorted by one of the variables
        Bins will be equally sized, weights cannot be negative or sum to 0

        Parameters
        ----------
        data_df : DataFrame
            pandas dataframe with one observation per row
        x_key : str
            column name with variable to be averaged in bin
        y_key : str
            column name with variable to be averaged in bin
        sort_key : str
            column name with variable to use for sorting dataframe values
        weight_key: str
            column name with variable to use for weights
        n_bins: int
            number of bins to use
        """
        
        # sort values by eccentricity
        data_df = data_df.sort_values(by=[sort_key])

        #divide in equally sized bins
        df_batches = np.array_split(data_df, n_bins)
        print('Bin size is %i'%int(len(data_df)/n_bins))
        
        mean_x = []
        mean_x_std = []
        mean_y = []
        mean_y_std = []
        
        # for each bin calculate rsq-weighted means and errors of binned ecc/gain 
        for j in np.arange(len(df_batches)):
            mean_x.append(weightstats.DescrStatsW(df_batches[j][x_key],
                                                weights = df_batches[j][weight_key]).mean)
            mean_x_std.append(weightstats.DescrStatsW(df_batches[j][x_key],
                                                    weights = df_batches[j][weight_key]).std_mean)

            mean_y.append(weightstats.DescrStatsW(df_batches[j][y_key],
                                                weights = df_batches[j][weight_key]).mean)
            mean_y_std.append(weightstats.DescrStatsW(df_batches[j][y_key],
                                                    weights = df_batches[j][weight_key]).std_mean)

        return mean_x, mean_x_std, mean_y, mean_y_std

    def get_ROI_verts_dict(self, ROIs = None, pysub = None, split_hemi = False):

        """
        Helper function to get hand-drawn ROI vertices
        to be used in plotting

        Parameters
        ----------
        ROIs : list/arr/str
            list with ROI names. 
            if string, then will look for ROI names with that string in it (also works with regex expression)
        pysub : pycortex subject
            pycortex subject where ROIs are drawn
        split_hemi: bool
            split into hemispheres?
        """

        # if pycortex subject not specified
        if pysub is None:
            pysub = self.pysub
        
        # load all ROI vertices
        all_roi_verts = cortex.get_roi_verts(pysub)

        ## subselect
        # if we input list of ROIs
        if isinstance(ROIs,list) or isinstance(ROIs, np.ndarray): 
            roi_verts = {key: all_roi_verts[key] for key in ROIs}
        
        # if we give reference string
        elif isinstance(ROIs, str): 
            rnames = [val for val in all_roi_verts.keys() if len(re.findall(ROIs, val)) > 0] 
            #[val for val in all_roi_verts.keys() if ROIs in val]
            roi_verts = {key: all_roi_verts[key] for key in rnames}
        
        # if we want all ROIs in overlay
        else: 
            roi_verts = all_roi_verts

        if split_hemi:
            ## get mid vertex index (diving hemispheres)
            left_index = cortex.db.get_surfinfo(pysub).left.shape[0] 
            LH_roi_verts = {key: roi_verts[key][roi_verts[key] < left_index] for key in roi_verts.keys()}
            RH_roi_verts = {key: roi_verts[key][roi_verts[key] >= left_index] for key in roi_verts.keys()}
            
            return LH_roi_verts, RH_roi_verts
        else:
            return roi_verts

    def get_percent_vert_atlas(self, roi_verts = {}, atlas_rois_keys = ['4', '3a', '3b', '1']):

        """
        Given a user-defined ROI (or several), calculate the percentage of vertices 
        that fall in a given atlas (Glasser) ROI  

        Parameters
        ----------
        roi_verts: dict
            dictionary with user-defined ROIs. 
            must have info split in hemispheres (ex: roi_verts['LH']['handband_1'])
        atlas_rois_keys: list/arr/dict
            if list, then should have atlas roi names that we are looking into
            if dictionary, then should have names of ROIs and list of glasser atlas labels 
        """

        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        # check if list, turn to dict
        if isinstance(atlas_rois_keys, list) or isinstance(atlas_rois_keys, np.ndarray):
            atlas_rois_keys = {val: [val] for val in atlas_rois_keys}

        # initialize empty DF
        df_percent_glasser = pd.DataFrame({'glasser_roi': [], 'roi': [], 'hemisphere': [], 'percent_vert': []})

        # for each roi of atlas
        for glasser_roi in atlas_rois_keys.keys():
            # for each hemisphere
            for hemi in self.hemi_labels:
                
                # get glasser vertices
                atlas_vert = self.get_atlas_roi_vert(roi_list = atlas_rois_keys[glasser_roi], 
                                                    hemi = hemi)
                
                # calculate percentage of ROI vertices that are in glasser ROI
                for rname in roi_verts[hemi].keys():
                    
                    perc_vert = sum(np.array([True for vert in roi_verts[hemi][rname] if vert in atlas_vert]))/len(roi_verts[hemi][rname])
                    
                    # append
                    df_percent_glasser = pd.concat((df_percent_glasser,
                                                pd.DataFrame({'glasser_roi': [glasser_roi], 
                                                                'roi': [rname], #[int(re.findall('\d{1,2}', rname)[0])], #
                                                                'hemisphere': [hemi], 
                                                                'percent_vert': [perc_vert * 100]})
                                                ), ignore_index = True) 

        return df_percent_glasser
    
    def plot_ratio_vert_in_altas(self, fig_abs_name = None, roi_verts = {}, 
                                 atlas_rois_keys = ['4', '3a', '3b', '1']):

        """
        Plot ratio of user-defined ROI vertices that fall in Glasser rois
        
        Parameters
        ----------
        roi_verts: dict
            dictionary with user-defined ROIs. 
            must have info split in hemispheres (ex: roi_verts['LH']['handband_1'])
        atlas_rois_keys: list/arr/dict
            if list, then should have atlas roi names that we are looking into
            if dictionary, then should have names of ROIs and list of glasser atlas labels 
        fig_abs_name: str
            absolute path where to store figure
        """
        ## get percentage of handband vertices that are in 
        # different glasser atlas ROIs
        df_percent_glasser = self.get_percent_vert_atlas(roi_verts = roi_verts, 
                                                    atlas_rois_keys = atlas_rois_keys)

        ## plot stacked barplot
        fig, axs = plt.subplots(1, 2, figsize=(15,5), sharey=True, sharex=True)

        for ind, hemi in enumerate(self.hemi_labels):
            
            # select df for hemisphere
            hemi_df = df_percent_glasser[(df_percent_glasser['hemisphere'] == hemi)]

            #create new 'sort' column that contains digits from 'product' column
            hemi_df['sort'] = hemi_df['roi'].str.extract('(\d+)', expand=False).astype(int)
            #sort rows based on digits in 'sort' column
            hemi_df = hemi_df.sort_values('sort')
            #drop 'sort' column
            hemi_df = hemi_df.drop('sort', axis=1)

            # pivot the dataframe into the wide form
            dfp = hemi_df.pivot_table(index='roi', 
                            columns='glasser_roi', 
                            values='percent_vert', sort=False)

            # plot stacked bar
            dfp.plot(kind='bar', stacked=True, rot=0, ax=axs[ind], fontsize = 12)
            
            # fig formating
            fig_title = 'Left Hemisphere' if hemi == 'LH' else 'Right Hemisphere'
            axs[ind].set_title(fig_title, fontsize=20, pad=10)
            axs[ind].set_ylim(0,101) 
            axs[ind].set_ylabel('Vertices in Glasser ROI (%)', fontsize=15, labelpad=10)
            axs[ind].set_xlabel('ROI', fontsize=15, labelpad=10)

        # rotate labels
        fig.autofmt_xdate(rotation=90)

        # control aspect ratio
        plt.subplots_adjust(wspace=0.1, hspace=0)

        if fig_abs_name is not None:
            # if output path doesn't exist, create it
            os.makedirs(op.split(fig_abs_name)[0], exist_ok = True)
            fig.savefig(fig_abs_name, bbox_inches='tight')
        else:
            plt.show()

        return df_percent_glasser
    
    def plot_glasser_rois(self, fig_pth = None, plot_all = True, atlas_rois_keys = ['4', '3a', '3b', '1'],
                                list_colors = []):

        """
        plot glasser atlas with specific color scheme for each ROI
        (need to re-furbish)

        Parameters
        ----------
        fig_pth: str
            path to save flatmap
        plot_all: bool
            if we want to plot all Glasser rois or just a subselection
        atlas_rois_keys: list/arr/dict
            atlas_rois_keys: list/arr/dict
            if list, then should have atlas roi names that we are looking into
            if dictionary, then should have names of ROIs and list of glasser atlas labels 
        list_colors: list
            list of hex labels to use in color map, only when we are subselecting regions to plot
        """

        #fig_pth = op.join(self.outputdir, 'glasser_atlas')

        # if output path doesn't exist, create it
        os.makedirs(fig_pth, exist_ok = True)

        # get ROI color map
        atlas_rgb_dict = self.get_atlas_roi_df(return_RGBA = True)

        if plot_all:
            # plot flatmap
            rgb_arr = np.stack((np.array(atlas_rgb_dict[key]) for key in ['R', 'G', 'B']), axis = -1)

            glasser_flatmap = self.plot_RGBflatmap(rgb_arr = rgb_arr, 
                                                alpha_arr = np.array(atlas_rgb_dict['A']), 
                                                fig_abs_name = op.join(fig_pth, 'glasser_flatmap.png'),
                                                recache = False, with_colorbar = True,
                                                with_curvature = True, with_sulci = True, with_labels=False,
                                                curvature_brightness = 0.4, curvature_contrast = 0.1, with_rois = True,
                                                figsize=(15,5), dpi=300)

        else:
            # check if list, turn to dict
            if isinstance(atlas_rois_keys, list) or isinstance(atlas_rois_keys, np.ndarray):
                atlas_rois_keys = {val: [val] for val in atlas_rois_keys}

            # empty surface
            surf2plot = np.zeros(len(np.array(atlas_rgb_dict['A']))); surf2plot[:] = np.nan

            # for each roi of atlas
            for ind, glasser_roi in enumerate(atlas_rois_keys.keys()):
                
                # get glasser vertices
                atlas_vert = self.get_atlas_roi_vert(roi_list = atlas_rois_keys[glasser_roi], 
                                                    hemi = 'BH')

                # fill with value
                surf2plot[atlas_vert] = ind

            if len(list_colors) < len(atlas_rois_keys.keys()):
                print('full color list not provided, using default list from husl palette')
                
                list_colors = list(sns.color_palette("husl", len(atlas_rois_keys.keys())).as_hex()) 

            cmap_glasser = self.make_colormap(colormap = list_colors, discrete = True,
                                  bins = 256, cmap_name = 'glasser_test', return_cmap = True) 
            
            self.plot_flatmap(surf2plot, 
                            vmin1 = -.5, vmax1 = len(atlas_rois_keys.keys())+.5,
                            cmap = cmap_glasser, 
                            with_sulci = True,with_colorbar = False,
                            fig_abs_name = op.join(fig_pth, 'glasser_flatmap_ROI-{r}.png'.format(r = atlas_rois_keys.keys())))
                        
        # # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
        # cutout_name = 'zoom_roi_left'
        # _ = cortex.quickflat.make_figure(glasser,
        #                                 with_curvature=True,
        #                                 with_sulci=True,
        #                                 with_roi=True,
        #                                 with_colorbar=False,
        #                                 cutout=cutout_name,height=2048)
        # filename = op.join(fig_pth, cutout_name+'_glasser_flatmap.png')
        # print('saving %s' %filename)
        # _ = cortex.quickflat.make_png(filename, glasser, recache=True,
        #                                 with_colorbar=False,with_curvature=True,with_sulci=True)

        # # Name of a sub-layer of the 'cutouts' layer in overlays.svg file
        # cutout_name = 'zoom_roi_right'
        # _ = cortex.quickflat.make_figure(glasser,
        #                                 with_curvature=True,
        #                                 with_sulci=True,
        #                                 with_roi=True,
        #                                 with_colorbar=False,
        #                                 cutout=cutout_name,height=2048)
        # filename = op.join(fig_pth, cutout_name+'_glasser_flatmap.png')
        # print('saving %s' %filename)
        # _ = cortex.quickflat.make_png(filename, glasser, recache=True,
        #                                 with_colorbar=False,with_curvature=True,with_sulci=True)

        # # save inflated 3D screenshots 
        # cortex.export.save_3d_views(glasser, 
        #                     base_name = op.join(fig_pth,'3D_glasser'),
        #                     list_angles = ['lateral_pivot', 'medial_pivot', 'left', 'right', 'top', 'bottom',
        #                                'left'],
        #                     list_surfaces = ['inflated', 'inflated', 'inflated', 'inflated','inflated','inflated',
        #                                   'flatmap'],
        #                     viewer_params=dict(labels_visible=[],
        #                                        overlays_visible=['rois','sulci']),
        #                     size=(1024 * 4, 768 * 4), trim=True, sleep=60)

    def get_geodesic_dist2vertex(self, origin_vert):

        """
        Find the distances (in mm) between a vertex and all other vertices 
        on a surface's hemisphere

        Parameters
        ----------
        origin_vert: int
            reference vertex
        """

        # First we need to import the surfaces for this subject
        surfs = [cortex.polyutils.Surface(*d)
                for d in cortex.db.get_surf(self.pysub, "fiducial")]

        ## get mid vertex index (diving hemispheres)
        left_index = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

        ## find which hemisphere vertex belongs to
        hemi = 'LH' if origin_vert < left_index else 'RH'

        # make filler array for hemisphere not used
        other_hemi_dist = np.zeros(left_index); other_hemi_dist[:] = np.nan

        # if left hemisphere
        if hemi == 'LH':
            # find distances for all vertices in hemisphere relative to origin
            dists = surfs[0].geodesic_distance(origin_vert)
            # stack with filler array, to return whole surface array of distances
            surf_dists = np.hstack((dists, other_hemi_dist))
            
        elif hemi == 'RH':
             # find distances for all vertices in hemisphere relative to origin
            dists = surfs[1].geodesic_distance(origin_vert - left_index)
            # stack with filler array, to return whole surface array of distances
            surf_dists = np.hstack((other_hemi_dist,dists))

        return surf_dists



class somaViewer(Viewer):

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

        # need to initialize parent class (Model), indicating output infos
        super().__init__(pysub = pysub, derivatives_pth = somaModelObj.MRIObj.derivatives_pth, MRIObj = somaModelObj.MRIObj)

        # set object to use later on
        self.somaModelObj = somaModelObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.derivatives_pth, 'plots')
        else:
            self.outputdir = outputdir

    
    def plot_handband_COM_scatter(self, participant_list, r_thresh = .1, fit_type = 'loo_run', pysub = None):

        """
        Plot COM values in scatter plot, for all handband ROIs
        Vertically oriented, and sorted from anterior to posterior
        
        Parameters
        ----------
        participant_list: list
            list with participant ID 
        r_thresh: float
            if putting a rsquare threshold on the data being showed
        """

        # if pycortex subject not specified
        if pysub is None:
            pysub = self.pysub

        # set figures path
        figures_pth = op.join(self.outputdir, 'handband_COM', fit_type)

        ## get vertex dictionary for hand band
        LH_roi_verts, RH_roi_verts = self.get_ROI_verts_dict(pysub = pysub, 
                                                            ROIs = "handband_\d{1,2}", split_hemi = True)
        # save in dict for ease of use
        roi_verts = {'LH': LH_roi_verts, 'RH': RH_roi_verts}

        ## get surface x and y coordinates, for each hemisphere
        x_coord_surf, y_coord_surf, _ = self.get_fs_coords(merge = True)
        
        ## ROTATE COORDINATES IN MAIN AXIS
        roi_coord = {'LH': {}, 'RH': {}}

        for hemi in self.hemi_labels:
            for rname in roi_verts[hemi].keys():

                roi_coord[hemi][rname] = self.transform_roi_coords(np.vstack((x_coord_surf[roi_verts[hemi][rname]], 
                                                                            y_coord_surf[roi_verts[hemi][rname]])), 
                                                    fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 
                                                                    'plots', 'PCA_ROI'), 
                                                    roi_name = '{rn}_{h}'.format(rn = rname, h = hemi), 
                                                    theta = None)
        
        ## get handband COM dataframe for participant list
        handband_COM_df = self.get_handband_COM_df(participant_list, 
                                                      roi_coord = roi_coord, 
                                                      roi_verts = roi_verts, 
                                                      fit_type = fit_type, return_surf = False)
        ## get sorted ROI names
        # from 1 to 22 --> anterior to posterior
        roi_names_all = np.array(sorted(handband_COM_df.ROI.unique(), 
                                        key=lambda rn: int(re.findall('\d{1,2}',rn)[0])))

        # for each participant, 
        for pp in participant_list:
            # and hemisphere, plot
            for hemi in self.hemi_labels:

                # depending on hemisphere, select relevant movement
                movements = ['R_hand', 'B_hand'] if hemi == 'LH' else ['L_hand', 'B_hand']

                # set figure
                fig, axs = plt.subplots(2, len(roi_names_all), figsize=(22,8), sharey=True)

                # iterate over rows of subplots
                for row_ind, movement_region in enumerate(movements):
                    
                    # subselect relevant part of full DF
                    df2plot = handband_COM_df[(handband_COM_df['sj'] == pp) & \
                                            (handband_COM_df['r2'] > r_thresh) & \
                                            (handband_COM_df['hemisphere'] == hemi) & \
                                            (handband_COM_df['movement_region'] == movement_region)]

                    # actually plot handband
                    for ind, rname in enumerate(roi_names_all):

                        im = axs[row_ind][ind].scatter(df2plot[df2plot['ROI'] == rname].x_coordinates,
                                                    df2plot[df2plot['ROI'] == rname].y_coordinates,
                                                    c = df2plot[df2plot['ROI'] == rname].COM,
                                                    cmap = 'rainbow_r',
                                                    vmin = 0, vmax = 4)
                        axs[row_ind][ind].set_title(int(re.findall('\d{1,2}',rname)[0]), fontsize=20, pad=10)
                        
                        if ind == 0:
                            if row_ind == 0:
                                row_label = 'Right Hand' if hemi == 'LH' else 'Left Hand'
                            else:
                                row_label = 'Both Hands'
                            
                            axs[row_ind][ind].set_ylabel(row_label+'\n\ny coordinates (a.u.)', fontsize=20, labelpad=10)
                            # change fontsize of yticks
                            axs[row_ind][ind].tick_params(axis='y',labelsize=12)
                            
                        # remove the x ticks
                        axs[row_ind][ind].set_xticks([])        

                fig.subplots_adjust(hspace=0.5, right=.82)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                fig_name = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), 
                                                'sub-{sj}_handband_scatter_hemi-{h}.png'.format(sj = pp, h=hemi)) 
                
                # if output path doesn't exist, create it
                os.makedirs(op.split(fig_name)[0], exist_ok = True)
                fig.savefig(fig_name)

    def makefig_handband_COM_over_y(self, handband_COM_df, 
                                    participant_list = [], hemi = 'LH', movement_region = 'R_hand', roi_ind_list = [8,9,10,11],
                                    r_thresh = .1, df_models = None, model_names = ['linear', 'piecewise'], model_colors = ['grey', 'k']):
        
        """
        Make figure of COM values vs y coordinates
        for select handband-ROIs, hemisphere and hand movement
        across participants

        Parameters
        ----------
        handband_COM_df: DataFrame
            dataframe with COM values for all handband rois
        participant_list: list
            list with participant ID 
        r_thresh: float
            if putting a rsquare threshold on the data being showed
        hemi: str
            hemisphere to focus on
        movement_region: str
            movement of right/left or both hands
        roi_ind_list: list
            list of handband indices to plot
        df_models: DF
            if provided, will also plot model fit on top (ex: linear fit, or piecewise)
        """

        # make custom colormap
        cmap_hands = self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['upper_limb'],
                                  bins = 256, cmap_name = 'custom_hand', return_cmap = True) 

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]

        # start fig
        fig, axs = plt.subplots(len(participant_list), len(roi_ind_list), 
                                figsize=(int(len(roi_ind_list) * 3.75) ,22), sharex=True, sharey=True)

        # loop over participants
        for ind, pp in enumerate(participant_list):
            
            # plot all columns in row (with values for the participant and handband rois in list)
            for col_ind, rname in enumerate(roi_names_list):
                        
                # subselect relevant part of DF
                df2plot = handband_COM_df[(handband_COM_df['ROI'] == rname) & \
                                        (handband_COM_df['hemisphere'] == hemi) & \
                                        (handband_COM_df['sj'] == pp) & \
                                        (handband_COM_df['r2'] > r_thresh) & \
                                        (handband_COM_df['movement_region'] == movement_region)]

                sns.scatterplot(data = df2plot, x = 'y_coordinates', y = 'COM',
                            hue = 'COM', palette = 'rainbow_r', hue_norm = (0,4), ax = axs[ind][col_ind])
                if df2plot.empty == False:
                    axs[ind][col_ind].get_legend().remove()
                axs[ind][col_ind].set_ylim([0, 4])
                
                # if first row, set title
                if pp == participant_list[0]:
                    axs[ind][col_ind].set_title(rname,fontsize=20, pad=10)
                
                elif pp == participant_list[-1]: # if last row, format x ticks and label
                    
                    axs[ind][col_ind].set_xlabel('y coordinates (a.u.)', fontsize = 18, labelpad=10)
                    axs[ind][col_ind].tick_params(axis='x',labelsize=12)

                # if model fit dataframe provided, plot prediction on top of scatter 
                if (df_models is not None) and (df2plot.empty == False):
                    
                    # get model values for pp
                    pp_models_df = df_models[(df_models['ROI'] == rname) & \
                                            (df_models['hemisphere'] == hemi) & \
                                            (df_models['sj'] == pp) & \
                                            (df_models['movement_region'] == movement_region)]
                    
                    new_x = np.linspace(df2plot.y_coordinates.values.min(), df2plot.y_coordinates.values.max(), len(df2plot.y_coordinates))

                    # plot lines for each model
                    for ind_mod, model in enumerate(model_names):
                        # get coeffs and R2
                        coeff = pp_models_df[pp_models_df['model'] == model].coeffs.values[0]
                        r2_model = pp_models_df[pp_models_df['model'] == model].R2.values[0]

                        # get prediction array
                        if model == 'piecewise':
                            model_prediction = self.somaModelObj.piecewise_linear(new_x, *coeff)
                        elif model == 'linear':
                            model_prediction = self.somaModelObj.linear_func(dm = np.vstack((new_x, 
                                                                                    np.ones(new_x.shape))).T, 
                                                                            betas = coeff)
                             
                        # actually plot
                        axs[ind][col_ind].plot(new_x, model_prediction, color = model_colors[ind_mod])
                        axs[ind][col_ind].annotate("R2 {mod} - {r_val:.2f}".format(r_val = r2_model, mod = model), 
                                                   xy=(0, .7 - ind_mod/3), xycoords='data')

            # format y ticks and label
            axs[ind][0].set_ylabel('sub-{sj}\n\nCOM'.format(sj = pp), fontsize = 18, labelpad=10)
            axs[ind][0].tick_params(axis='y',labelsize=12)
            
            # if first row, put legend on the right side (common for all)
            if ind == 0:
                axs[ind][-1].legend(bbox_to_anchor=(1.04, 1), fontsize=15, 
                            handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                            label = ['thumb', 'index', 'middle', 'ring', 'pinky'][l]) for l in range(5)])

        fig.subplots_adjust(hspace=0.1,wspace=0.1, right=.82)

        return fig

    def get_handband_COM_model_fit_df(self, handband_COM_df, 
                                    participant_list = [], hemi = 'LH', movement_region = 'R_hand', roi_ind_list = [8,9,10,11],
                                    r_thresh = .1):
        
        """
        Fit linear vs piecewise model
        for select handband-ROIs, hemisphere and hand movement
        across participants

        Parameters
        ----------
        handband_COM_df: DataFrame
            dataframe with COM values for all handband rois
        participant_list: list
            list with participant ID 
        r_thresh: float
            if putting a rsquare threshold on the data being showed
        hemi: str
            hemisphere to focus on
        movement_region: str
            movement of right/left or both hands
        roi_ind_list: list
            list of handband indices to plot
        """

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]

        ## save relevant values
        df_summary_models = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'movement_region': [],
                                        'model': [], 'AIC': [], 'BIC': [], 'R2': [], 'coeffs': []})

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]
            
        # loop over participants
        for ind, pp in enumerate(participant_list):

            # plot all columns in row (with values for the participant and handband rois in list)
            for col_ind, rname in enumerate(roi_names_list):

                # subselect relevant part of DF
                df2plot = handband_COM_df[(handband_COM_df['ROI'] == rname) & \
                                        (handband_COM_df['hemisphere'] == hemi) & \
                                        (handband_COM_df['sj'] == pp) & \
                                        (handband_COM_df['r2'] > r_thresh) & \
                                        (handband_COM_df['movement_region'] == movement_region)]
                
                # if no data in handband, fill with nans
                if df2plot.empty:
                    coeff_piecewise = [np.nan]
                    r2_piecewise = np.nan
                    aic_piecewise = np.nan
                    bic_piecewise = np.nan
                else:
                    ## fit piecewise function to data
                    coeff_piecewise, _, r2_piecewise = self.somaModelObj.fit_piecewise(x_data = df2plot.y_coordinates.values, 
                                                                        y_data = df2plot.COM.values, 
                                                                        x0 = df2plot.y_coordinates.values[np.argmax(df2plot.COM.values)], 
                                                                        y0 = np.max(df2plot.COM.values), 
                                                                        k1 = 1, k2 = -1,
                                                                        bounds=([-np.inf, -np.inf, 0, -np.inf], 
                                                                                [np.inf, np.inf, np.inf, 0]))
                    
                    # calc AIC and BIC
                    aic_piecewise = self.somaModelObj.calc_AIC(df2plot.COM.values, 
                                                self.somaModelObj.piecewise_linear(df2plot.y_coordinates.values, 
                                                                                        *coeff_piecewise), 
                                                n_params = len(coeff_piecewise))
                    
                    bic_piecewise = self.somaModelObj.calc_BIC(df2plot.COM.values, 
                                                self.somaModelObj.piecewise_linear(df2plot.y_coordinates.values, 
                                                                                        *coeff_piecewise), 
                                                n_params = len(coeff_piecewise))
                
                
                ## store in dataframe
                df_summary_models = pd.concat((df_summary_models,
                                        pd.DataFrame({'sj': [pp], 
                                                        'ROI': [rname], 
                                                        'hemisphere': [hemi], 
                                                        'movement_region': [movement_region],
                                                        'model': ['piecewise'], 
                                                        'AIC': [aic_piecewise], 
                                                        'BIC': [bic_piecewise], 
                                                        'R2': [r2_piecewise], 
                                                        'coeffs': [coeff_piecewise]})), 
                                            ignore_index = True)

                # if no data in handband, fill with nans
                if df2plot.empty:
                    coeff_linear = [np.nan]
                    r2_linear = np.nan
                    aic_linear = np.nan
                    bic_linear = np.nan
                else:
                    ## fit simple linear regression
                    coeff_linear, dm_linear, r2_linear = self.somaModelObj.fit_linear(df2plot.COM.values, 
                                                                                        df2plot.y_coordinates.values, 
                                                                                        add_intercept = True)
                    
                    # calc AIC and BIC
                    aic_linear = self.somaModelObj.calc_AIC(df2plot.COM.values, 
                                                        self.somaModelObj.linear_func(dm = dm_linear, 
                                                                                        betas = coeff_linear), 
                                                        n_params = len(coeff_linear))
                    
                    bic_linear = self.somaModelObj.calc_BIC(df2plot.COM.values, 
                                                        self.somaModelObj.linear_func(dm = dm_linear, 
                                                                                        betas = coeff_linear), 
                                                        n_params = len(coeff_linear))
                    
                ## store in dataframe
                df_summary_models = pd.concat((df_summary_models,
                                        pd.DataFrame({'sj': [pp], 
                                                        'ROI': [rname], 
                                                        'hemisphere': [hemi], 
                                                        'movement_region': [movement_region],
                                                        'model': ['linear'], 
                                                        'AIC': [aic_linear], 
                                                        'BIC': [bic_linear], 
                                                        'R2': [r2_linear], 
                                                        'coeffs': [coeff_linear]})), 
                                            ignore_index = True)
                
        return df_summary_models

    def get_handband_COM_correlation_df(self, handband_COM_df, corr_method = 'spearman', alpha = .05,
                                            hemi = 'LH', movement_region_dict = {'LH': ['R_hand', 'B_hand'], 'RH': ['L_hand', 'B_hand']}):
        
        """
        Correlate single hand with both hand movement COM values
        done per hemisphere

        Parameters
        ----------
        handband_COM_df: DataFrame
            dataframe with COM values for all handband rois
        hemi: str
            hemisphere of interest
        movement_region_dict: dict
            type of movements to correlate per hemisphere
        corr_method: str
            type of correlation (spearman, pearson etc)
        alpha: float
            alpha level for confidence interval
        """

        # subselect for hemisphere
        df_hemi = handband_COM_df[(handband_COM_df['movement_region'].isin(movement_region_dict[hemi])) & \
                                    (handband_COM_df['hemisphere'] == hemi)]

        ## correlate single and both hand movement COM values
        # for this hemisphere
        corr_df = pd.pivot_table(df_hemi, values = 'COM', 
                            index = ['sj', 'ROI', 'hemisphere', 'x_coordinates', 'y_coordinates', 'vertex'],
                            columns = ['movement_region']).groupby(['sj', 'ROI']).corr(method = corr_method).reset_index()

        # remove irrelevant column
        corr_df = corr_df[(corr_df['movement_region'] == movement_region_dict[hemi][-1])].drop(columns=[movement_region_dict[hemi][-1]])
        # and rename correlation column
        corr_df.rename(columns={movement_region_dict[hemi][0]:'rho'}, inplace=True)

        # add number of points used for correlation (to calculate CI later)
        corr_df['n_obs'] = pd.pivot_table(df_hemi, values = 'COM', 
                            index = ['sj', 'ROI', 'hemisphere', 'x_coordinates', 'y_coordinates', 'vertex'],
                            columns = ['movement_region']).groupby(['sj', 'ROI']).size().values
        
        ## z fisher transform
        z1, se, lo, hi = self.somaModelObj.corr2zFischer(corr_df.rho.values, 
                                                        alpha = alpha,
                                                        n_obs = corr_df.n_obs.values)

        ## add to dataframe
        corr_df['zFisher'] = z1
        corr_df['seFisher'] = se
        corr_df['ci_min'] = lo
        corr_df['ci_max'] = hi

        return corr_df
    
    def plot_handband_COM_correlation(self, corr_df, roi_ind_list = [8,9,10,11]):

        """
        plot COM correlations, between single and both hand, for handband, 
        quite crude for now, will generalize later
        """

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]

        # plot correlation values
        fig, axs = plt.subplots(1, len(roi_names_list), figsize=(int(len(roi_names_list) * 3.75) ,5), sharey = True, dpi = 200)

        for ind, rname in enumerate(roi_names_list):
            
            # plot mean rho
            # which is average zFisher then transformed back to rho
            g1 = sns.pointplot(data = pd.DataFrame({'ROI': [rname],
                                            'mean_rho': np.tanh(corr_df[corr_df['ROI'] == rname].mean().zFisher)}),
                        x = 'ROI', y = 'mean_rho', color = 'k', ax = axs[ind], markers = 'D', scale = 2)
            
            # remove axis labels
            g1.set(xlabel=None)
            g1.set(ylabel=None)
            
            # set title
            axs[ind].set_title(int(re.findall('\d{1,2}',rname)[0]),fontsize=55, pad=30)

            # remove x ticks
            axs[ind].set_xticks([])
            
            # plot individual participant rhos and CI
            temp_x = np.linspace(-.2,.2, len(corr_df[corr_df['ROI'] == rname].rho))

            for i_r, r in enumerate(corr_df[corr_df['ROI'] == rname].rho.values):

                axs[ind].plot([temp_x[i_r],temp_x[i_r]], 
                            [corr_df[corr_df['ROI'] == rname].ci_min.values[i_r],
                            corr_df[corr_df['ROI'] == rname].ci_max.values[i_r]], alpha = .5)

                axs[ind].scatter(temp_x[i_r], r, alpha = .5)
                axs[ind].hlines(y=0, xmin=temp_x.min(), xmax=temp_x.max(),
                            linestyle = '--', color = 'grey')

        # format y ticks and label
        axs[0].set_ylabel(r'Spearman $\rho$', fontsize = 55, labelpad = 30)
        axs[0].tick_params(axis='y',labelsize=35)
        axs[0].set_ylim([-1,1])

        return fig


    def get_handband_deltaBIC_df(self, df_summary_models):
        
        """
        Given summary DF (with BIC values for linear and piecewise models)
        make DF with delta BIC --> if positive, then piecewise fits data better than linear

        Parameters
        ----------
        df_summary_models: DataFrame
            dataframe with BIC values for all handband rois
        """

        df_bic_diff = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'movement_region': [], 'delta_BIC': []})

        for pp in df_summary_models.sj.unique():
            for rname in df_summary_models.ROI.unique():
                for hemi in df_summary_models.hemisphere.unique():
                    for movement_region in df_summary_models.movement_region.unique():
                    
                        pp_df = df_summary_models[(df_summary_models['ROI'] == rname) & \
                                            (df_summary_models['hemisphere'] == hemi) & \
                                            (df_summary_models['sj'] == pp) & \
                                            (df_summary_models['movement_region'] == movement_region)]
                        
                        # if is infinite (due to poor model fit) replace with nan
                        if np.isinf(pp_df[pp_df['model'] == 'linear']['BIC'].values) or \
                            np.isinf(pp_df[pp_df['model'] == 'piecewise']['BIC'].values):
                            delta_bic = np.nan
                        else:
                            delta_bic = pp_df[pp_df['model'] == 'linear']['BIC'].values - pp_df[pp_df['model'] == 'piecewise']['BIC'].values

                        # append
                        df_bic_diff = pd.concat((df_bic_diff,
                                                pd.DataFrame({'sj': [pp], 
                                                            'ROI': [rname], 
                                                            'hemisphere': [hemi], 
                                                            'movement_region': [movement_region], 
                                                            'delta_BIC': delta_bic
                                                            })), ignore_index = True)
                        
        ## fill nan with 0s
        df_bic_diff['delta_BIC'] = df_bic_diff['delta_BIC'].fillna(0)

        return df_bic_diff

    def plot_deltaBIC(self, df_bic_diff):

        """
        plot BIC for handband, quite crude for now, will generalize later
        """

        fig, axs = plt.subplots(1, len(df_bic_diff.ROI.unique()), figsize=(int(len(df_bic_diff.ROI.unique()) * 3.75) ,15),
                       sharey = True, dpi = 200)

        for ind, rname in enumerate(df_bic_diff.ROI.unique()):
            
            # plot barplot for roi
            g1 = sns.barplot(x = 'ROI', y = 'delta_BIC', 
                    data = df_bic_diff[df_bic_diff['ROI'] == rname], 
                        estimator = np.mean, ci=68,
                    capsize = .3 ,linewidth = 3, errcolor= 'k',errwidth = 3,
                            ax=axs[ind], color = 'teal')
            
            # remove axis labels
            g1.set(xlabel=None)
            g1.set(ylabel=None)
            
            # calc t test against 0
            p_val_ttest = scipy.stats.ttest_1samp(df_bic_diff[df_bic_diff['ROI'] == rname].delta_BIC, 
                                                popmean=0, alternative='greater')[1]
            #print(p_val_ttest)
            
            # if significant, then add asterisk
            if p_val_ttest < 0.001:
                axs[ind].annotate('***', (.32,.92), xycoords='axes fraction',fontsize=60)
            elif p_val_ttest < 0.01:
                axs[ind].annotate('**', (.38,.92), xycoords='axes fraction',fontsize=60)
            elif p_val_ttest < 0.05:
                axs[ind].annotate('*', (.42,.92), xycoords='axes fraction',fontsize=60)
            
            # set title
            axs[ind].set_title(int(re.findall('\d{1,2}',rname)[0]),fontsize=55, pad=30)
            
            # remove x and y ticks
            axs[ind].set_xticks([])
            #axs[ind]

        # format y ticks and label
        axs[0].set_ylabel(r'$\Delta$ BIC', fontsize = 55, labelpad = 30)
        axs[0].tick_params(axis='y',labelsize=35)

        #fig.subplots_adjust(hspace=0.5, right=.82)
        #fig.autofmt_xdate(rotation=90)

        return fig


    def open_click_viewer(self, participant, custom_dm = True, model2plot = 'glm', data_RFmodel = None,
                                            fit_type = 'mean_run', keep_b_evs = False, fixed_effects = True):

        """
        Opens viewer with flatmap, timeseries and beta estimates
        of GLM model fit or soma RF fitting
        """

        # get list with gii files
        gii_filenames = self.somaModelObj.get_proc_file_list(participant, file_ext = self.somaModelObj.proc_file_ext)

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
            com_betas_dir = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)

        else:
            # load GLM estimates, and get betas and prediction
            r2 = self.somaModelObj.load_GLMestimates(participant, fit_type = fit_type, run_id = None)['r2']

            ## COM map dir
            com_betas_dir = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = participant), fit_type)

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
        region_mask_alpha = self.somaModelObj.normalize(np.clip(r2,0,.6)) 
        
        # if model is GLM, load COM maps 
        if model2plot == 'glm':

            ########## load face plots ##########
            
            COM_face = np.load(op.join(com_betas_dir, 'COM_reg-face.npy'), allow_pickle = True)

            # create custom colormp J4
            n_bins = 256
            col2D_name = op.splitext(op.split(self.make_colormap(colormap = ['navy','forestgreen','darkorange','purple'],
                                                                bins = n_bins, cmap_name = 'custom_face'))[-1])[0]

            click_plotter.images['face'] = cortex.Vertex2D(COM_face, 
                                                            region_mask_alpha,
                                                            subject = self.pysub,
                                                            vmin=0, vmax=3,
                                                            vmin2 = 0, vmax2 = 1,
                                                            cmap = col2D_name)

            ########## load right hand plots ##########
            
            COM_RH = np.load(op.join(com_betas_dir, 'COM_reg-upper_limb_R.npy'), allow_pickle = True)

            col2D_name = op.splitext(op.split(self.make_colormap(colormap = 'rainbow_r',
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

            # create custome colormp J4
            n_bins = 256
            col2D_name = op.splitext(op.split(self.make_colormap(colormap = ['navy','forestgreen','darkorange','purple'],
                                                                bins = n_bins, cmap_name = 'custom_face'))[-1])[0]

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

            col2D_name = op.splitext(op.split(self.make_colormap(colormap = 'rainbow_r',
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
    
    def plot_rsq(self, participant_list, fit_type = 'loo_run', 
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 
                                            'S2': ['OP1'],'SMA': ['6mp', '6ma', 'SCEF'],
                                            'sPMC': ['6d', '6a'],'iPMC': ['6v', '6r']}):
                                            
        """
        plot flatmap of data (1D)
        with R2 estimates for subject (or group)
        will also plot flatmaps for rois of interest

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels
        """

        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        # loop over participant list
        r2_all = []
        cv_r2_all = []
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get average beta values (all used in GLM)
                _, r2 = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list, use_cv_r2 = False)
                _, cv_r2 = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list, use_cv_r2 = True) 
                cv_r2_all.append(cv_r2[np.newaxis, ...])
            else:
                # load GLM estimates, and get betas and prediction
                r2 = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)['r2']

            # append r2
            r2_all.append(r2[np.newaxis,...])
        r2_all = np.nanmean(np.vstack(r2_all), axis = 0)

        # set figures path
        figures_pth = op.join(self.outputdir, 'rsq', 'soma', fit_type)

        ## plot flatmap whole suface
        if len(participant_list) == 1: # if one participant
            fig_name = op.join(figures_pth, 'sub-{sj}'.format(sj = participant_list[0]), 
                                                 'sub-{sj}_task-soma_flatmap_RSQ.png'.format(sj = pp)) 
        else:
            fig_name = op.join(figures_pth,
                                 'sub-group_task-soma_flatmap_RSQ.png')

        ## plot and save fig for whole surface
        self.plot_flatmap(r2_all, vmin1 = 0, vmax1 = 1, cmap='hot', 
                                    fig_abs_name = fig_name)
        
        if fit_type == 'loo_run':
            cv_r2_all = np.nanmean(np.vstack(cv_r2_all), axis = 0)

            ## plot and save fig for whole surface
            self.plot_flatmap(cv_r2_all, vmin1 = 0, vmax1 = .6, cmap='hot', 
                                    fig_abs_name = fig_name.replace('RSQ', 'CV_RSQ'))

        # ## plot flatmap for each region
        # for region in all_rois.keys():
            
        #     # get roi vertices for BH
        #     roi_vertices_BH = self.get_atlas_roi_vert(roi_list = all_rois[region],
        #                                             hemi = 'BH')

        #     self.plot_flatmap(r2_all, vmin1 = 0, vmax1 = .6, cmap='hot', 
        #                             verts = roi_vertices_BH,
        #                             fig_abs_name = fig_name.replace('.png', '_{r}.png'.format(r=region)))
            
    def plot_betas_over_y(self, participant_list, fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 150,
                                custom_dm = True, hrf_model = 'glover',
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'],
                                            'S2': ['OP1'],'SMA': ['6mp', '6ma', 'SCEF'],
                                            'sPMC': ['6d', '6a'],'iPMC': ['6v', '6r']}):
                                            
        """
        plot imshow
        showing beta distribution over y axis,
        for each regressor of interest
        and for selected ROIs

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)
        all_regions: list
            with movement region name 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels  
        n_bins: int
            number of bins for colormap 
        """
        
        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        ## use CS as major axis for ROI coordinate rotation
        ref_theta = self.get_rotation_angle(roi_list = self.somaModelObj.MRIObj.params['plotting']['soma']['reference_roi'])

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.get_fs_coords(merge = True)

        # loop over participant list
        betas = []
        r2 = []
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get average beta values (all used in GLM)
                betas_pp, r2_pp = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)
            else:
                # load GLM estimates, and get betas and prediction
                soma_estimates = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)
                r2_pp = soma_estimates['r2']
                betas_pp = soma_estimates['betas']

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

        ## Get DM
        design_matrix = self.somaModelObj.load_design_matrix(pp, keep_b_evs = keep_b_evs, 
                                                    custom_dm = custom_dm, nTRs = nr_TRs, 
                                                    hrf_model = hrf_model)

        ## set beta values and reg names in dict
        ## for all relevant regions
        region_regs_dict = {}
        region_betas_dict = {}

        reg_list = [] # also store all regressor names
        for region in all_regions:
            region_betas_dict[region] = self.somaModelObj.get_region_betas(betas, region = region, dm = design_matrix)
            region_regs_dict[region] = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region]
            reg_list += region_regs_dict[region]

        ## make array of weights to use in bin
        weight_arr = r2.copy()
        weight_arr[weight_arr<=0] = 0 # to not use negative weights

        ## for each roi, make plot
        for roi2plot in roi2plot_list:

            ## get FS coordinates for each ROI vertex
            roi_vertices = {}
            roi_coords = {}
            for hemi in self.hemi_labels:
                roi_vertices[hemi] = self.get_atlas_roi_vert(roi_list = all_rois[roi2plot],
                                                            hemi = hemi)
                roi_coords[hemi] = self.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                        y_coord_surf[roi_vertices[hemi]])), 
                                                                    theta = ref_theta[hemi],
                                                                    fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                    roi_name = roi2plot+'_'+hemi)

            # make figure
            # row is hemisphere, columns regressor
            fig, axs = plt.subplots(1, len(self.hemi_labels), sharey=True, figsize=(18,8))

            for hi, hemi in enumerate(self.hemi_labels):
                
                # for each regressor, get median beta val for each bin
                betas_binned_arr = []
                for region in all_regions:
                    for ind in np.arange(region_betas_dict[region].shape[0]):

                        betas_reg_bin, _, _, _ = self.get_weighted_mean_bins(pd.DataFrame({'beta': region_betas_dict[region][ind][roi_vertices[hemi]], 
                                                                         'coords': roi_coords[hemi][1], 
                                                                         'r2': weight_arr[roi_vertices[hemi]]}), 
                                                           x_key = 'beta', y_key = 'coords', sort_key = 'coords',
                                                           weight_key = 'r2', n_bins = n_bins)

                        betas_binned_arr.append(betas_reg_bin) 
                
                sc = axs[hi].imshow(np.vstack(betas_binned_arr).T, #interpolation = 'spline36',
                                        extent=[-.5,len(reg_list)-.5,
                                                np.min(roi_coords[hemi][1]), np.max(roi_coords[hemi][1])],
                                        aspect='auto', origin='lower', cmap = 'RdBu_r', vmin = -1, vmax = 1) 
                axs[hi].set_xticks(range(len(reg_list)))
                axs[hi].set_xticklabels(reg_list, rotation=90, fontsize=15)
                axs[hi].set_ylabel('y coordinates (a.u.)', fontsize=20, labelpad=10)
                axs[hi].set_title('Left Hemisphere', fontsize=20) if hemi == 'LH' else axs[hi].set_title('Right Hemisphere', fontsize=20)
            fig.colorbar(sc)

            fig.savefig(fig_name.replace('.png', '_{roi_name}.png'.format(roi_name = roi2plot)), 
                                dpi=100,bbox_inches = 'tight')

    def plot_COM_over_y(self, participant_list, fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 50,
                                z_threshold = 3.1,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                'BA43': ['43', 'OP4'], 'S2': ['OP1'],
            'Insula': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3','MI', 'AVI', 'AAIC']
            }):
                                            
        """
        plot scatter plot and binned average of COM values over y axis,
        for each movement region of interest
        and for selected ROIs

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)
        roi2plot_list: list
            list with ROI names to plot
        all_regions: list
            with movement region name 
        all_rois: dict
            dictionary with names of ROIs and list of glasser atlas labels  
        n_bins: int
            number of y coord bins to divide COM values into
        """

        ## make custom colormap for face and hands
        cmap_face = self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['face'],
                                                                bins = 256, cmap_name = 'custom_face', return_cmap = True)
        cmap_hands = self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['upper_limb'],
                                                                bins = 256, cmap_name = 'custom_hand', return_cmap = True) 

        ## get COM df
        COM_df, COM_df_binned = self.get_COM_coords_df(participant_list, fit_type = fit_type, keep_b_evs = keep_b_evs,
                                                            nr_TRs = nr_TRs, roi2plot_list = roi2plot_list, n_bins = n_bins,
                                                            z_threshold = z_threshold, all_regions = all_regions, all_rois = all_rois)

        for pp in participant_list:

            # set fig name
            fig_name = op.join(self.outputdir, 'COM_vs_coord',
                                                'sub-{sj}'.format(sj = pp), 
                                                fit_type, 'COM.png')
            # if output path doesn't exist, create it
            os.makedirs(op.split(fig_name)[0], exist_ok = True)

            # for each roi, 
            for roi2plot in roi2plot_list:
                # for each hemi, make plot
                for hemi in self.hemi_labels:
                    
                    ########## SCATTER PLOT ALL VALUES #########
                    fig, axs = plt.subplots(1, 3, figsize=(18,4))

                    if hemi == 'LH':
                        # right hand
                        aa = sns.scatterplot(data = COM_df[(COM_df['hemisphere'] == hemi) & \
                                                (COM_df['ROI'] == roi2plot) & \
                                                (COM_df['sj'] == pp) & \
                                                ((COM_df['movement_region'] == 'right_hand'))], 
                                        x = 'COM', y = 'coordinates', hue = 'COM', palette = cmap_hands, ax = axs[0], hue_norm = (0,4))
                        axs[0].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['right_hand'][l]) for l in range(5)])
                        axs[0].set_title('Right Hand', fontsize=20)
                    else:
                        # left hand
                        aa = sns.scatterplot(data = COM_df[(COM_df['hemisphere'] == hemi) & \
                                                (COM_df['ROI'] == roi2plot) & \
                                                (COM_df['sj'] == pp) & \
                                                ((COM_df['movement_region'] == 'left_hand'))], 
                                        x = 'COM', y = 'coordinates', hue = 'COM', palette = cmap_hands, ax = axs[0], hue_norm = (0,4))
                        axs[0].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['left_hand'][l]) for l in range(5)])
                        axs[0].set_title('Left Hand', fontsize=20)
                    
                    axs[0].set_ylim(-20,20) 
                    axs[0].set_xlim(0, 4)
                    axs[0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_xlabel('Betas COM', fontsize=15, labelpad=10)

                    # both hands
                    aa = sns.scatterplot(data = COM_df[(COM_df['hemisphere'] == hemi) & \
                                            (COM_df['ROI'] == roi2plot) & \
                                            (COM_df['sj'] == pp) & \
                                            ((COM_df['movement_region'] == 'both_hand'))], 
                                    x = 'COM', y = 'coordinates', hue = 'COM', palette = cmap_hands, ax = axs[1], hue_norm = (0,4))
                    
                    axs[1].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['both_hand'][l]) for l in range(5)])
                    axs[1].set_ylim(-20,20) 
                    axs[1].set_xlim(0, 4)
                    axs[1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)

                    # face
                    aa = sns.scatterplot(data = COM_df[(COM_df['hemisphere'] == hemi) & \
                                            (COM_df['ROI'] == roi2plot) & \
                                            (COM_df['sj'] == pp) & \
                                            ((COM_df['movement_region'] == 'face'))], 
                                    x = 'COM', y = 'coordinates', hue = 'COM', palette = cmap_face, ax = axs[2], hue_norm = (0,3))
                    
                    axs[2].legend(loc='lower left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_face(int(256/3*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['face'][l]) for l in range(4)])
                    axs[2].set_ylim(-50,0)
                    axs[2].set_xlim(0, 3)
                    axs[2].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[2].set_xlabel('Betas COM', fontsize=15, labelpad=10)
                    axs[2].set_title('Face', fontsize=20)

                    fig.savefig(fig_name.replace('.png', '_scatter_hemisphere-{h}_{roi_name}.png'.format(roi_name = roi2plot, 
                                                                                                        h = hemi)), 
                                dpi=100,bbox_inches = 'tight')

                    ########## SCATTER PLOT BINNED #########
                    fig, axs = plt.subplots(1, 3, figsize=(18,4))

                    if hemi == 'LH':
                        # right hand
                        df2plot = COM_df_binned[(COM_df_binned['hemisphere'] == hemi) & \
                                                (COM_df_binned['ROI'] == roi2plot) & \
                                                (COM_df_binned['sj'] == pp) & \
                                                ((COM_df_binned['movement_region'] == 'right_hand'))]
                        name = 'right_hand'
                    else:
                        # left hand
                        df2plot = COM_df_binned[(COM_df_binned['hemisphere'] == hemi) & \
                                                (COM_df_binned['ROI'] == roi2plot) & \
                                                (COM_df_binned['sj'] == pp) & \
                                                ((COM_df_binned['movement_region'] == 'left_hand'))]
                        name = 'left_hand'
                        
                    aa = sns.scatterplot(data = df2plot, 
                                        y = 'COM', x = 'coordinates', hue = 'COM', palette = cmap_hands, ax = axs[0], hue_norm = (0,4))
                    axs[0].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][name][l]) for l in range(5)])

                    axs[0].errorbar(x = df2plot.coordinates.values, 
                                    y = df2plot.COM.values, 
                                    yerr = df2plot.COM_std.values,
                                    xerr = df2plot.coordinates_std.values,
                                    zorder=0, c='grey', alpha = 0.5)
                    axs[0].set_xlim(-20,20) 
                    axs[0].set_ylim(0, 4)
                    axs[0].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[0].set_title('Right Hand', fontsize=20) if name == 'right_hand' else axs[0].set_title('Left Hand', fontsize=20) 

                    # both hands
                    df2plot = COM_df_binned[(COM_df_binned['hemisphere'] == hemi) & \
                                            (COM_df_binned['ROI'] == roi2plot) & \
                                            (COM_df_binned['sj'] == pp) & \
                                            ((COM_df_binned['movement_region'] == 'both_hand'))]
                    aa = sns.scatterplot(data = df2plot, 
                                    y = 'COM', x = 'coordinates', hue = 'COM', palette = cmap_hands, ax = axs[1],hue_norm = (0,4))
                    
                    axs[1].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['both_hand'][l]) for l in range(5)])
                    axs[1].errorbar(x = df2plot.coordinates.values, 
                                    y = df2plot.COM.values, 
                                    yerr = df2plot.COM_std.values,
                                    xerr = df2plot.coordinates_std.values,
                                    zorder=0, c='grey', alpha = 0.5)
                    axs[1].set_xlim(-20,20) 
                    axs[1].set_ylim(0, 4)
                    axs[1].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[1].set_title('Both Hand', fontsize=20)

                    # face
                    df2plot = COM_df_binned[(COM_df_binned['hemisphere'] == hemi) & \
                                            (COM_df_binned['ROI'] == roi2plot) & \
                                            (COM_df_binned['sj'] == pp) & \
                                            ((COM_df_binned['movement_region'] == 'face'))]
                    aa = sns.scatterplot(data = df2plot, 
                                    y = 'COM', x = 'coordinates', hue = 'COM', palette = cmap_face, ax = axs[2],hue_norm = (0,3))
                    
                    axs[2].legend(loc='lower left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_face(int(256/3*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['face'][l]) for l in range(4)])
                    axs[2].errorbar(x = df2plot.coordinates.values, 
                                    y = df2plot.COM.values, 
                                    yerr = df2plot.COM_std.values,
                                    xerr = df2plot.coordinates_std.values,
                                    zorder=0, c='grey', alpha = 0.5)
                    axs[2].set_xlim(-50,0)
                    axs[2].set_ylim(0, 3)
                    axs[2].set_xlabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[2].set_ylabel('Betas COM', fontsize=15, labelpad=10)
                    axs[2].set_title('Face', fontsize=20)

                    fig.savefig(fig_name.replace('.png', '_binned_hemisphere-{h}_{roi_name}.png'.format(roi_name = roi2plot, 
                                                                                                        h = hemi)), 
                                dpi=100,bbox_inches = 'tight')

    def plot_RF_over_y(self, participant_list, data_RFmodel = None,
                                fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 50,
                                z_threshold = 3.1,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                'BA43': ['43', 'OP4'], 'S2': ['OP1'],
            'Insula': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3','MI', 'AVI', 'AAIC']
            }):
                                            
        """
        plot scatter plot and binned average of RF values over y axis,
        for each movement region of interest
        and for selected ROIs

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)
        roi2plot_list: list
            list with ROI names to plot
        all_regions: list
            with movement region name 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels  
        n_bins: int
            number of y coord bins to divide COM values into
        """

        ## make custom colormap for face and hands
        cmap_face = self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['face'],
                                                                bins = 256, cmap_name = 'custom_face', return_cmap = True)
        cmap_hands = self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['upper_limb'],
                                                                bins = 256, cmap_name = 'custom_hand', return_cmap = True) 

        ## get RF df
        RF_df = self.get_RF_coords_df(participant_list, data_RFmodel = data_RFmodel,
                                                        fit_type = fit_type, keep_b_evs = keep_b_evs,
                                                            roi2plot_list = roi2plot_list, 
                                                            z_threshold = z_threshold, all_regions = all_regions, all_rois = all_rois)

        for pp in participant_list:

            # set fig name
            fig_name = op.join(self.outputdir, 'RF_vs_coord',
                                                'sub-{sj}'.format(sj = pp), 
                                                fit_type, 'RF.png')
            # if output path doesn't exist, create it
            os.makedirs(op.split(fig_name)[0], exist_ok = True)

            # for each roi, 
            for roi2plot in roi2plot_list:
                # for each hemi, make plot
                for hemi in self.hemi_labels:
                    
                    ########## SCATTER PLOT ALL VALUES #########
                    fig, axs = plt.subplots(2, 3, figsize=(18,10))

                    if hemi == 'LH':
                        # right hand
                        aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                                (RF_df['ROI'] == roi2plot) & \
                                                (RF_df['sj'] == pp) & \
                                                (RF_df['RF_r2'] > 0) & \
                                                (RF_df['slope'] > 0) & \
                                                ((RF_df['movement_region'] == 'right_hand'))], 
                                        x = 'center', y = 'coordinates', hue = 'center', palette = cmap_hands, 
                                        ax = axs[0][0], hue_norm = (0,4))
                        axs[0][0].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['right_hand'][l]) for l in range(5)])
                        axs[0][0].set_title('Right Hand', fontsize=20)
                        name = 'right_hand'
                    else:
                        # left hand
                        aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                                (RF_df['ROI'] == roi2plot) & \
                                                (RF_df['sj'] == pp) & \
                                                (RF_df['RF_r2'] > 0) & \
                                                (RF_df['slope'] > 0) & \
                                                ((RF_df['movement_region'] == 'left_hand'))], 
                                        x = 'center', y = 'coordinates', hue = 'center', palette = cmap_hands, 
                                        ax = axs[0][0], hue_norm = (0,4))
                        axs[0][0].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['left_hand'][l]) for l in range(5)])
                        axs[0][0].set_title('Left Hand', fontsize=20)
                        name = 'left_hand'
                    
                    axs[0][0].set_ylim(-20,20) 
                    axs[0][0].set_xlim(-.5, 4.5)
                    axs[0][0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0][0].set_xlabel('RF center', fontsize=15, labelpad=10)

                    # center + size
                    aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                                (RF_df['ROI'] == roi2plot) & \
                                                (RF_df['sj'] == pp) & \
                                                (RF_df['RF_r2'] > 0) & \
                                                (RF_df['slope'] > 0) & \
                                                ((RF_df['movement_region'] == name))], 
                                    x = 'center', y = 'coordinates', hue = 'size', palette = 'magma_r', 
                                    ax = axs[1][0], hue_norm = (0,4))
                    axs[1][0].set_ylim(-20,20) 
                    axs[1][0].set_xlim(-.5, 4.5)
                    axs[1][0].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1][0].set_xlabel('RF center', fontsize=15, labelpad=10)
                    aa.get_legend().remove()
                    sm = plt.cm.ScalarMappable(cmap= 'magma_r', norm=plt.Normalize(-.5, 4.5))
                    sm.set_array([])
                    aa.figure.colorbar(sm)

                    # both hands
                    aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                            (RF_df['ROI'] == roi2plot) & \
                                            (RF_df['sj'] == pp) & \
                                            (RF_df['RF_r2'] > 0) & \
                                            (RF_df['slope'] > 0) & \
                                            ((RF_df['movement_region'] == 'both_hand'))], 
                                    x = 'center', y = 'coordinates', hue = 'center', palette = cmap_hands, 
                                    ax = axs[0][1], hue_norm = (0,4))
                    
                    axs[0][1].legend(loc='upper left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_hands(int(256/4*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['both_hand'][l]) for l in range(5)])
                    axs[0][1].set_ylim(-20,20) 
                    axs[0][1].set_xlim(-.5, 4.5)
                    axs[0][1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0][1].set_xlabel('RF center', fontsize=15, labelpad=10)
                    axs[0][1].set_title('Both Hand', fontsize=20)
                    
                    # center + size
                    aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                                (RF_df['ROI'] == roi2plot) & \
                                                (RF_df['sj'] == pp) & \
                                                (RF_df['RF_r2'] > 0) & \
                                                (RF_df['slope'] > 0) & \
                                                ((RF_df['movement_region'] == 'both_hand'))], 
                                    x = 'center', y = 'coordinates', hue = 'size', palette = 'magma_r', 
                                    ax = axs[1][1], hue_norm = (0,4))
                    axs[1][1].set_ylim(-20,20) 
                    axs[1][1].set_xlim(-.5, 4.5)
                    axs[1][1].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1][1].set_xlabel('RF center', fontsize=15, labelpad=10)
                    aa.get_legend().remove()

                    # face
                    aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                            (RF_df['ROI'] == roi2plot) & \
                                            (RF_df['sj'] == pp) & \
                                            (RF_df['RF_r2'] > 0) & \
                                            (RF_df['slope'] > 0) & \
                                            ((RF_df['movement_region'] == 'face'))], 
                                    x = 'center', y = 'coordinates', hue = 'center', palette = cmap_face, 
                                    ax = axs[0][2], hue_norm = (0,3))
                    
                    axs[0][2].legend(loc='lower left',fontsize=5, 
                                handles = [mpatches.Patch(color = cmap_face(int(256/3*l)), 
                                label = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts']['face'][l]) for l in range(4)])
                    axs[0][2].set_ylim(-50,0)
                    axs[0][2].set_xlim(-.5, 3.5)
                    axs[0][2].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[0][2].set_xlabel('RF center', fontsize=15, labelpad=10)
                    axs[0][2].set_title('Face', fontsize=20)

                    # center + size
                    aa = sns.scatterplot(data = RF_df[(RF_df['hemisphere'] == hemi) & \
                                                (RF_df['ROI'] == roi2plot) & \
                                                (RF_df['sj'] == pp) & \
                                                (RF_df['RF_r2'] > 0) & \
                                                (RF_df['slope'] > 0) & \
                                                ((RF_df['movement_region'] == 'face'))], 
                                    x = 'center', y = 'coordinates', hue = 'size', palette = 'magma_r', 
                                    ax = axs[1][2], hue_norm = (0,4))
                    axs[1][2].set_ylim(-50,0)
                    axs[1][2].set_xlim(-.5, 3.5)
                    axs[1][2].set_ylabel('y coordinates (a.u.)', fontsize=15, labelpad=10)
                    axs[1][2].set_xlabel('RF center', fontsize=15, labelpad=10)
                    aa.get_legend().remove()

                    fig.savefig(fig_name.replace('.png', '_scatter_hemisphere-{h}_{roi_name}.png'.format(roi_name = roi2plot, 
                                                                                                        h = hemi)), 
                                dpi=100,bbox_inches = 'tight')

    def get_handband_COM_df(self, participant_list, roi_coord = {}, roi_verts = {}, fit_type = 'loo_run', return_surf = False):

        """
        For handband, get COM values for each participant, hemisphere and ROI

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        roi_coord: DF/dict
            x-y coordinates for each ROI and hemisphere
        roi_verts: DF/dict
            vertex number for each ROI and hemisphere
        """

        # initialize DF to save estimates
        handband_COM_df = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'x_coordinates': [], 'y_coordinates': [],
                                        'COM': [], 'r2': [], 'movement_region': [], 'vertex': []})

        surf_COM_df = {'L_hand': {'sj': {}}, 'R_hand': {'sj': {}}, 'B_hand': {'sj': {}}} # to save whole surface values

        side_list = ['L', 'R', 'B'] # we want to store COM values for left, right and both hands

        ## load Hand COM
        for pp in participant_list:

            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get average CV-r2 (all used in GLM)
                _, r2_pp = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)

                ## get com_filepath
                com_dir = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = pp), 'fixed_effects', fit_type)

            else:
                # load GLM estimates, and get r2
                r2_pp = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)['r2']

                ## get com_filepath
                com_dir = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = pp), fit_type)
                
            ## load COM
            for side in side_list:
                
                COM_region = np.load(op.join(com_dir, 'COM_reg-upper_limb_{s}.npy'.format(s=side)), 
                                allow_pickle = True)
                surf_COM_df['{s}_hand'.format(s=side)]['sj'][pp] = COM_region
                
                # select values per hemisphere
                for hemi in self.hemi_labels:
                    # and ROI
                    for rname in roi_coord[hemi].keys():
                        
                        # select COM values for hemi and ROI
                        com_roi_hemi = COM_region[roi_verts[hemi][rname]]
                        
                        # mask out regions that are nan
                        mask_bool = (~np.isnan(com_roi_hemi)).astype(bool)
                
                        # save values in df
                        handband_COM_df = pd.concat((handband_COM_df,
                                                    pd.DataFrame({'sj': np.tile(pp, len(com_roi_hemi[mask_bool])), 
                                                                'ROI': np.tile(rname, len(com_roi_hemi[mask_bool])), 
                                                                'hemisphere': np.tile(hemi, len(com_roi_hemi[mask_bool])), 
                                                                'x_coordinates': roi_coord[hemi][rname][0][mask_bool], 
                                                                'y_coordinates': roi_coord[hemi][rname][1][mask_bool],
                                                                'COM': com_roi_hemi[mask_bool], 
                                                                'r2': r2_pp[roi_verts[hemi][rname]][mask_bool], 
                                                                'movement_region': np.tile('{s}_hand'.format(s=side), 
                                                                                            len(com_roi_hemi[mask_bool])),
                                                                'vertex': roi_verts[hemi][rname][mask_bool]
                                                                })
                                                    ), ignore_index = True)
        
        # if we also want whole surface values
        if return_surf:
            return handband_COM_df, surf_COM_df
        else:
            return handband_COM_df
        
    def get_mean_hand_handband_COM_df(self, handband_COM_df):

        """
        For handband, average COM of both hands and single hand movements,
        for each participant, hemisphere and ROI

        Parameters
        ----------
        handband_COM_df: DataFrame
            dataframe with COM values for all handband rois
        """

        df_mean_handband_COM_df = pd.DataFrame({})

        movement_region_dict = {'LH': ['R_hand', 'B_hand'], 'RH': ['L_hand', 'B_hand']}

        # iterate over hemi
        for hemi in self.hemi_labels:

            df_hemi = handband_COM_df[(handband_COM_df['movement_region'].isin(movement_region_dict[hemi])) & \
                                      (handband_COM_df['hemisphere'] == hemi)]
            df_hemi = df_hemi.groupby(['sj', 'ROI', 'hemisphere', 'x_coordinates', 'y_coordinates', 'vertex']).mean().reset_index()
            df_hemi['movement_region'] = 'combined_hand'

            # append
            df_mean_handband_COM_df = pd.concat((df_mean_handband_COM_df,
                                             df_hemi.copy()), ignore_index=True)
        
        return df_mean_handband_COM_df

    def get_COM_piecewise4plotting(self, handband_COM_df, df_summary_models = None,
                                    participant_list = [], hemi = 'LH', movement_region = 'R_hand', roi_ind_list = [8,9,10,11],
                                    r_thresh = .1):
        
        """
        Get piecewise model prediction array, given fitted coefficients,
        for select handband-ROIs, hemisphere and hand movement
        across participants

        Parameters
        ----------
        handband_COM_df: DataFrame
            dataframe with COM values for all handband rois
        participant_list: list
            list with participant ID 
        r_thresh: float
            if putting a rsquare threshold on the data being showed
        hemi: str
            hemisphere to focus on
        movement_region: str
            movement of right/left or both hands
        roi_ind_list: list
            list of handband indices to plot
        df_summary_models: DataFrame
            dataframe with BIC values for all handband rois

        """

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]

        # save prediction arrays in DF
        df_predictions = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [],  'movement_region': [],
                                    'prediction_COM': [], 'prediction_y_coordinates': []})

        # save best fitting participant label, to use later
        bestfit_pp_roi = {}

        for _, rname in enumerate(roi_names_list):
            
            # subselect relevant part of DF
            region_df = handband_COM_df[(handband_COM_df['ROI'] == rname) & \
                                        (handband_COM_df['hemisphere'] == hemi) & \
                                        (handband_COM_df['r2'] > r_thresh) & \
                                        (handband_COM_df['movement_region'] == movement_region)]

            # coordinates for ROI
            prediction_y_coord = np.linspace(region_df.y_coordinates.values.min(), 
                                            region_df.y_coordinates.values.max(), 300)
            
            bestfit_pp_roi[rname] = []
            
            # loop over participants
            for ind, pp in enumerate(participant_list):

                # subselect for participant
                df2plot = region_df[region_df['sj'] == pp]
                
                # get model values for pp
                pp_models_df = df_summary_models[(df_summary_models['ROI'] == rname) & \
                                                (df_summary_models['hemisphere'] == hemi) & \
                                                (df_summary_models['sj'] == pp) & \
                                                (df_summary_models['movement_region'] == movement_region)]

                ## get prediction array
                # if any BIC val is nan, means model fit failed (ex: missing data)
                if np.isnan(pp_models_df.BIC.values).any():
                    prediction_arr = np.zeros(len(prediction_y_coord)); prediction_arr[:] = np.nan
                    
                else:
                    coeff = pp_models_df[pp_models_df['model'] == 'piecewise'].coeffs.values[0]
                    prediction_arr = self.somaModelObj.piecewise_linear(prediction_y_coord, *coeff)
                    
                    # if piecewise was a better fit, store for bookeeping
                    if pp_models_df[pp_models_df['model'] == 'piecewise'].BIC.values < pp_models_df[pp_models_df['model'] == 'linear'].BIC.values:
                        bestfit_pp_roi[rname].append(pp)

                # append in dataframe
                df_predictions = pd.concat((df_predictions,
                                        pd.DataFrame({'sj': np.tile(pp, len(prediction_arr)), 
                                                        'ROI': np.tile(rname, len(prediction_arr)), 
                                                        'hemisphere': np.tile(hemi, len(prediction_arr)),  
                                                        'movement_region': np.tile(movement_region, len(prediction_arr)),
                                                        'prediction_COM': prediction_arr, 
                                                        'prediction_y_coordinates': prediction_y_coord})
                                        ), ignore_index = True)

        return df_predictions, bestfit_pp_roi

    def plot_piecewisefits(self, df_predictions, bestfit_pp_roi = None,
                                        participant_list = [], roi_ind_list = [8,9,10,11]):

        """
        plot model fits for handband, across participants and averaged
        quite crude for now, will generalize later
        """

        # get list of ROIs to plot
        roi_names_list = ['handband_{i}'.format(i = val) for val in roi_ind_list]

        fig, axs = plt.subplots(1, len(roi_names_list), figsize=(int(len(roi_names_list) * 3.75) ,5),
                       sharey = True, dpi = 200)

        for ind, rname in enumerate(roi_names_list):
            
            # check number of participants where model fitted better than linear 
            num_pp_roi = len(bestfit_pp_roi[rname])
            
            if num_pp_roi > len(participant_list)/2: # if more than half the participants, then still plot  
            
                # plot lineplot for roi
                sns.lineplot(data = df_predictions[df_predictions['ROI'] == rname], 
                                x = 'prediction_y_coordinates', y = 'prediction_COM',
                                ci = None, linewidth = 1, alpha = .3, hue = 'sj', ax=axs[ind])
                g1 = sns.lineplot(data = df_predictions[df_predictions['ROI'] == rname], 
                                    x = 'prediction_y_coordinates', y = 'prediction_COM',
                                    ci = None, linewidth = 3, ax=axs[ind], color = 'k')
            
                # remove axis labels
                g1.set(xlabel=None)
                g1.set(ylabel=None)
                axs[ind].get_legend().remove()
            else:
                axs[ind].annotate('n.a.', (.22,.5), xycoords='axes fraction',fontsize=60)
                # remove x 
                axs[ind].set_xticks([])
            
            # set title
            axs[ind].set_title(int(re.findall('\d{1,2}',rname)[0]),fontsize=55, pad=30)

            # limit y
            axs[ind].tick_params(axis='x',labelsize=30)
            
            if ind == 0:
                axs[ind].set_xlabel('y coordinates (mm)', fontsize = 55, labelpad = 30, loc = 'left')
            
        # format y ticks and label
        axs[0].set_ylim([0,4])
        axs[0].set_ylabel('COM', fontsize = 55, labelpad = 30)
        axs[0].tick_params(axis='y',labelsize=30)

        return fig


    def plot_COM_maps(self, participant, region = 'face', fit_type = 'mean_run', fixed_effects = True,
                                    n_bins = 256, custom_dm = True, keep_b_evs = False,
                                    all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                            'S2': ['OP1'],'SMA': ['6mp', '6ma', 'SCEF'],
                                            'sPMC': ['6d', '6a'],'iPMC': ['6v', '6r']}):

        """
        plot COM maps from GLM betas
        throughout surface and for a specific brain region

        Parameters
        ----------
        participant: str
            participant ID 
        region: str
            movement region name 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        fixed_effects: bool
            if we want to use fixed effects across runs  
        custom_dm: bool
            if we are defining DM manually (this is, using specifc HRF), or using nilearn function for DM
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands) 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels  
        n_bins: int
            number of bins for colormap 
        """
        
        ## labels for went using regresors for both sides
        sides_list = ['L', 'R'] if keep_b_evs == False else ['L', 'R', 'B']

        # if we want to used loo betas, and fixed effects t-stat
        if (fit_type == 'loo_run') and (fixed_effects == True): 

            fig_pth = op.join(self.outputdir, 'glm_COM_maps',
                                                'sub-{sj}'.format(sj = participant), 
                                                'fixed_effects', fit_type)
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            # get all run lists
            run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(participant, file_ext = self.somaModelObj.proc_file_ext))

            ## get average beta values 
            _, r2 = self.somaModelObj.average_betas(participant, fit_type = fit_type, 
                                                        weighted_avg = True, runs2load = run_loo_list)

            # path to COM betas
            com_filepath = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = participant), 'fixed_effects', fit_type)

        else:
            fig_pth = op.join(self.outputdir, 'glm_COM_maps',
                                                'sub-{sj}'.format(sj = participant), fit_type)
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            # load GLM estimates, and get betas and prediction
            r2 = self.somaModelObj.load_GLMestimates(participant, fit_type = fit_type, run_id = None)['r2']

            # path to COM betas
            com_filepath = op.join(self.somaModelObj.COM_outputdir, 'sub-{sj}'.format(sj = participant), fit_type)

        ## make alpha mask 
        # normalize the distribution, for better visualization
        region_mask_alpha = self.somaModelObj.normalize(np.clip(r2, 0, .5)) 

        ## call COM function
        self.somaModelObj.make_COM_maps(participant, region = region, fit_type = fit_type, fixed_effects = fixed_effects,
                                                    custom_dm = custom_dm, keep_b_evs = keep_b_evs)

        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        ## load COM values and plot
        if region == 'face':
            
            # load COM
            COM_region = np.load(op.join(com_filepath, 'COM_reg-face.npy'), allow_pickle = True)
            
            # create custome colormp J4
            col2D_name = op.splitext(op.split(self.make_colormap(colormap = self.somaModelObj.MRIObj.params['plotting']['soma']['colormaps']['face'],
                                                                bins = n_bins, cmap_name = 'custom_face'))[-1])[0]
            print('created custom colormap %s'%col2D_name)

            self.plot_flatmap(COM_region, 
                                est_arr2 = region_mask_alpha, 
                                vmin1 = 0, vmax1 = 3, vmin2 = 0, vmax2 = 1,
                                cmap = col2D_name, 
                                fig_abs_name = op.join(fig_pth, 'COM_flatmap_region-face.png'))

            ## save same plot but for a few glasser ROIs
            for region, region_label in all_rois.items():
                
                # get roi vertices for BH
                roi_vertices_BH = self.get_atlas_roi_vert(roi_list = region_label,
                                                        hemi = 'BH')

                self.plot_flatmap(COM_region, est_arr2 = region_mask_alpha, 
                                verts = roi_vertices_BH,
                                vmin1 = 0, vmax1 = 3, vmin2 = 0, vmax2 = 1,
                                cmap = col2D_name, 
                                fig_abs_name = op.join(fig_pth, 
                                                        'COM_flatmap_region-face_{r}.png'.format(r = region)))
                
        else:
            for side in sides_list:
                
                ## load COM
                COM_region = np.load(op.join(com_filepath, 'COM_reg-upper_limb_{s}.npy'.format(s=side)), 
                                        allow_pickle = True)
                
                ## create custom colormap
                col2D_name = op.splitext(op.split(self.make_colormap(colormap = 'rainbow_r',
                                                                    bins = n_bins, 
                                                                    cmap_name = 'rainbow_r'))[-1])[0]
                print('created custom colormap %s'%col2D_name)

                self.plot_flatmap(COM_region, est_arr2 = region_mask_alpha, 
                                vmin1 = 0, vmax1 = 4, vmin2 = 0, vmax2 = 1,
                                cmap = col2D_name, 
                                fig_abs_name = op.join(fig_pth, 
                                                        'COM_flatmap_region-upper_limb_{s}hand.png'.format(s=side)))
               
                ## save same plot but for a few glasser ROIs
                for region, region_label in all_rois.items():
                    
                    # get roi vertices for BH
                    roi_vertices_BH = self.get_atlas_roi_vert(roi_list = region_label,
                                                            hemi = 'BH')

                    self.plot_flatmap(COM_region, est_arr2 = region_mask_alpha, 
                                verts = roi_vertices_BH,
                                vmin1 = 0, vmax1 = 4, vmin2 = 0, vmax2 = 1,
                                cmap = col2D_name, 
                                fig_abs_name = op.join(fig_pth, 
                                'COM_flatmap_region-upper_limb_{s}hand_{r}.png'.format(s=side,r = region)))
                
    def get_COM_coords_df(self, participant_list, fit_type = 'loo_run', keep_b_evs = True,
                                nr_TRs = 141, roi2plot_list = ['M1', 'S1'], n_bins = 40, z_threshold = 3.1,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                'BA43': ['43', 'OP4'], 'S2': ['OP1'],
            'Insula': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3','MI', 'AVI', 'AAIC']
            }):

        """
        Helper function to get COM values over y axis 
        (for all relevant vertices and also binned)
        for each movement region of interest
        and for selected ROIs.
        Returns df with info

        Parameters
        ----------
        participant: str
            participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)
        roi2plot_list: list
            list with ROI names to plot
        all_regions: list
            with movement region name 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels  
        n_bins: int
            number of y coord bins to divide COM values into
        """

        ## store all COM values but also the binned version
        output_df = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'coordinates': [],
                                'COM': [], 'r2': [], 'movement_region': []})
        output_df_binned = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'coordinates': [],
                                'coordinates_std': [], 'COM': [], 'COM_std': [], 'movement_region': []})

        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        ## use CS as major axis for ROI coordinate rotation
        ref_theta = self.get_rotation_angle(roi_list = self.somaModelObj.MRIObj.params['plotting']['soma']['reference_roi'])

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.get_fs_coords(merge = True)

        ## loop over participant list
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get average beta values (all used in GLM)
                betas_pp, r2_pp = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)

                # path where Region contrasts were stored
                stats_dir = op.join(self.somaModelObj.stats_outputdir, 'sub-{sj}'.format(sj = pp), 'fixed_effects', fit_type)

                # load z-score localizer area, for region movements
                region_mask = {}
                region_mask['upper_limb'] = self.somaModelObj.load_zmask(region = 'upper_limb', filepth = stats_dir, 
                                                            fit_type = fit_type, fixed_effects = True, 
                                                            z_threshold = z_threshold, keep_b_evs = keep_b_evs)['B']
                region_mask['face'] = self.somaModelObj.load_zmask(region = 'face', filepth = stats_dir, 
                                                            fit_type = fit_type, fixed_effects = True, 
                                                            z_threshold = z_threshold, keep_b_evs = keep_b_evs)
                
                ## get positive and relevant r2
                r2_mask = np.zeros(r2_pp.shape)
                r2_mask[r2_pp > 0] = 1
            else:
                # load GLM estimates, and get betas and prediction
                soma_estimates = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)
                r2_pp = soma_estimates['r2']
                betas_pp = soma_estimates['betas']

            ## Get DM
            design_matrix = self.somaModelObj.load_design_matrix(pp, keep_b_evs = keep_b_evs, 
                                            custom_dm = True, nTRs = nr_TRs)

            ## set beta values and reg names in dict
            ## for all relevant regions
            region_regs_dict = {}
            region_betas_dict = {}

            reg_list = [] # also store all regressor names
            for region in all_regions:
                region_betas_dict[region] = self.somaModelObj.get_region_betas(betas_pp, region = region, dm = design_matrix)
                region_regs_dict[region] = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region]
                reg_list += region_regs_dict[region]

            ## make array of weights to use in bin
            weight_arr = r2_pp.copy()
            weight_arr[weight_arr<=0] = 0 # to not use negative weights

            # for each roi, get values and store in DF
            for roi2plot in roi2plot_list:

                ## get FS coordinates for each ROI vertex
                roi_vertices = {}
                roi_coords = {}
                for hemi in self.hemi_labels:
                    roi_vertices[hemi] = self.get_atlas_roi_vert(roi_list = all_rois[roi2plot],
                                                                hemi = hemi)
                    ## get FS coordinates for each ROI vertex
                    roi_coords[hemi] = self.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                            y_coord_surf[roi_vertices[hemi]])), 
                                                                        fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                        theta = ref_theta[hemi],
                                                                        roi_name = roi2plot+'_'+hemi)

                    # for each movement region
                    for region in all_regions:
                        
                        ## fixed effects mask * positive CV-r2
                        if region != 'face':
                            mask_bool = ((~np.isnan(region_mask['upper_limb'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
                        else:
                            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)

                        if not ((hemi == 'LH') and (region == 'left_hand')) or \
                            not ((hemi == 'RH') and (region == 'right_hand')):
                            
                            # calculate COM values
                            com_vals = self.somaModelObj.COM(region_betas_dict[region][:,roi_vertices[hemi]])[mask_bool]

                            # append
                            output_df = pd.concat((output_df,
                                                    pd.DataFrame({'sj': np.tile(pp, len(com_vals)), 
                                                                'ROI': np.tile(roi2plot, len(com_vals)), 
                                                                'hemisphere': np.tile(hemi, len(com_vals)), 
                                                                'movement_region': np.tile(region, len(com_vals)),
                                                                'coordinates': roi_coords[hemi][1][mask_bool],
                                                                'COM': com_vals, 
                                                                'r2': r2_pp[roi_vertices[hemi]][mask_bool]})
                                        ),ignore_index=True)
                            
                            # calculate weighted bins
                            binned_com, binned_com_std, binned_coord, binned_coord_std = self.get_weighted_mean_bins(pd.DataFrame({'com': com_vals, 
                                                                         'coords': roi_coords[hemi][1][mask_bool], 
                                                                         'r2': weight_arr[roi_vertices[hemi]][mask_bool]}), 
                                                           x_key = 'com', y_key = 'coords', sort_key = 'coords',
                                                           weight_key = 'r2', n_bins = n_bins)
                            
                            # append
                            output_df_binned = pd.concat((output_df_binned,
                                                    pd.DataFrame({'sj': np.tile(pp, len(binned_com)), 
                                                                'ROI': np.tile(roi2plot, len(binned_com)), 
                                                                'hemisphere': np.tile(hemi, len(binned_com)), 
                                                                'movement_region': np.tile(region, len(binned_com)),
                                                                'coordinates': binned_coord,
                                                                'coordinates_std': binned_coord_std,
                                                                'COM': binned_com,
                                                                'COM_std': binned_com_std})
                                        ),ignore_index=True)

        return output_df, output_df_binned

    def get_RF_coords_df(self, participant_list, data_RFmodel = None,
                                fit_type = 'loo_run', keep_b_evs = True,
                                roi2plot_list = ['M1', 'S1'], z_threshold = 3.1,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'],
                                all_rois = {'M1': ['4'], 'S1': ['3b'], 'CS': ['3a'], 
                                'BA43': ['43', 'OP4'], 'S2': ['OP1'],
            'Insula': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3','MI', 'AVI', 'AAIC']
            }):

        """
        Helper function to get RF values over y axis 
        (for all relevant vertices and also binned)
        for each movement region of interest
        and for selected ROIs.
        Returns df with info

        Parameters
        ----------
        participant: str
            participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out)  
        keep_b_evs: bool
            if we want to specify regressors for simultaneous movement or not (ex: both hands)
        roi2plot_list: list
            list with ROI names to plot
        all_regions: list
            with movement region name 
        all_rois: dict
            dictionary with names of ROIs and and list of glasser atlas labels  
        n_bins: int
            number of y coord bins to divide COM values into
        """

        ## store all COM values but also the binned version
        output_df = pd.DataFrame({'sj': [], 'ROI': [], 'hemisphere': [], 'coordinates': [],
                                'center': [], 'size': [], 'slope': [], 'RF_r2': [], 'r2': [], 'movement_region': []})

        # check if atlas df exists
        try:
            self.atlas_df
        except AttributeError:
            # load atlas ROI df
            self.get_atlas_roi_df(return_RGBA = False)

        ## use CS as major axis for ROI coordinate rotation
        ref_theta = self.get_rotation_angle(roi_list = self.somaModelObj.MRIObj.params['plotting']['soma']['reference_roi'])

        ## get surface x and y coordinates
        x_coord_surf, y_coord_surf, _ = self.get_fs_coords(merge = True)

        ## loop over participant list
        for pp in participant_list:
            
            ## LOAD R2
            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get r2 values (all used in GLM)
                _, r2_pp = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)

                # path where Region contrasts were stored
                stats_dir = op.join(self.somaModelObj.stats_outputdir, 'sub-{sj}'.format(sj = pp), 'fixed_effects', fit_type)

                # load z-score localizer area, for region movements
                region_mask = {}
                region_mask['upper_limb'] = self.somaModelObj.load_zmask(region = 'upper_limb', filepth = stats_dir, 
                                                            fit_type = fit_type, fixed_effects = True, 
                                                            z_threshold = z_threshold, keep_b_evs = keep_b_evs)['B']
                region_mask['face'] = self.somaModelObj.load_zmask(region = 'face', filepth = stats_dir, 
                                                            fit_type = fit_type, fixed_effects = True, 
                                                            z_threshold = z_threshold, keep_b_evs = keep_b_evs)
                
                ## get positive and relevant r2
                r2_mask = np.zeros(r2_pp.shape)
                r2_mask[r2_pp > 0] = 1
            else:
                # load GLM estimates, and get betas and prediction
                r2_pp = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)['r2']

            ## load RF estimates
            RF_estimates = data_RFmodel.load_estimates(pp, betas_model = 'glm', region_keys = all_regions,
                                                            fit_type = fit_type)

            ## set reg names in dict
            ## for all relevant regions
            region_regs_dict = {}
            for region in all_regions:
                region_regs_dict[region] = self.somaModelObj.MRIObj.params['fitting']['soma']['all_contrasts'][region]

            ## make array of weights to use in bin
            weight_arr = r2_pp.copy()
            weight_arr[weight_arr<=0] = 0 # to not use negative weights

            # for each roi, get values and store in DF
            for roi2plot in roi2plot_list:

                ## get FS coordinates for each ROI vertex
                roi_vertices = {}
                roi_coords = {}
                for hemi in self.hemi_labels:
                    roi_vertices[hemi] = self.get_atlas_roi_vert(roi_list = all_rois[roi2plot],
                                                                hemi = hemi)
                    ## get FS coordinates for each ROI vertex
                    roi_coords[hemi] = self.transform_roi_coords(np.vstack((x_coord_surf[roi_vertices[hemi]], 
                                                                                            y_coord_surf[roi_vertices[hemi]])), 
                                                                        fig_pth = op.join(self.somaModelObj.MRIObj.derivatives_pth, 'plots', 'PCA_ROI'), 
                                                                        theta = ref_theta[hemi],
                                                                        roi_name = roi2plot+'_'+hemi)

                    # for each movement region
                    for region in all_regions:
                        
                        ## fixed effects mask * positive CV-r2
                        if region != 'face':
                            mask_bool = ((~np.isnan(region_mask['upper_limb'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)
                        else:
                            mask_bool = ((~np.isnan(region_mask['face'][roi_vertices[hemi]]))*r2_mask[roi_vertices[hemi]]).astype(bool)

                        if not ((hemi == 'LH') and (region == 'left_hand')) or \
                            not ((hemi == 'RH') and (region == 'right_hand')):
                            
                            # append
                            output_df = pd.concat((output_df,
                                                    pd.DataFrame({'sj': np.tile(pp, len(roi_coords[hemi][1][mask_bool])), 
                                                                'ROI': np.tile(roi2plot, len(roi_coords[hemi][1][mask_bool])), 
                                                                'hemisphere': np.tile(hemi, len(roi_coords[hemi][1][mask_bool])), 
                                                                'movement_region': np.tile(region, len(roi_coords[hemi][1][mask_bool])),
                                                                'coordinates': roi_coords[hemi][1][mask_bool],
                                                                'center': np.array(RF_estimates[region]['mu'])[roi_vertices[hemi]][mask_bool], 
                                                                'size': np.array(RF_estimates[region]['size'])[roi_vertices[hemi]][mask_bool],
                                                                'slope': np.array(RF_estimates[region]['slope'])[roi_vertices[hemi]][mask_bool],
                                                                'RF_r2': np.array(RF_estimates[region]['r2'])[roi_vertices[hemi]][mask_bool],
                                                                'r2': r2_pp[roi_vertices[hemi]][mask_bool]})
                                        ),ignore_index=True)
                            
        return output_df

class pRFViewer(Viewer):

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

        # need to initialize parent class (Model), indicating output infos
        super().__init__(pysub = pysub, derivatives_pth = pRFModelObj.MRIObj.derivatives_pth, MRIObj = pRFModelObj.MRIObj)

        # set object to use later on
        self.pRFModelObj = pRFModelObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.derivatives_pth, 'plots')
        else:
            self.outputdir = outputdir

    def plot_pa_colorwheel(self, resolution=800, angle_thresh = 3*np.pi/4, cmap_name = 'hsv', continuous = True, fig_name = None):

        """
        Helper function to create colorwheel image
        for polar angle plots returns 
        Parameters
        ----------
        resolution : int
            resolution of mesh
        angle_thresh: float
            value upon which to make it red for this hemifield (above angle or below 1-angle will be red in a retinotopy hsv color wheel)
            if angle threh different than PI then assumes non uniform colorwheel
        cmap_name: str/list
            colormap name (if string) or list of colors to use for colormap
        continuous: bool
            if continuous colormap or binned
        """

        ## make circle
        circle_x, circle_y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
        circle_radius = np.sqrt(circle_x**2 + circle_y**2)
        circle_pa = np.arctan2(circle_y, circle_x) # all polar angles calculated from our mesh
        circle_pa[circle_radius > 1] = np.nan # then we're excluding all parts of bitmap outside of circle

        if isinstance(cmap_name, str):

            cmap = plt.get_cmap('hsv')
            norm = colors.Normalize(-angle_thresh, angle_thresh) # normalize between the point where we defined our color threshold
        
        elif isinstance(cmap_name, list) or isinstance(cmap_name, np.ndarray):

            if continuous:
                cvals  = np.arange(len(cmap_name))
                norm = plt.Normalize(min(cvals),max(cvals))
                tuples = list(zip(map(norm,cvals), cmap_name))
                
                colormap = colors.LinearSegmentedColormap.from_list("", tuples)
                norm = colors.Normalize(-angle_thresh, angle_thresh) 

            else:
                colormap = colors.ListedColormap(cmap_name)
                #boundaries = np.linspace(0,1,len(cmap_name))
                #norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)
                norm = colors.Normalize(-angle_thresh, angle_thresh) 

        # non-uniform colorwheel
        if angle_thresh != np.pi:
            
            ## for LH (RVF)
            circle_pa_left = circle_pa.copy()
            # between thresh angle make it red
            circle_pa_left[(circle_pa_left < -angle_thresh) | (circle_pa_left > angle_thresh)] = angle_thresh

            plt.imshow(circle_pa_left, cmap=cmap, norm=norm,origin='lower') # origin lower because imshow flips it vertically, now in right order for VF
            plt.axis('off')

            plt.savefig('{fn}_colorwheel_4LH-RVF.png'.format(fn = fig_name),dpi=100)

            ## for RH (LVF)
            circle_pa_right = circle_pa.copy()
            circle_pa_right = np.fliplr(circle_pa_right)
            # between thresh angle make it red
            circle_pa_right[(circle_pa_right < -angle_thresh) | (circle_pa_right > angle_thresh)] = angle_thresh

            plt.imshow(circle_pa_right, cmap=cmap, norm=norm,origin='lower')
            plt.axis('off')

            plt.savefig('{fn}_colorwheel_4RH-LVF.png'.format(fn = fig_name),dpi=100)

        else:
            plt.imshow(circle_pa, cmap = colormap, norm=norm, origin='lower')
            plt.axis('off')

            if continuous:
                plt.savefig('{fn}_colorwheel_continuous.png'.format(fn = fig_name),dpi=100)
            else:
                plt.savefig('{fn}_colorwheel_discrete.png'.format(fn = fig_name),dpi=100)

    def get_NONuniform_polar_angle(self, xx = [], yy = [], rsq = [], angle_thresh = 3*np.pi/4, rsq_thresh = 0):

        """
        Helper function to transform polar angle values into RGB values
        guaranteeing a non-uniform representation
        (this is, when we want to use half the color wheel to show the pa values)
        (useful for better visualization of boundaries)
        Parameters
        ----------
        xx : arr
            array with x position values
        yy : arr
            array with y position values
        rsq: arr
            rsq values, to be used as alpha level/threshold
        angle_thresh: float
            value upon which to make it red for this hemifield (above angle or below 1-angle will be red in a retinotopy hsv color wheel)
        rsq_thresh: float/int
            minimum rsq threshold to use 
        pysub: str
            name of pycortex subject folder
        """

        hsv_angle = []
        hsv_angle = np.ones((len(rsq), 3))

        ## calculate polar angle
        polar_angle = np.angle(xx + yy * 1j)

        ## set normalized polar angle (0-1), and make nan irrelevant vertices
        hsv_angle[:, 0] = np.nan 
        hsv_angle[:, 0][rsq > rsq_thresh] = ((polar_angle + np.pi) / (np.pi * 2.0))[rsq > rsq_thresh]

        ## normalize angle threshold for overepresentation
        angle_thresh_norm = (angle_thresh + np.pi) / (np.pi * 2.0)

        ## get mid vertex index (diving hemispheres)
        left_index = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

        ## set angles within threh interval to 0
        ind_thresh = np.where((hsv_angle[:left_index, 0] > angle_thresh_norm) | (hsv_angle[:left_index, 0] < 1-angle_thresh_norm))[0]
        hsv_angle[:left_index, 0][ind_thresh] = 0

        ## now take angles from RH (thus LVF) 
        #### ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres) ###
        # also normalize it
        hsv_angle[left_index:, 0] = ((np.angle(-1*xx + yy * 1j) + np.pi) / (np.pi * 2.0))[left_index:]

        # set angles within threh interval to 0
        ind_thresh = np.where((hsv_angle[left_index:, 0] > angle_thresh_norm) | (hsv_angle[left_index:, 0] < 1-angle_thresh_norm))[0]
        hsv_angle[left_index:, 0][ind_thresh] = 0

        ## make final RGB array
        rgb_angle = np.ones((len(rsq), 3))
        rgb_angle[:] = np.nan

        rgb_angle[rsq > rsq_thresh] = colors.hsv_to_rgb(hsv_angle[rsq > rsq_thresh])

        return rgb_angle

    def get_estimates_roi_df(self, participant, estimates_pp, ROIs = None, roi_verts = None, est_key = 'r2', model = 'gauss'):

        """
        Helper function to get estimates dataframe values for each ROI
        will select values based on est key param 
        """

        ## save rsq values in dataframe, for plotting
        df_est = pd.DataFrame({'sj': [], 'index': [], 'ROI': [], 'value': [], 'model': []})

        for idx,rois_ks in enumerate(ROIs): 
            
            # mask estimates
            print('masking estimates for ROI %s'%rois_ks)

            if len(roi_verts[rois_ks]) > 0:
                if isinstance(estimates_pp, dict):
                    roi_arr = estimates_pp[est_key][roi_verts[rois_ks]]
                else:
                    roi_arr = estimates_pp[roi_verts[rois_ks]]
            else:
                print('No vertices found for ROI')
                roi_arr = [np.nan]

            df_est = pd.concat((df_est,
                                pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = participant), len(roi_arr)), 
                                            'index': roi_verts[rois_ks], 
                                            'ROI': np.tile(rois_ks, len(roi_arr)), 
                                            'value': roi_arr,
                                            'model': np.tile(model, len(roi_arr))})
                            ))

        return df_est

    def get_Wmean_estimate_roi_df(self, participant, estimates_arr = [], weights_arr = [],
                                   ROIs = None, roi_verts = None, model = 'gauss'):

        """
        Helper function to get estimates dataframe values for each ROI
        will average values based on est key param and weight (r2)
        """

        ## save rsq values in dataframe, for plotting
        df_est = pd.DataFrame({'sj': [], 'ROI': [], 'value': [], 'model': []})

        for idx,rois_ks in enumerate(ROIs): 
            
            # mask estimates
            print('masking estimates for ROI %s'%rois_ks)

            if len(roi_verts[rois_ks]) > 0:
                roi_arr = estimates_arr[roi_verts[rois_ks]]
                roi_weights = weights_arr[roi_verts[rois_ks]]

                # remove nans, and average
                not_nan_ind = np.where((~np.isnan(roi_weights)))[0]
                avg_value = np.average(roi_arr[not_nan_ind], axis = 0, weights = roi_weights[not_nan_ind])
            else:
                print('No vertices found for ROI')
                avg_value = np.nan

            df_est = pd.concat((df_est,
                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = participant)], 
                                            'ROI': [rois_ks], 
                                            'value': [avg_value],
                                            'model': [model]})
                            ))

        return df_est

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
                                fit_type = 'mean_run', prf_model_name = 'gauss', max_ecc_ext = None,
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


        # Now actually plot results
         
        ### RSQ ###
        self.plot_rsq(participant_list = participant_list, group_estimates = group_estimates, fit_type = fit_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)
        
        ### PA ###
        self.plot_pa(participant_list = participant_list, group_estimates = group_estimates, fit_type = fit_type,
                    model_name = prf_model_name, figures_pth = figures_pth, n_bins_colors = 256, max_x_lim = 5, angle_thresh = 3*np.pi/4)
        
        ### Exponent - only for CSS ###
        if prf_model_name == 'css':
            self.plot_exponent(participant_list = participant_list, group_estimates = group_estimates, fit_type = fit_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)
            
        ### ECC SIZE ###
        self.plot_ecc_size(participant_list = participant_list, group_estimates = group_estimates, fit_type = fit_type,
                            model_name = prf_model_name, figures_pth = figures_pth,
                            n_bins_colors = 256, max_ecc_ext = 6, max_size_ext = 14, n_bins_dist = 8)
        
        ### Visual Field coverage ###
        self.plot_VFcoverage(participant_list = participant_list, group_estimates = group_estimates, fit_type = fit_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)
        
    def plot_rsq(self, participant_list = [], group_estimates = [], figures_pth = None, 
                        model_name = 'gauss', fit_type = 'mean_run'):

        """
        Plot rsq - flatmap and violinplot for ROIs
        for all participants in list
        """

        ## get ROI vertices
        roi_verts = self.get_ROI_verts_dict(ROIs = self.pRFModelObj.MRIObj.params['plotting']['prf']['ROIs'])

        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq', 'pRF', fit_type)
        
        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()

        ## loop over participants in list
        for pp in participant_list:
            
            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            fig_name = op.join(sub_figures_pth, 'sub-{sj}_task-pRF_model-{model}_flatmap_RSQ.png'.format(sj = pp,
                                                                                                    model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            #### plot flatmap ###
            flatmap = self.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                                        vmin1 = 0, vmax1 = .7,
                                                        cmap = 'hot',
                                                        fig_abs_name = fig_name)
            
            ## for each participant, get dataframe with 
            # estimate value per ROI
            df_estimates_ROIs = self.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                         ROIs = roi_verts.keys(), 
                                                        roi_verts = roi_verts, 
                                                        est_key = 'r2', model = model_name)
            
            ## plot violinplot
            fig, ax1 = plt.subplots(1,1, figsize=(20,7.5), dpi=100, facecolor='w', edgecolor='k')
            v1 = sns.violinplot(data = df_estimates_ROIs, x = 'ROI', y = 'value', 
                                cut=0, inner='box', palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'],
                                linewidth=2.7,saturation = 1, ax = ax1) 
            # change alpha
            plt.setp(v1.collections, alpha=.7, edgecolor = None)
                    
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))
            
            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df,
                                    df_estimates_ROIs))
            
        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            # point plot with group average and standard error of the mean
            v1 = sns.barplot(data = avg_roi_df, x = 'ROI', y = 'value', 
                        palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'],
                        ci = 95,n_boot=5000, ax=ax1)#, scale=2, capsize = .2) 
            # change alpha
            plt.setp(v1.collections, alpha=.7, edgecolor = None)

            # striplot with median value for all participants
            sns.stripplot(data = avg_roi_df.groupby(['sj', 'ROI'])['value'].median().reset_index(),
                        x = 'ROI', y = 'value', order = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'].keys(),
                        color = 'gray', alpha = 0.3,linewidth=.2, edgecolor='k',ax=ax1)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

            fig_name = op.join(figures_pth, 'sub-group_task-pRF_model-{model}_pointplot_RSQ.png'.format(model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 
            fig.savefig(fig_name)

    def plot_pa(self, participant_list = [], group_estimates = [], figures_pth = None, 
                    model_name = 'gauss', fit_type = 'mean_run', n_bins_colors = 256, max_x_lim = 5, angle_thresh = 3*np.pi/4):
        
        """
        Plot polar angle estimates - flatmaps -
        for all participants in list
        """

        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'polar_angle', fit_type)

        # get matplotlib color map from segmented colors
        PA_cmap = self.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                        '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                        cmap_name = 'PA_mackey_custom',
                                        discrete = False, add_alpha = False, return_cmap = True)

        ## loop over participants in list
        for pp in participant_list:
            
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = self.pRFModelObj.normalize(np.clip(r2, 0, .5)) # normalize 
            alpha_level[np.where((np.isnan(r2)))[0]] = np.nan

            ## position estimates
            xx = group_estimates['sub-{sj}'.format(sj = pp)]['x']
            yy = group_estimates['sub-{sj}'.format(sj = pp)]['y']

            ## calculate polar angle 
            complex_location = xx + yy * 1j 

            polar_angle = np.angle(complex_location)
            polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0)) # normalize PA between 0 and 1
            polar_angle_norm[np.where((np.isnan(r2)))[0]] = np.nan

            
            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            fig_name = op.join(sub_figures_pth, 'sub-{sj}_task-pRF_model-{model}_flatmap_PA.png'.format(sj = pp,
                                                                                                    model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            #### plot flatmap ###
            flatmap = self.plot_flatmap(polar_angle_norm, est_arr2 = alpha_level,
                                        cmap = PA_cmap, 
                                        vmin1 = 0, vmax1 = 1, 
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            # also plot non-uniform color wheel
            rgb_pa = self.get_NONuniform_polar_angle(xx = xx, yy = yy, rsq = r2, 
                                                angle_thresh = angle_thresh, 
                                                rsq_thresh = 0)
            ## make ones mask,
            #ones_mask = np.ones(r2.shape)
            #ones_mask[np.where((np.isnan(r2)))[0]] = np.nan

            fig_name = fig_name.replace('_PA', '_PAnonUNI')

            #### plot flatmap ###
            flatmap = self.plot_RGBflatmap(rgb_arr = rgb_pa, alpha_arr = alpha_level, #ones_mask,
                                            fig_abs_name = fig_name)
            
            # plot x and y separately, for sanity check
            # XX
            fig_name = fig_name.replace('_PAnonUNI', '_XX')

            #### plot flatmap ###
            flatmap = self.plot_flatmap(xx, est_arr2 = alpha_level,
                                        cmap = 'BuBkRd_alpha_2D', 
                                        vmin1 = -max_x_lim, vmax1 = max_x_lim, 
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            # YY
            fig_name = fig_name.replace('_XX', '_YY')

            #### plot flatmap ###
            flatmap = self.plot_flatmap(yy, est_arr2 = alpha_level,
                                        cmap = 'BuBkRd_alpha_2D', 
                                        vmin1 = -max_x_lim, vmax1 = max_x_lim, 
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## plot the colorwheels as figs
            
            # non uniform colorwheel
            self.plot_pa_colorwheel(resolution=800, angle_thresh = angle_thresh, cmap_name = 'hsv', 
                                            continuous = True, fig_name = op.join(sub_figures_pth, 'hsv'))

            # uniform colorwheel, continuous
            self.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                                    cmap_name = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'], 
                                            continuous = True, fig_name = op.join(sub_figures_pth, 'PA_mackey'))

            # uniform colorwheel, discrete
            self.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                                    cmap_name = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'], 
                                            continuous = False, fig_name = op.join(sub_figures_pth, 'PA_mackey'))

    def plot_exponent(self, participant_list = [], group_estimates = [], figures_pth = None, 
                        model_name = 'css', fit_type = 'mean_run'):

        """
        Plot exponent - flatmap and violinplot for ROIs
        for all participants in list
        """

        ## get ROI vertices
        roi_verts = self.get_ROI_verts_dict(ROIs = self.pRFModelObj.MRIObj.params['plotting']['prf']['ROIs'])

        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'exponent', fit_type)
        
        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()

        ## loop over participants in list
        for pp in participant_list:
            
            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            fig_name = op.join(sub_figures_pth, 'sub-{sj}_task-pRF_model-{model}_flatmap_Exponent.png'.format(sj = pp,
                                                                                                    model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = self.pRFModelObj.normalize(np.clip(r2, 0, .5)) # normalize 

            #### plot flatmap ###
            flatmap = self.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                        est_arr2 = alpha_level,
                                        cmap = 'plasma', 
                                        vmin1 = 0, vmax1 = 1, 
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## for each participant, get dataframe with 
            # estimate value per ROI
            df_estimates_ROIs = self.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                         ROIs = roi_verts.keys(), 
                                                        roi_verts = roi_verts, 
                                                        est_key = 'ns', model = model_name)
            
            ## plot violinplot
            fig, ax1 = plt.subplots(1,1, figsize=(20,7.5), dpi=100, facecolor='w', edgecolor='k')
            v1 = sns.violinplot(data = df_estimates_ROIs, x = 'ROI', y = 'value', 
                                cut=0, inner='box', palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'],
                                linewidth=2.7,saturation = 1, ax = ax1) 
            # change alpha
            plt.setp(v1.collections, alpha=.7, edgecolor = None)
                    
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('CSS Exponent',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))
            
            ## concatenate average per participant, weighted by r2
            # to make group plot
            avg_roi_df = pd.concat((avg_roi_df,
                                    self.get_Wmean_estimate_roi_df(pp, 
                                                                   estimates_arr = group_estimates['sub-{sj}'.format(sj = pp)]['ns'],
                                                                    weights_arr = group_estimates['sub-{sj}'.format(sj = pp)]['r2'],
                                                            ROIs = roi_verts.keys(), roi_verts = roi_verts, model = model_name)))
            
        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            # point plot with group average and standard error of the mean
            v1 = sns.barplot(data = avg_roi_df, x = 'ROI', y = 'value', 
                        palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'],
                        ci = 95,n_boot=5000, ax=ax1)#, scale=2, capsize = .2) 
            # change alpha
            plt.setp(v1.collections, alpha=.7, edgecolor = None)

            # striplot with median value for all participants
            sns.stripplot(data = avg_roi_df,
                        x = 'ROI', y = 'value', order = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'].keys(),
                        color = 'gray', alpha = 0.3,linewidth=.2, edgecolor='k',ax=ax1)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('CSS Exponent',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

            fig_name = op.join(figures_pth, 'sub-group_task-pRF_model-{model}_pointplot_Exponent.png'.format(model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 
            fig.savefig(fig_name)

    def plot_VFcoverage(self, participant_list = [], group_estimates = [], figures_pth = None, 
                        model_name = 'gauss', fit_type = 'mean_run', vert_lim_dva = 5.5, hor_lim_dva = 8.8):

        """
        Plot visual field coverage - hexbins for each ROI
        for all participants in list
        """

        ## get ROI vertices
        roi_verts = self.get_ROI_verts_dict(ROIs = self.pRFModelObj.MRIObj.params['plotting']['prf']['ROIs'])

        # get mid vertex index (diving hemispheres)
        left_index = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'VF_coverage', fit_type)
        
        # save values per roi in dataframe
        df_merge = pd.DataFrame()

        ## loop over participants in list
        for pp in participant_list:
            
            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)
            
            ## for each participant, get dataframe with 
            # estimate value per ROI
            xx_pp_roi_df = self.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                        ROIs = roi_verts.keys(), 
                                                        roi_verts = roi_verts, 
                                                        est_key = 'x', model = model_name)
            yy_pp_roi_df = self.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                        ROIs = roi_verts.keys(), 
                                                        roi_verts = roi_verts, 
                                                        est_key = 'y', model = model_name)
            
            # left hemisphere
            df_LH = pd.concat((xx_pp_roi_df[xx_pp_roi_df['index'] < left_index].rename(columns={'value': 'x'}),
                    yy_pp_roi_df[yy_pp_roi_df['index'] < left_index].rename(columns={'value': 'y'})[['y']]), axis = 1)
            df_LH['hemisphere'] = 'LH'

            # right hemisphere
            df_RH = pd.concat((xx_pp_roi_df[xx_pp_roi_df['index'] >= left_index].rename(columns={'value': 'x'}),
                    yy_pp_roi_df[yy_pp_roi_df['index'] >= left_index].rename(columns={'value': 'y'})[['y']]), axis = 1)
            df_RH['hemisphere'] = 'RH'

            ## save in merged DF
            df_merge = pd.concat((df_merge, pd.concat((df_LH, df_RH))))

            # actually plot hexabins
            fig_name = op.join(sub_figures_pth, 'sub-{sj}_task-pRF_model-{model}_VFcoverage.png'.format(sj = pp,
                                                                                                        model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            for r_name in roi_verts.keys():

                f, ss = plt.subplots(1, 1, figsize=(8,4.5))

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'LH') & \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].x.values,
                        df_merge[(df_merge['hemisphere'] == 'LH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].y.values,
                    gridsize=15, 
                    cmap='Greens',
                    extent= np.array([-1, 1, -1, 1]) * hor_lim_dva,
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=1)

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].x.values,
                        df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].y.values,
                    gridsize=15, 
                    cmap='Reds',
                    extent= np.array([-1, 1, -1, 1]) * hor_lim_dva,
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=.5)

                plt.xticks(fontsize = 20)
                plt.yticks(fontsize = 20)
                plt.tight_layout()
                plt.ylim(-vert_lim_dva, vert_lim_dva) #-6,6)#
                ss.set_aspect('auto')
                # set middle lines
                ss.axvline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
                ss.axhline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')

                # custom lines only to make labels
                custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                                Line2D([0], [0], color='r',alpha=0.5, lw=4)]

                plt.legend(custom_lines, self.hemi_labels, fontsize = 18)
                fig_hex = plt.gcf()

                fig_hex.savefig(fig_name.replace('_VFcoverage','_VFcoverage_{rn}'.format(rn = r_name)))

        if len(participant_list) > 1:

            # actually plot hexabins
            fig_name = op.join(figures_pth, 'sub-group_task-pRF_model-{model}_VFcoverage.png'.format(model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            for r_name in roi_verts.keys():

                f, ss = plt.subplots(1, 1, figsize=(8,4.5))

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'LH')& \
                                (df_merge['ROI'] == r_name)].x.values,
                        df_merge[(df_merge['hemisphere'] == 'LH')& \
                                (df_merge['ROI'] == r_name)].y.values,
                    gridsize=15, 
                    cmap='Greens',
                    extent= np.array([-1, 1, -1, 1]) * hor_lim_dva,
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=1)

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'RH')& \
                                (df_merge['ROI'] == r_name)].x.values,
                        df_merge[(df_merge['hemisphere'] == 'RH')& \
                                (df_merge['ROI'] == r_name)].y.values,
                    gridsize=15, 
                    cmap='Reds',
                    extent= np.array([-1, 1, -1, 1]) * hor_lim_dva,
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=.5)

                plt.xticks(fontsize = 20)
                plt.yticks(fontsize = 20)
                plt.tight_layout()
                plt.ylim(-vert_lim_dva, vert_lim_dva) #-6,6)#
                ss.set_aspect('auto')
                # set middle lines
                ss.axvline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
                ss.axhline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')

                # custom lines only to make labels
                custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                                Line2D([0], [0], color='r',alpha=0.5, lw=4)]

                plt.legend(custom_lines, self.hemi_labels,fontsize = 18)
                fig_hex = plt.gcf()

                fig_hex.savefig(fig_name.replace('_VFcoverage','_VFcoverage_{rn}'.format(rn = r_name)))

    def plot_ecc_size(self, participant_list = [], group_estimates = [], figures_pth = None, 
                    model_name = 'gauss', fit_type = 'mean_run', n_bins_colors = 256, 
                    max_ecc_ext = 6, max_size_ext = 14, n_bins_dist = 20):
        
        """
        Plot ecc and size - flatmaps and linear relationship - 
        for all participants in list
        """

        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'ecc_size', fit_type)

        # get matplotlib color map from segmented colors
        ecc_cmap = self.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                    bins = n_bins_colors, cmap_name = 'ECC_mackey_custom', 
                                    discrete = False, add_alpha = False, return_cmap = True)
        
        ## get ROI vertices
        roi_verts = self.get_ROI_verts_dict(ROIs = self.pRFModelObj.MRIObj.params['plotting']['prf']['ROIs'])
        
        avg_bin_df = pd.DataFrame()

        ## loop over participants in list
        for pp in participant_list:

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)
            
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = self.pRFModelObj.normalize(np.clip(r2, 0, .5)) # normalize 

            ## get ECCENTRICITY estimates
            complex_location = group_estimates['sub-{sj}'.format(sj = pp)]['x'] + group_estimates['sub-{sj}'.format(sj = pp)]['y'] * 1j # calculate eccentricity values
            eccentricity = np.abs(complex_location)

            fig_name = op.join(sub_figures_pth, 'sub-{sj}_task-pRF_model-{model}_flatmap_ECC.png'.format(sj = pp,
                                                                                                    model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 
            
            #### plot flatmap ###
            flatmap = self.plot_flatmap(eccentricity,
                                        est_arr2 = alpha_level,
                                        cmap = ecc_cmap, 
                                        vmin1 = 0, vmax1 = max_ecc_ext, 
                                        vmin2 = 0, vmax2 = 1,
                                        fig_abs_name = fig_name)
            
            ## get SIZE estimates
            if model_name in ['dn', 'dog']:
                size_fwhmax, fwatmin = self.pRFModelObj.fwhmax_fwatmin(model_name, group_estimates['sub-{sj}'.format(sj = pp)])
            else: 
                size_fwhmax = self.pRFModelObj.fwhmax_fwatmin(model_name, group_estimates['sub-{sj}'.format(sj = pp)])
            size_fwhmax[np.isnan(r2)] = np.nan

            fig_name = fig_name.replace('_ECC', '_SIZEfwhm')

            #### plot flatmap ###
            flatmap = self.plot_flatmap(size_fwhmax,
                                        est_arr2 = alpha_level,
                                        cmap = 'hot', 
                                        vmin1 = 0, vmax1 = max_size_ext, 
                                        vmin2 = 0, vmax2 = 1,
                                        fig_abs_name = fig_name)
            
            ## GET values per ROI ##
            ecc_pp_roi_df = self.get_estimates_roi_df(pp, eccentricity, 
                                                            ROIs = roi_verts.keys(), 
                                                            roi_verts = roi_verts, 
                                                            model = model_name)

            size_pp_roi_df = self.get_estimates_roi_df(pp, size_fwhmax, 
                                                            ROIs = roi_verts.keys(), 
                                                            roi_verts = roi_verts, 
                                                            model = model_name)

            r2_pp_roi_df = self.get_estimates_roi_df(pp, r2, 
                                                        ROIs = roi_verts.keys(), 
                                                        roi_verts = roi_verts, 
                                                        model = model_name)
            
            # merge them into one
            df_ecc_siz = pd.merge(ecc_pp_roi_df.rename(columns={'value': 'ecc'}),
                                size_pp_roi_df.rename(columns={'value': 'size'}))
            df_ecc_siz = pd.merge(df_ecc_siz, r2_pp_roi_df.rename(columns={'value': 'r2'}))

            ## drop the nans
            df_ecc_siz = df_ecc_siz[~np.isnan(df_ecc_siz.r2)]

            ##### plot unbinned df #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = df_ecc_siz, 
                           scatter_kws={'alpha':0.05}, scatter=True, 
                        palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = fig_name.replace('_flatmap_SIZEfwhm', '_ecc_vs_size_UNbinned')

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ## bin it, for cleaner plot
            for r_name in roi_verts.keys():

                mean_x, _, mean_y, _ = self.get_weighted_mean_bins(df_ecc_siz.loc[(df_ecc_siz['ROI'] == r_name) & \
                                                                                    (df_ecc_siz['r2'].notna())],
                                                                                    x_key = 'ecc', y_key = 'size', weight_key = 'r2', n_bins = n_bins_dist)

                avg_bin_df = pd.concat((avg_bin_df,
                                        pd.DataFrame({ 'sj': np.tile('sub-{sj}'.format(sj = pp), len(mean_x)),
                                                    'ROI': np.tile(r_name, len(mean_x)),
                                                    'ecc': mean_x,
                                                    'size': mean_y
                                        })))
                
            ##### plot binned df #########
            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df.loc[avg_bin_df['sj'] == 'sub-{sj}'.format(sj = pp)], 
                           scatter_kws={'alpha':0.15}, scatter=True, 
                        palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal']) 

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = fig_name.replace('_ecc_vs_size_UNbinned', '_ecc_vs_size_binned')

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

        if len(participant_list) > 1:

            ##### plot binned df for GROUP #########
            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df,
                           scatter_kws={'alpha':0.15}, scatter=True, 
                        palette = self.pRFModelObj.MRIObj.params['plotting']['prf']['colormaps']['ROI_pal'],
                        x_bins = 8) 

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = op.join(figures_pth, 'sub-group_task-pRF_model-{model}_ecc_vs_size_binned.png'.format(model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ## subdivide into areas
            ### plot for Occipital Areas - V1 V2 V3 V3AB hV4 LO ###
            roi2plot = self.pRFModelObj.MRIObj.params['plotting']['prf']['occipital']

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df[avg_bin_df.ROI.isin(roi2plot)],
                           scatter_kws={'alpha':0.15}, scatter=True, 
                        palette="YlGnBu_r", markers=['^','s','o','v','D','h'],
                        x_bins = 8)

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name_region = fig_name.replace('_binned', '_binned_occipital')
            fig2.savefig(fig_name_region, dpi=100,bbox_inches = 'tight')

            ### plot for Parietal Areas - IPS0 IPS1 IPS2+ ###
            roi2plot = self.pRFModelObj.MRIObj.params['plotting']['prf']['parietal']

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df[avg_bin_df.ROI.isin(roi2plot)],
                           scatter_kws={'alpha':0.15}, scatter=True, 
                        palette="YlOrRd",markers=['^','s','o'],
                        x_bins = 8)

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name_region = fig_name.replace('_binned', '_binned_parietal')
            fig2.savefig(fig_name_region, dpi=100,bbox_inches = 'tight')

            ### plot for Frontal Areas - sPCS iPCS ###
            roi2plot = self.pRFModelObj.MRIObj.params['plotting']['prf']['frontal']

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df[avg_bin_df.ROI.isin(roi2plot)],
                           scatter_kws={'alpha':0.15}, scatter=True, 
                        palette="PuRd",markers=['^','s'],
                        x_bins = 8)

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name_region = fig_name.replace('_binned', '_binned_frontal')
            fig2.savefig(fig_name_region, dpi=100,bbox_inches = 'tight')


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
        alpha_level = self.pRFModelObj.normalize(np.clip(estimates_dict['r2'], rsq_threshold, .5)) # normalize 
        alpha_level[nan_mask] = np.nan
        
        ## pRF rsq
        click_plotter.images['pRF_rsq'] = self.get_flatmaps(estimates_dict['r2'], 
                                                            vmin1 = 0, vmax1 = .8,
                                                            cmap = 'Reds')

        ## pRF Eccentricity

        # make custom colormap
        ecc_cmap = self.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = 256, cmap_name = 'ECC_mackey_custom', 
                                            discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['ecc'] = self.make_raw_vertex_image(eccentricity, 
                                                            cmap = ecc_cmap, 
                                                            vmin = 0, vmax = 6, 
                                                            data2 = alpha_level, 
                                                            vmin2 = 0, vmax2 = 1, 
                                                            subject = self.pysub, data2D = True)

        ## pRF Size
        click_plotter.images['size_fwhmax'] = self.make_raw_vertex_image(size_fwhmax, 
                                                    cmap = 'hot', vmin = 0, vmax = 14, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub, data2D = True)

        ## pRF Polar Angle
        # get matplotlib color map from segmented colors
        PA_cmap = self.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = 256, 
                                                    cmap_name = 'PA_mackey_custom',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['PA'] = self.make_raw_vertex_image(polar_angle_norm, 
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

class MultiViewer(Viewer):

    def __init__(self, pRFModelObj = None, somaModelObj = None, outputdir = None, pysub = 'fsaverage'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        pRFModelObj : soma Model object
            object from one of the classes defined in soma_model
        somaModelObj : soma Model object
            object from one of the classes defined in soma_model
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(pysub = pysub, derivatives_pth = pRFModelObj.MRIObj.derivatives_pth, MRIObj=pRFModelObj.MRIObj)

        # set object to use later on
        self.pRFModelObj = pRFModelObj
        # set object to use later on
        self.somaModelObj = somaModelObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.derivatives_pth, 'plots')
        else:
            self.outputdir = outputdir
    

    def plot_rsq(self, participant_list, fit_type = 'mean_run', prf_model_name = 'gauss', max_ecc_ext = None,
                                mask_arr = True, rsq_threshold = .2, iterative = True):
                                            
        """
        plot flatmap of data for visual vs motor (2D)
        with R2 estimates for subject (or group)

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        fit_type: str
            type of run to fit (mean of all runs, or leave one out) 
        """

        # loop over participant list
        avg_r2_soma = []
        group_estimates = {} # stores estimates for all participants in dict, for ease of access

        for pp in participant_list:

            ## LOAD MOTOR R2 ##
            print('Loading Motor R2')

            if fit_type == 'loo_run':
                # get all run lists
                run_loo_list = self.somaModelObj.get_run_list(self.somaModelObj.get_proc_file_list(pp, file_ext = self.somaModelObj.proc_file_ext))

                ## get average beta values (all used in GLM)
                _, r2 = self.somaModelObj.average_betas(pp, fit_type = fit_type, 
                                                            weighted_avg = True, runs2load = run_loo_list)
            else:
                # load GLM estimates, and get betas and prediction
                r2 = self.somaModelObj.load_GLMestimates(pp, fit_type = fit_type, run_id = None)['r2']

            # append r2
            avg_r2_soma.append(r2[np.newaxis,...])

            ## LOAD pRF R2 ##
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

        # average across group
        avg_r2_soma = np.nanmedian(np.vstack(avg_r2_soma), axis = 0)
        avg_r2_pRF = np.nanmedian(np.stack((group_estimates['sub-{sj}'.format(sj = pp)]['r2'] for pp in participant_list)), axis = 0)

        # replace nans with 0
        avg_r2_soma[np.isnan(avg_r2_soma)] = 0
        avg_r2_pRF[np.isnan(avg_r2_pRF)] = 0

        # set figures path
        figures_pth = op.join(self.outputdir, 'rsq', 'Both', fit_type)

        ## plot flatmap whole suface
        if len(participant_list) == 1: # if one participant
            fig_name = op.join(figures_pth, 'sub-{sj}'.format(sj = participant_list[0]), 
                                    'sub-{sj}_task-Both_pRFmodel-{model}_flatmap_RSQ.png'.format(sj = pp,
                                                                                                model = prf_model_name))                                   
        else:
            fig_name = op.join(figures_pth,
                                 'sub-group_task-Both_pRFmodel-{model}_flatmap_RSQ.png'.format(model = prf_model_name))

        # create custome colormp red blue
        n_bins = 256
        col2D_name = op.splitext(op.split(self.make_2D_colormap(rgb_color='110',bins = n_bins,scale=[1,0.65]))[-1])[0]
        print('created custom colormap %s'%col2D_name)

        ## plot and save fig for whole surface
        flatmap = self.plot_flatmap(avg_r2_pRF, 
                            est_arr2 = avg_r2_soma,
                            cmap = col2D_name, 
                            vmin1 = 0.2, vmax1 = .3, 
                            vmin2 = 0.2, vmax2 = .8, 
                            fig_abs_name = fig_name,
                            figsize=(20,5), dpi=300)

