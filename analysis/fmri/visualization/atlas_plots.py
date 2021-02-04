
################################################
#       Plot Glasser atlas and Wang atlas
#      over fsaverage surface, to visualize 
#  our data borders/meridians relative to atlas
################################################


import re, os
import glob, yaml
import sys

import numpy as np

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions


import matplotlib.pyplot as plt
import cortex
import nibabel as nb


# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

path2annot_glasser = os.path.join(deriv_pth,'Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG','HCP_PhaseTwo',
                          'Q1-Q6_RelatedParcellation210','MNINonLinear','fsaverage_LR32k')

figures_pth = os.path.join(deriv_pth,'plots','atlas') # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 


## plot Glasser atlas

# get list of absolute path to both files (left and right hemi)
annotfile_dir = [os.path.join(path2annot_glasser,x) for x in os.listdir(path2annot_glasser) if 'HCP-MMP1.annot' in x]
annotfile_dir.sort()


# read annotation file, save:
# labels - annotation id at each vertex.
# ctab - RGBT + label id colortable array.
# names - The names of the labels

lh_labels, lh_ctab, lh_names = nb.freesurfer.io.read_annot(annotfile_dir[0])
rh_labels, rh_ctab, rh_names = nb.freesurfer.io.read_annot(annotfile_dir[-1])


# create 4 arrays with all vertices (left+right hemi)
# for RGB + A (or T actually) 
# correctly selected according to label

# Left hemi
LH_R = []
LH_G = []
LH_B = []
LH_T = []

for _,lbl in enumerate(lh_labels):
    
    LH_R.append(lh_ctab[lbl,0])
    LH_G.append(lh_ctab[lbl,1])
    LH_B.append(lh_ctab[lbl,2])
    LH_T.append(lh_ctab[lbl,3])
    
# Right hemi
RH_R = []
RH_G = []
RH_B = []
RH_T = []

for _,lbl in enumerate(rh_labels):
    
    RH_R.append(rh_ctab[lbl,0])
    RH_G.append(rh_ctab[lbl,1])
    RH_B.append(rh_ctab[lbl,2])
    RH_T.append(rh_ctab[lbl,3])
    

glasser = cortex.VertexRGB(np.concatenate((LH_R,RH_R))/255,
                           np.concatenate((LH_G,RH_G))/255, 
                           np.concatenate((LH_B,RH_B))/255,
                           alpha=(255 - np.concatenate((LH_T,RH_T)))/255,
                           subject='fsaverage_meridians')


cortex.quickshow(glasser,with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(figures_pth,'glasser_flatmap_space-fsaverage_all_ROI.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, glasser, recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(glasser,
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=True,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_glasser_flatmap_space-fsaverage_all_ROI.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, glasser, recache=False,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(glasser,
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=True,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_glasser_flatmap_space-fsaverage_all_ROI.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, glasser, recache=False,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)








