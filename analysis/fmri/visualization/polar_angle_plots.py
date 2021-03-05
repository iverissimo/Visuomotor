
################################################
#          Plot polar angle flatmaps 
################################################


import re, os
import glob, yaml
import sys

import numpy as np

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions


import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns

from matplotlib import colors

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01)  or "median" '	
                    'as 1st argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	


# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# gii file extension, for both tasks
file_extension = {'prf': params['fitting']['prf']['extension']}
# fit model used
fit_model = params['fitting']['prf']['fit_model']

# estimates file extensions
estimate_prf_ext = file_extension['prf'].replace('.func.gii','')+'_'+fit_model+'_estimates.npz'

# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

figures_pth = os.path.join(deriv_pth,'plots','polar_angle','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# set threshold for plotting
rsq_threshold = params['plotting']['prf']['rsq_threshold']

# change this to simplify appending all subs and making median plot
if sj == 'median':
    all_sj = params['general']['subs']
    sj = [str(x).zfill(2) for _,x in enumerate(all_sj) if x not in params['general']['exclude_subs']]
else:
    sj = [sj]

# set hemifiled names
hemi = ['hemi-L','hemi-R']


# to use later in flatmaps
estimates_dict_smooth = {'rsq': [], 'x': [], 'y': []}

for i,s in enumerate(sj): # for each subject (if all)

    fits_pth = os.path.join(deriv_pth,'pRF_fit','sub-{sj}'.format(sj = s),fit_model) # path to pRF fits

    # absolute path to estimates (combined chunks)
    estimates_combi = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if x.endswith(estimate_prf_ext) and 'chunks' not in x]

    # load estimates
    estimates = []

    for _,h in enumerate(hemi): # each hemifield

        est = [x for _,x in enumerate(estimates_combi) if h in x][0]
        print('loading %s'%est)
        estimates.append(np.load(est)) #save both hemisphere estimates in same array

    # mask estimates
    print('masking estimates')
    masked_est = mask_estimates(estimates, s, params, ROI = 'None', fit_model = fit_model)

    # save some estimates in dict
    estimates_dict = {'rsq': masked_est['rsq'],
                      'x': masked_est['x'],
                      'y': masked_est['y']}
    
    ## now smooth estimates and save
    # now smooth estimates and save
    # path to save compute estimates
    out_estimates_dir = os.path.join(deriv_pth,'estimates','prf','sub-{sj}'.format(sj = s))
    if not os.path.exists(out_estimates_dir):
        os.makedirs(out_estimates_dir) 

    # get path to post_fmriprep files, for header info 
    post_proc_gii_pth = os.path.join(deriv_pth,'post_fmriprep','prf','sub-{sj}'.format(sj = s), 'median')
    post_proc_gii = [os.path.join(post_proc_gii_pth,x) for _,x in enumerate(os.listdir(post_proc_gii_pth)) if params['processing']['space'] in x and x.endswith(file_extension['prf'])]
    post_proc_gii.sort()

    ## smooth for flatmaps
    for _,est in enumerate(estimates_dict.keys()):

        smooth_file = os.path.split(post_proc_gii[0])[-1].replace('hemi-L','hemi-both').replace('.func.gii','_%s_smooth%d.npy'%(est,params['processing']['smooth_fwhm']))
        smooth_file = os.path.join(out_estimates_dir,smooth_file)

        if os.path.isfile(smooth_file): # if smooth file exists
            print('loading %s'%smooth_file)
            estimates_dict[est] = np.load(smooth_file) # load                      

        else: # smooth array
                                   
            estimates_dict[est] = smooth_nparray(estimates_dict[est], 
                                                   post_proc_gii, 
                                                   out_estimates_dir, 
                                                   '_%s'%est, 
                                                   sub_space = params['processing']['space'], 
                                                   n_TR = params['plotting']['prf']['n_TR'], 
                                                   smooth_fwhm = params['processing']['smooth_fwhm'],
                                                   sub_ID = s)

        # append smooth estimates
        estimates_dict_smooth[est].append(estimates_dict[est])

 
# now do median estimates (for when we are looking at all subs)

xx_median = np.nanmedian(np.array(estimates_dict_smooth['x']), axis = 0)
yy_median = np.nanmedian(np.array(estimates_dict_smooth['y']), axis = 0)
rsq_median = np.nanmedian(np.array(estimates_dict_smooth['rsq']), axis = 0)

# need to mask again because smoothing removes nans
rsq_median = mask_arr(rsq_median, threshold = rsq_threshold, side = 'above')
xx_median[np.isnan(rsq_median)] = np.nan
yy_median[np.isnan(rsq_median)] = np.nan

# compute eccentricity
complex_location = xx_median + yy_median * 1j
masked_polar_angle = np.angle(complex_location)


images = {}

# normalize polar angles to have values in circle between 0 and 1
masked_polar_ang_norm = (masked_polar_angle + np.pi) / (np.pi * 2.0)

# normalize the distribution, for better visualization
rsq_norm = normalize(np.clip(rsq_median,rsq_threshold,.3)) 

# make costum colormap, similar to curtis mackey paper
# orange to red, counter clockwise
n_bins = 8
PA_colors = add_alpha2colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'],bins = n_bins, cmap_name = 'PA_mackey_costum',
                              discrete = True)

# create costume colormp rainbow_r
col2D_name = os.path.splitext(os.path.split(PA_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

images['PA'] = cortex.Vertex2D(masked_polar_ang_norm,rsq_norm, 
                                        subject = 'fsaverage_meridians', #params['processing']['space'], #'fsaverage_meridians',
                                        vmin = 0, vmax = 1,
                                        vmin2 = 0, vmax2 = 1,
                                        cmap = col2D_name)

cortex.quickshow(images['PA'],with_curvature=True,with_sulci=True,with_colorbar=True,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = os.path.join(figures_pth,'flatmap_space-fsaverage_rsq-%0.2f_type-PA_mackey_colorwheel.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True,
                                curvature_brightness = 0.4, curvature_contrast = 0.1)

# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['PA'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_space-fsaverage_type-PA_mackey_colorwheel.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA'], recache=False,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                curvature_brightness = 0.4, curvature_contrast = 0.1)

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['PA'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_space-fsaverage_type-PA_mackey_colorwheel.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA'], recache=False,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048,
                                curvature_brightness = 0.4, curvature_contrast = 0.1)



# # plot colorwheel and save in folder
# import matplotlib as mpl

# resolution = 800
# x, y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
# radius = np.sqrt(x**2 + y**2)
# polar_angle = np.arctan2(y, x)

# polar_angle_circle = polar_angle.copy() # all polar angles calculated from our mesh
# polar_angle_circle[radius > 1] = np.nan # then we're excluding all parts of bitmap outside of circle

# # normal color wheel

# # make linear range of colors
# colormap = colors.ListedColormap(['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'])
# boundaries = np.linspace(0,1,8)
# norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)

# # normalize between the point where we defined our color threshold
# norm = mpl.colors.Normalize(-np.pi, np.pi) 

# plt.imshow(polar_angle_circle, cmap = colormap, norm=norm, origin='lower')
# plt.axis('off')
# plt.savefig(os.path.join(os.path.split(figures_pth)[0],'color_wheel_mackey.svg'),dpi=100)







