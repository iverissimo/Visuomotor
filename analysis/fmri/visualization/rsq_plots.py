
################################################
#      Plot RSQ distribution for prf and soma, 
#    	with some violinplots and flatmaps
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
file_extension = {'prf': params['fitting']['prf']['extension'],
                 'soma': params['fitting']['soma']['extension']}
# fit model used
fit_model = params['fitting']['prf']['fit_model']

# estimates file extensions
estimate_prf_ext = file_extension['prf'].replace('.func.gii','')+'_'+fit_model+'_estimates.npz'
estimate_soma_ext = file_extension['soma'].replace('.func.gii','')+'_estimates.npz'

# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

figures_pth = os.path.join(deriv_pth,'plots','rsq_plots','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# set threshold for plotting
rsq_threshold = params['plotting']['prf']['rsq_threshold']
z_threshold = params['plotting']['soma']['z_threshold']

# change this to simplify appending all subs and making median plot
if sj == 'median':
    all_sj = params['general']['subs']
    sj = [str(x).zfill(2) for _,x in enumerate(all_sj) if x not in params['general']['exclude_subs']]
else:
    sj = [sj]

# set hemifiled names
hemi = ['hemi-L','hemi-R']

#### visual task ####

# get vertices for subject fsaverage
ROIs = params['plotting']['prf']['ROIs']

roi_verts = {} #empty dictionary  
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(params['processing']['space'],val)[val]


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

    #loop for all ROIs
    for idx,rois_ks in enumerate(ROIs+['None']):
        
        # mask estimates
        print('masking estimates for ROI %s'%rois_ks)
        masked_est = mask_estimates(estimates, s, params, ROI = rois_ks, fit_model = fit_model)

        new_rsq = masked_est['rsq']

        # save values in DF
        if (idx == 0) and (i == 0):
            df_rsq_visual = pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s})
        else:
            df_rsq_visual = df_rsq_visual.append(pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s}))


# ravel for violin plot 
# (we want to include all voxels of all subs to see distribution)
for idx,rois_ks in enumerate(ROIs): 

    if len(sj)>1: 
        rsq_4plot = np.concatenate(list(df_rsq_visual.loc[(df_rsq_visual['roi'] == rois_ks)]['rsq'][0])).ravel()
    else:
        rsq_4plot = df_rsq_visual.loc[(df_rsq_visual['roi'] == rois_ks)]['rsq'][0]
    
    # threshold it
    rsq_4plot = rsq_4plot[rsq_4plot >= rsq_threshold] 
    
    # save values in DF
    if idx == 0:
        df_rsq_4plot = pd.DataFrame({'roi': rois_ks,'rsq': [rsq_4plot]})
    else:
        df_rsq_4plot = df_rsq_4plot.append(pd.DataFrame({'roi': rois_ks,'rsq': [rsq_4plot]}))


# plot violin of distribution of RSQ

# Make a dictionary with one specific color per group - similar to fig3 colors
ROI_pal = {'V1': (0.03137255, 0.11372549, 0.34509804), 'V2': (0.14136101, 0.25623991, 0.60530565),
           'V3': (0.12026144, 0.50196078, 0.72156863), 'V3AB': (0.25871588, 0.71514033, 0.76807382), 
           'hV4': (0.59215686, 0.84052288, 0.72418301), 'LO': (0.88207612, 0.9538639 , 0.69785467),
           'IPS0': (0.99764706, 0.88235294, 0.52862745), 'IPS1': (0.99529412, 0.66901961, 0.2854902), 
           'IPS2+': (0.83058824, 0.06117647, 0.1254902),
           'sPCS': (0.88221453, 0.83252595, 0.91109573), 'iPCS': (0.87320261, 0.13071895, 0.47320261)
         }


fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

df_visual_plot = df_rsq_4plot.explode('rsq')
df_visual_plot['rsq'] = df_visual_plot['rsq'].astype('float')

v1 = sns.violinplot(data = df_visual_plot, x = 'roi', y = 'rsq', 
                    cut=0, inner='box', palette = ROI_pal,linewidth=1.8) # palette ='Set3',linewidth=1.8)

v1.set(xlabel=None)
v1.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('ROI',fontsize = 20,labelpad=18)
plt.ylabel('RSQ',fontsize = 20,labelpad=18)
plt.ylim(0,1)

fig.savefig(os.path.join(figures_pth,'rsq_visual_violinplot.svg'), dpi=100)


## now do same for motor task

ROIs = list(params['fitting']['soma']['all_contrasts'].keys()) # list of key names (of different body regions)

for i,s in enumerate(sj): # for each subject (if all)

    fits_pth = os.path.join(deriv_pth,'soma_fit','sub-{sj}'.format(sj = s)) # path to pRF fits
    
    for idx,rois_ks in enumerate(['None']+ROIs):

        if rois_ks == 'None': # save rsq of all vertices

            estimates_filename = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if 'hemi-both' in x and x.endswith(estimate_soma_ext)][0]
            estimates = np.load(estimates_filename,allow_pickle = True)
            all_rsq = estimates['r2']
            new_rsq = all_rsq.copy()

        else:
            zmask_filename = os.path.join(fits_pth,'zscore_thresh-%.2f_%s_vs_all_contrast.npy' %(z_threshold,rois_ks))
            zmask = np.load(zmask_filename, allow_pickle = True)

            # mask rsq - only significant voxels for region
            new_rsq = all_rsq.copy()
            new_rsq[np.isnan(zmask)] = np.nan

        # save values in DF
        if idx == 0 and i == 0:
            df_rsq_soma = pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s})
        else:
            df_rsq_soma = df_rsq_soma.append(pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s}))


# ravel for violin plot 
# (we want to include all voxels of all subs to see distribution)
for idx,rois_ks in enumerate(ROIs): 

    if len(sj)>1: 
        rsq_4plot = p.concatenate(list(df_rsq_soma.loc[(df_rsq_soma['roi'] == rois_ks)]['rsq'][0])).ravel()
    else:
        rsq_4plot = df_rsq_soma.loc[(df_rsq_soma['roi'] == rois_ks)]['rsq'][0]
    
    # threshold it
    rsq_4plot = rsq_4plot[rsq_4plot >= rsq_threshold] 
    
    # save values in DF
    if idx == 0:
        df_rsq_4plot = pd.DataFrame({'roi': rois_ks,'rsq': [rsq_4plot]})
    else:
        df_rsq_4plot = df_rsq_4plot.append(pd.DataFrame({'roi': rois_ks,'rsq': [rsq_4plot]}))


# plot violin of distribution of RSQ
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

df_soma_plot = df_rsq_4plot.explode('rsq')
df_soma_plot['rsq'] = df_soma_plot['rsq'].astype('float')

v1 = sns.violinplot(data = df_soma_plot, x = 'roi', y = 'rsq',
                    cut=0, inner='box', palette=['#2367b0','#23b086','#c72828'],linewidth=1.8)

v1.set(xlabel=None)
v1.set(ylabel=None)
v1.set_xticklabels(['Face','Hands','Feet'])
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('ROI',fontsize = 20,labelpad=18)
plt.ylabel('RSQ',fontsize = 20,labelpad=18)
plt.ylim(0,1)

fig.savefig(os.path.join(figures_pth,'rsq_soma_violinplot.svg'), dpi=100)


# now smooth RSQ, to plot in flatmap

tasks = params['general']['tasks']

rsq_median = {'prf': [],
              'soma': []}

df_rsq_median = {'prf': df_rsq_visual.loc[(df_rsq_visual['roi'] == 'None')],
              'soma': df_rsq_soma.loc[(df_rsq_soma['roi'] == 'None')]}

for _,tsk in enumerate(tasks):

    for i,s in enumerate(sj): # for each subject (if all)
        
        rsq_median[tsk].append(list(df_rsq_median[tsk].loc[(df_rsq_median[tsk]['sub'] == s)]['rsq'][0])) 
    
    # do median (for when we are looking at all subs)   
    rsq_median[tsk] = np.nanmedian(np.array(rsq_median[tsk]), axis = 0)
    
    # path to save compute estimates
    out_estimates_dir = os.path.join(deriv_pth,'estimates',tsk,'sub-{sj}'.format(sj = str(sys.argv[1]).zfill(2)))
    if not os.path.exists(out_estimates_dir):
        os.makedirs(out_estimates_dir) 

    # get path to post_fmriprep files, for header info 
    post_proc_gii_pth = os.path.join(deriv_pth,'post_fmriprep', tsk,'sub-{sj}'.format(sj = sj[0]), 'median')
    post_proc_gii = [os.path.join(post_proc_gii_pth,x) for _,x in enumerate(os.listdir(post_proc_gii_pth)) if params['processing']['space'] in x and x.endswith(file_extension[tsk])]
    post_proc_gii.sort()

    smooth_file = os.path.split(post_proc_gii[0])[-1].replace('hemi-L','hemi-both').replace('.func.gii','_rsq_smooth%d.npy'%params['processing']['smooth_fwhm'])
    smooth_file = os.path.join(out_estimates_dir,smooth_file).replace('sub-{sj}'.format(sj = sj[0]),'sub-{sj}'.format(sj = str(sys.argv[1]).zfill(2)))

    if os.path.isfile(smooth_file): # if smooth file exists
        print('loading %s'%smooth_file)
        rsq_smooth = np.load(smooth_file) # load

    else: # smooth rsq

        rsq_smooth = smooth_nparray(rsq_median[tsk], 
                                   post_proc_gii, 
                                   out_estimates_dir, 
                                   '_rsq', 
                                   sub_space = params['processing']['space'], 
                                   n_TR = params['plotting'][tsk]['n_TR'], 
                                   smooth_fwhm = params['processing']['smooth_fwhm'],
                                   sub_ID = str(sys.argv[1]).zfill(2))

    rsq_median[tsk] = rsq_smooth


# do this to replace nans with 0s, so flatmaps look nicer
rsq_visual_smooth = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_median['prf'])])
rsq_soma_smooth = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_median['soma'])])


# normalize RSQ 
rsq_visual_smooth_norm = normalize(rsq_visual_smooth) 
rsq_soma_smooth_norm = normalize(rsq_soma_smooth) 

# create costume colormp red blue
n_bins = 256
col2D_name = os.path.splitext(os.path.split(make_2D_colormap(rgb_color='101',bins = n_bins))[-1])[0]
print('created costum colormap %s'%col2D_name)


# make flatmaps of the above distributions
print('making flatmaps')

images = {}

images['rsq_visual_norm'] = cortex.Vertex(rsq_visual_smooth_norm, 
                                          params['processing']['space'],
                                           vmin = 0.125, vmax = 0.2,
                                           cmap='Reds')
#cortex.quickshow(images['rsq_visual_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_type-rsquared-normalized_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_visual_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

##### add flatmap to subject overlay
#cortex.utils.add_roi(images['rsq_visual_norm'], name='sub-median_rsq_visual_norm', open_inkscape=False)


images['rsq_soma_norm'] = cortex.Vertex(rsq_soma_smooth_norm, 
                                          params['processing']['space'],
                                           vmin = 0.2, vmax = .6,
                                           cmap='Blues')
#cortex.quickshow(images['rsq_soma_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_type-rsquared-normalized_soma.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_soma_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

##### add flatmap to subject overlay
#cortex.utils.add_roi(images['rsq_soma_norm'], name='sub-median_rsq_soma_norm', open_inkscape=False)


images['rsq_combined'] = cortex.Vertex2D(rsq_visual_smooth_norm,rsq_soma_smooth_norm, 
                                        subject = 'fsaverage_meridians', #params['processing']['space'],
                                        vmin = 0.125, vmax = 0.2,
                                        vmin2 = 0.2,vmax2 = 0.6,
                                        cmap = col2D_name)
#cortex.quickshow(images['rsq_combined'],recache=True,with_curvature=True,with_sulci=True,with_roi=False,with_labels=False)#,height=2048)
filename = os.path.join(figures_pth,'flatmap_space-fsaverage_type-rsquared-normalized_multimodal.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,with_labels=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figures_pth,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,with_labels=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# save inflated 3D screenshots 
figures_pth = os.path.join(figures_pth,'3d_views_screenshots')
if not os.path.exists(figures_pth):  # check if path exists
        os.makedirs(figures_pth)

cortex.export.save_3d_views(images['rsq_combined'], 
             base_name=os.path.join(figures_pth,'sub-%s'%(str(sys.argv[1]).zfill(2))), 
             list_angles=['lateral_pivot', 'medial_pivot', 'left', 'right', 'top', 'bottom',
                         'left'],
                  list_surfaces=['inflated', 'inflated', 'inflated', 'inflated','inflated','inflated',
                                'flatmap'],
                  viewer_params=dict(labels_visible=[],
                                     overlays_visible=['rois','sulci']),
                  size=(1024 * 4, 768 * 4), trim=True, sleep=60)






