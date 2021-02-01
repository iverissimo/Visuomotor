
################################################
#      Plot VF coverage for each ROI, 
#    diferentiating right and left hemi
################################################


import re, os
import glob, yaml
import sys

import numpy as np

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import cortex

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01) '	
                    'as 1st argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	


# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# gii file extension
file_extension = params['fitting']['prf']['extension']
# fit model used
fit_model = params['fitting']['prf']['fit_model']
# estimates file extensions
estimate_ext = file_extension.replace('.func.gii','')+'_'+fit_model+'_estimates.npz'

# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

figures_pth = os.path.join(deriv_pth,'plots','VF_coverage','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# set threshold for plotting
rsq_threshold = params['plotting']['prf']['rsq_threshold']

# change this to simplify appending all subs and making median plot
if sj == 'all':
    all_sj = params['general']['subs']
    sj = [str(x).zfill(2) for _,x in enumerate(all_sj) if x not in params['general']['exclude_subs']]
else:
    sj = [sj]

# set hemifiled names
hemi = ['hemi-L','hemi-R']

# get vertices for subject fsaverage
ROIs = params['plotting']['prf']['ROIs']

roi_verts = {} #empty dictionary 
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(params['processing']['space'],val)[val]

# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo(params['processing']['space']).left.shape[0] 

for idx,roi in enumerate(ROIs):

	# get roi indices for each hemisphere
    left_roi_verts = roi_verts[roi][roi_verts[roi]<left_index]
    right_roi_verts = roi_verts[roi][roi_verts[roi]>=left_index]

    left_xx_4plot = []
    left_yy_4plot = []

    right_xx_4plot = []
    right_yy_4plot = []

    for i,s in enumerate(sj): # for each subject (if all)

        fits_pth = os.path.join(deriv_pth,'pRF_fit','sub-{sj}'.format(sj = s),fit_model) # path to pRF fits
    
        # absolute path to estimates (combined chunks)
        estimates_combi = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if x.endswith(estimate_ext) and 'chunks' not in x]
        
        # load estimates
        estimates = []

        for _,h in enumerate(hemi): # each hemifield

            est = [x for _,x in enumerate(estimates_combi) if h in x][0]
            print('loading %s'%est)
            estimates.append(np.load(est)) #save both hemisphere estimates in same array
            
        # mask estimates
        print('masking estimates')
        masked_est = mask_estimates(estimates, s, params, ROI = 'None', fit_model = fit_model)

        # LEFT HEMI
        left_xx = masked_est['x'][left_roi_verts]
        left_yy = masked_est['y'][left_roi_verts]
        left_rsq = masked_est['rsq'][left_roi_verts]
        
        left_xx_4plot.append(left_xx[left_rsq>=rsq_threshold]) 
        left_yy_4plot.append(left_yy[left_rsq>=rsq_threshold]) 

        # RIGHT HEMI
        right_xx = masked_est['x'][right_roi_verts]
        right_yy = masked_est['y'][right_roi_verts]
        right_rsq = masked_est['rsq'][right_roi_verts] 
        
        right_xx_4plot.append(right_xx[right_rsq>=rsq_threshold]) 
        right_yy_4plot.append(right_yy[right_rsq>=rsq_threshold]) 

    # set screen limits for plotting
    res = params['general']['screenRes']
    vert_lim_dva = (res[-1]/2) * dva_per_pix(params['general']['screen_width'],params['general']['screen_distance'],res[0])
    hor_lim_dva = (res[0]/2) * dva_per_pix(params['general']['screen_width'],params['general']['screen_distance'],res[0])

    # actually plot hexabins
    f, ss = plt.subplots(1, 1, figsize=(16,9))#figsize=[2*x for x in plt.rcParams["figure.figsize"]], sharey=True)

    ss.hexbin(np.hstack(left_xx_4plot), 
              np.hstack(left_yy_4plot),
              gridsize=15, 
              cmap='Greens',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),#
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=1)

    ss.hexbin(np.hstack(right_xx_4plot), 
              np.hstack(right_yy_4plot),
              gridsize=15, 
              cmap='Reds', #'YlOrRd_r',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),#
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=0.5)

    plt.xticks(fontsize = 32)
    plt.yticks(fontsize = 32)
    plt.tight_layout()
    plt.ylim(-vert_lim_dva, vert_lim_dva) #-6,6)#
    ss.set_aspect('auto')
    # set middle lines
    ss.axvline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
    ss.axhline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')

    # custom lines only to make labels
    custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                    Line2D([0], [0], color='r',alpha=0.5, lw=4)]

    plt.legend(custom_lines, ['LH', 'RH'],fontsize = 35)

    fig_hex = plt.gcf()
    fig_hex.savefig(os.path.join(figures_pth,'VF_coverage_ROI-%s_hemi-both.svg'%roi),dpi=100)


