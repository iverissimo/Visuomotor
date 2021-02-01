
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

figures_pth = os.path.join(deriv_pth,'plots','rsq_plots','sub-{sj}'.format(sj=sj)) # path to save plots
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


for idx,rois_ks in enumerate(ROIs+['None']):

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
        masked_est = mask_estimates(estimates, s, params, ROI = rois_ks, fit_model = fit_model)

        new_rsq = masked_est['rsq']
        
        # save values in DF
        if idx == 0 and i == 0:
            df_rsq = pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s})
        else:
            df_rsq = df_rsq.append(pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq],'sub': s}))


# do median (for when we are looking at all subs)
for idx,rois_ks in enumerate(ROIs+['None']):   
    rsq_4plot = []

    for i,s in enumerate(sj): # for each subject (if all)

        rsq_4plot.append(list(df_rsq.loc[(df_rsq['roi'] == rois_ks)&(df_rsq['sub'] == s)]['rsq'][0]))

    new_rsq4plot = np.nanmedian(np.array(rsq_4plot), axis = 0)
    if rois_ks == 'None':
        rsq_4flatmap = new_rsq4plot.copy()
    new_rsq4plot = new_rsq4plot[new_rsq4plot >= rsq_threshold] # threshold it
    
    # save values in DF
    if idx == 0:
        df_rsq_median = pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq4plot]})
    else:
        df_rsq_median = df_rsq_median.append(pd.DataFrame({'roi': rois_ks,'rsq': [new_rsq4plot]}))


# plot violin of distribution of RSQ
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

df_visual_plot = df_rsq_median[df_rsq_median['roi'].isin(ROIs)]
df_visual_plot = df_visual_plot.explode('rsq')
df_visual_plot['rsq'] = df_visual_plot['rsq'].astype('float')

v1 = sns.violinplot(data = df_visual_plot, x = 'roi', y = 'rsq',cut=0, inner='box', palette='Set3',linewidth=1.5)

v1.set(xlabel=None)
v1.set(ylabel=None)
#plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.xlabel('ROI',fontsize = 20,labelpad=16)
plt.ylabel('RSQ',fontsize = 20,labelpad=16)
plt.ylim(0,1)

fig.savefig(os.path.join(figures_pth,'rsq_visual_violinplot.svg'), dpi=100)








