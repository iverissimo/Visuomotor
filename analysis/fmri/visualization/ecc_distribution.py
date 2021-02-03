
################################################
#      Plot ecc distributions as shown by, 
# pairs of ECD curves and KS statistic for each ROI
################################################


import re, os
import glob, yaml
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

import matplotlib.pyplot as plt

import cortex
import scipy

import seaborn as sns

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01) or "median" '	
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

figures_pth = os.path.join(deriv_pth,'plots','ecc_distribution','sub-{sj}'.format(sj=sj)) # path to save plots
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

# get vertices for subject fsaverage
ROIs = params['plotting']['prf']['ROIs']

roi_verts = {} #empty dictionary 
ecc_roi_all = {} #empty dictionary  
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(params['processing']['space'],val)[val]
    ecc_roi_all[val] = [] # to be filled later


# plot empirical cumulative distribution of ecc for all ROIs in same plot

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

for _,rois_ks in enumerate(ROIs):
    
    ecc_4plot = []

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
        complex_location = masked_est['x'] + masked_est['y'] * 1j
        new_ecc = np.abs(complex_location)

        indices4plot = np.where((new_rsq>=rsq_threshold) & (np.logical_not(np.isnan(new_ecc))))
        
        ecc_4plot.append(np.sort(new_ecc[indices4plot]))


    ecc_roi_all[rois_ks] = np.sort(np.hstack(ecc_4plot))

    n1 = np.arange(1,len(ecc_roi_all[rois_ks])+1) / np.float(len(ecc_roi_all[rois_ks]))
    ax.step(ecc_roi_all[rois_ks],n1,label = rois_ks)

    ax.legend()

    ax.set_title('Empirical cumulative distribution')
    ax.set_xlabel('eccentricity [dva]')
    ax.set_ylabel('Cumulative probability')
    ax.set_xlim([0,params['fitting']['prf']['max_eccen']])
    ax.set_ylim([0,1])

plt.savefig(os.path.join(figures_pth,'ECDF_ROI-all.svg'),dpi=100)


# plot Kernel Density Estimate for eccs

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

for _,rois_ks in enumerate(ROIs):
    
    
    ax = sns.kdeplot(ecc_roi_all[rois_ks], label = rois_ks)
    ax.set_title('Kernel density')
    ax.set_xlabel('eccentricity [dva]')
    ax.set_ylabel('Kernel Density Estimate')
    ax.set_xlim([0,params['fitting']['prf']['max_eccen']])
    
    ax.legend()

plt.savefig(os.path.join(figures_pth,'KDE_ROI-all.svg'),dpi=100)

## Make distribuition difference matrix to compare all ROIs ###

# compute KS statistic for each ROI of each participant (in case of sj 'median')
# and append in dict

KS_stats = {}

for _,rois_ks in enumerate(ROIs): # compare ROI (ex V1)
    
    roi_stats = []
    
    for _,cmp_roi in enumerate(ROIs): # to all other ROIS
        
        sub_roi_stats = []

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

            new_rsq1 = masked_est['rsq'][roi_verts[rois_ks]]
            complex_location = masked_est['x'] + masked_est['y'] * 1j
            new_ecc1 = np.abs(complex_location)[roi_verts[rois_ks]]

            indices4plot_1 = np.where((new_rsq1>=rsq_threshold) & (np.logical_not(np.isnan(new_ecc1))))

            ecc_roi1 = np.sort(new_ecc1[indices4plot_1])

            new_rsq2 = masked_est['rsq'][roi_verts[cmp_roi]]
            new_ecc2 = np.abs(complex_location)[roi_verts[cmp_roi]]

            indices4plot_2 = np.where((new_rsq2>=rsq_threshold) & (np.logical_not(np.isnan(new_ecc2))))

            ecc_roi2 = np.sort(new_ecc2[indices4plot_2])

            sub_roi_stats.append(scipy.stats.ks_2samp(ecc_roi1, ecc_roi2)[0])


        roi_stats.append(np.median(sub_roi_stats)) # median max distance 
        
    KS_stats[rois_ks] = roi_stats


#Create DataFrame
DF_var = pd.DataFrame.from_dict(KS_stats).T
DF_var.columns = ROIs

# mask out repetitive values, making triangular matrix (still has identity diag)
for i,region in enumerate(ROIs):
    if i>0:
        for k in range(i):
            DF_var[region][k] = np.nan


# plot eccentricity difference matrix
fig, ax = plt.subplots(1, 1, figsize=[2.5*x for x in plt.rcParams["figure.figsize"]], sharey=True)

matrix = ax.matshow(DF_var,cmap='OrRd')
plt.xticks(range(DF_var.shape[1]), DF_var.columns, fontsize=18)#, rotation=45)
plt.yticks(range(DF_var.shape[1]), DF_var.columns, fontsize=18)
cbar = fig.colorbar(matrix)
matrix.set_clim(vmin=0,vmax=0.35)
cbar.set_label('KS statistic', rotation=270, fontsize=30, labelpad=50)
cbar.ax.tick_params(labelsize=18)

# This is very hack-ish, but works to make grid
plt.gca().set_xticks([x - 0.51 for x in plt.gca().get_xticks()][1:], minor='true')
plt.gca().set_yticks([y - 0.52 for y in plt.gca().get_yticks()][1:], minor='true')
plt.grid(which='minor')

fig.savefig(os.path.join(figures_pth,'RSA_ROI-all.svg'),dpi=100)




