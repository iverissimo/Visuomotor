
################################################
#      Do graphs with ECC vs Size, 
#    relationship per subject (or for all combined) 
################################################

import re, os
import glob, yaml
import sys

import numpy as np

from nilearn import surface

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

import matplotlib.pyplot as plt
import cortex
from statsmodels.stats import weightstats
import pandas as pd
import seaborn as sns

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01 or "all") '
                    'as 1st argument in the command line!')
else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets


# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# gii file extension
file_extension = params['fitting']['prf']['extension']
# fit model used
fit_model = params['fitting']['prf']['fit_model']
# estimates file extensions
estimate_ext = file_extension.replace('.func.gii','')+'_'+fit_model+'_estimates.npz'

# paths
fit_where = params['general']['paths']['fitting']['where'] # where we are working
deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder

figures_pth = os.path.join(deriv_pth,'plots','ecc_size','sub-{sj}'.format(sj = sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

    
# change this to simplify appending all subs and making median plot
if sj == 'all':
    all_sj = params['general']['subs']
    sj = [str(x).zfill(2) for _,x in enumerate(all_sj)] #if x not in params['general']['exclude_subs']]
else:
    sj = [sj]


# get vertices for subject fsaverage
ROIs = params['plotting']['prf']['ROIs']

roi_verts = {} #empty dictionary 
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(params['processing']['space'],val)[val]


# create empty dataframe to store all relevant values for rois
all_roi = [] 
n_bins = params['plotting']['prf']['n_bins']
min_ecc = params['plotting']['prf']['min_ecc']
max_ecc = params['plotting']['prf']['max_ecc']
rsq_threshold = params['plotting']['prf']['rsq_threshold']

for i,s in enumerate(sj):

    fits_pth = os.path.join(deriv_pth,'pRF_fit','sub-{sj}'.format(sj = s),fit_model) # path to pRF fits
    
    # absolute path to estimates (combined chunks)
    estimates_combi = [os.path.join(fits_pth,x) for _,x in enumerate(os.listdir(fits_pth)) if x.endswith(estimate_ext) and 'chunks' not in x]
    
    # load estimates
    estimates = []
    hemi = ['hemi-L','hemi-R']

    for _,h in enumerate(hemi): # each hemifield

        est = [x for _,x in enumerate(estimates_combi) if h in x][0]
        print('loading %s'%est)
        estimates.append(np.load(est)) #save both hemisphere estimates in same array
        
    # mask estimates
    print('masking estimates')
    masked_est = mask_estimates(estimates, s, params, ROI = 'None', fit_model = fit_model)
    
    
    for idx,roi in enumerate(ROIs): # go over ROIs
        
        # get datapoints for RF only belonging to roi
        new_size = masked_est['size'][roi_verts[roi]]

        complex_location = masked_est['x'][roi_verts[roi]] + masked_est['y'][roi_verts[roi]] * 1j # calculate eccentricity values
        new_ecc = np.abs(complex_location)
        
        new_rsq = masked_est['rsq'][roi_verts[roi]]
        
        # define indices of voxels within region to plot
        # with rsq > 0.17, and where value not nan, ecc values between 0.25 and 3.3
        indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq >= rsq_threshold) & (np.logical_not(np.isnan(new_size))))[0]

        df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size': new_size[indices4plot],
                            'rsq': new_rsq[indices4plot],'sub': np.tile(s,len(indices4plot))})
        
        # sort values by eccentricity
        df = df.sort_values(by=['ecc'])  

        #divide in equally sized bins
        bin_size = int(len(df)/n_bins) 
        mean_ecc = []
        mean_ecc_std = []
        mean_size = []
        mean_size_std = []
        
        # for each bin calculate rsq-weighted means and errors of binned ecc/size 
        for j in range(n_bins): 
            mean_size.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
            mean_size_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)
            mean_ecc.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
            mean_ecc_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)

        if idx == 0 and i == 0:
            all_roi = pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std': mean_ecc_std,
                                    'mean_size': mean_size,'mean_size_std': mean_size_std,
                                    'ROI':np.tile(roi,n_bins),'sub':np.tile(s,n_bins)})
        else:
            all_roi = all_roi.append(pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std': mean_ecc_std,
                                                   'mean_size': mean_size,'mean_size_std': mean_size_std,
                                                   'ROI': np.tile(roi,n_bins),'sub': np.tile(s,n_bins)}),ignore_index=True)
    
        
    
# get median bins for plotting 
# (useful for all subject plot)

med_subs_df = []

for idx,roi in enumerate(ROIs):
      
    for j in range(n_bins):
        med_ecc = []
        med_ecc_std = []
        med_size = []
        med_size_std = []

        for _,w in enumerate(sj):
            
            med_ecc.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['ROI'] == roi)]['mean_ecc'].iloc[j])
            med_ecc_std.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['ROI'] == roi)]['mean_ecc_std'].iloc[j])
            med_size.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['ROI'] == roi)]['mean_size'].iloc[j])
            med_size_std.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['ROI'] == roi)]['mean_size_std'].iloc[j])

        if idx == 0 and j == 0:
            med_subs_df = pd.DataFrame({'med_ecc': [np.nanmedian(med_ecc)],'med_ecc_std':[np.nanmedian(med_ecc_std)],
                                    'med_size':[np.nanmedian(med_size)],'med_size_std':[np.nanmedian(med_size_std)],
                                    'ROI':[roi]})
        else:
            med_subs_df = med_subs_df.append(pd.DataFrame({'med_ecc': [np.nanmedian(med_ecc)],'med_ecc_std':[np.nanmedian(med_ecc_std)],
                                                   'med_size':[np.nanmedian(med_size)],'med_size_std':[np.nanmedian(med_size_std)],
                                                   'ROI':[roi]}),ignore_index=True)


### plot for Occipital Areas - V1 V2 V3 V3AB hV4 LO ###

roi2plot = params['plotting']['prf']['occipital']

sns.set(font_scale=1.3)
sns.set_style("ticks")

ax = sns.lmplot(x = 'med_ecc', y = 'med_size', hue = 'ROI', data = med_subs_df[med_subs_df.ROI.isin(roi2plot)],
                scatter=True, palette="YlGnBu_r",markers=['^','s','o','v','D','h'])

ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0,5)

ax.set_xlabel('pRF eccentricity [dva]', fontsize = 18, labelpad = 15)
ax.set_ylabel('pRF size [dva]', fontsize = 18, labelpad = 15)
#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
fig1 = plt.gcf()
fig1.savefig(os.path.join(figures_pth,'%s_ecc_vs_size_binned_rsq-%0.2f.svg'%('occipital',rsq_threshold)), dpi=100,bbox_inches = 'tight')


### plot for Parietal Areas - IPS0 IPS1 IPS2+ ###

roi2plot = params['plotting']['prf']['parietal']

sns.set(font_scale=1.3)
sns.set_style("ticks")

ax = sns.lmplot(x = 'med_ecc', y = 'med_size', hue ='ROI',data = med_subs_df[med_subs_df.ROI.isin(roi2plot)],
                scatter=True, palette="YlOrRd",markers=['^','s','o'])
ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0,5)

if sj != 'all':
    ax.set_xlabel('pRF eccentricity [dva]', fontsize = 18, labelpad = 15)
    ax.set_ylabel('pRF size [dva]', fontsize = 18, labelpad = 15)

#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
fig1 = plt.gcf()
fig1.savefig(os.path.join(figures_pth,'%s_ecc_vs_size_binned_rsq-%0.2f.svg'%(str(roi2plot),rsq_threshold)), dpi=100,bbox_inches = 'tight')
 

### plot for Frontal Areas - sPCS iPCS ###

roi2plot = params['plotting']['prf']['frontal']

sns.set(font_scale=1.3)
sns.set_style("ticks")

ax = sns.lmplot(x = 'med_ecc', y = 'med_size', hue = 'ROI', data = med_subs_df[med_subs_df.ROI.isin(roi2plot)],
                scatter=True, palette="PuRd",markers=['^','s'])
ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0,5)

if sj != 'all':
    ax.set_xlabel('pRF eccentricity [dva]', fontsize = 18, labelpad = 15)
    ax.set_ylabel('pRF size [dva]', fontsize = 18, labelpad = 15)

#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
fig1 = plt.gcf()
fig1.savefig(os.path.join(figures_pth,'%s_ecc_vs_size_binned_rsq-%0.2f.svg'%(str(roi2plot),rsq_threshold)), dpi=100,bbox_inches = 'tight')






