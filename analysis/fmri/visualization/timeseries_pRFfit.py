################################################
#      Plot pRF fit on V1 vs sPCS, 
#  over timeseries data, for design figure
################################################


import re, os
import glob, yaml
import sys

import numpy as np

from nilearn import surface

sys.path.append(os.path.split(os.getcwd())[0]) # so script finds utils.py
from utils import * #import script to use relevante functions

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder,CSS_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities

import matplotlib.pyplot as plt

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

# file extension
file_extension = params['fitting']['prf']['extension']

# define input and output file dirs
# depending on machine used to fit

fit_where = params['general']['paths']['fitting']['where'] # where are we fitting

deriv_pth = params['general']['paths']['fitting'][fit_where]['derivatives'] # path to derivatives folder
postfmriprep_pth = os.path.join(deriv_pth,'post_fmriprep','prf','sub-{sj}'.format(sj=sj)) # path to post_fmriprep files
median_pth = os.path.join(postfmriprep_pth,'median') # path to median run files

fits_pth = os.path.join(deriv_pth,'pRF_fit','sub-{sj}'.format(sj=sj)) # path to pRF fits
# should add here an "if doesn't exist, fit single voxel", at a later stage.

figures_pth = os.path.join(deriv_pth,'plots','single_vertex','pRFfit','sub-{sj}'.format(sj=sj)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# set threshold for plotting
rsq_threshold = params['plotting']['prf']['rsq_threshold']

# load functional files
# for median run (so we plot over data)

# load data for median run, one hemisphere 
hemi = ['hemi-L','hemi-R']

med_filenames = [os.path.join(median_pth,x) for _,x in enumerate(os.listdir(median_pth)) if 'prf' in x and params['processing']['space'] in x and x.endswith(file_extension)]
data = []
for _,h in enumerate(hemi):
    gii_file = [x for _,x in enumerate(med_filenames) if h in x][0]
    print('loading %s' %gii_file)
    data.append(np.array(surface.load_surf_data(gii_file)))

data = np.vstack(data) # will be (vertex, TR)


# load DM and create pRF stim object

# get screenshot pngs
png_path = os.path.join(params['general']['paths']['analysis'],'fmri','fitting','prf_stimuli')
png_filename = [os.path.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

# set design matrix filename
dm_filename = os.path.join(os.getcwd(), 'prf_dm_square.npy')

# subjects that did pRF task with linux computer, so res was full HD
HD_subs = [str(num).zfill(2) for num in params['general']['HD_screen_subs']] 
res = params['general']['screenRes_HD'] if str(sj).zfill(2) in HD_subs else params['general']['screenRes']

# create design matrix
screenshot2DM(png_filename, 0.1, res, dm_filename, dm_shape = 'square')  # create it

print('computed %s' % (dm_filename))

# actually load DM
prf_dm = np.load(dm_filename,allow_pickle=True)
prf_dm = prf_dm.T # then it'll be (x, y, t)

# shift it to be average of every 2TRs DM
prf_dm = shift_DM(prf_dm)

# crop DM because functional data also cropped 
prf_dm = prf_dm[:,:,params['processing']['crop_pRF_TR']:]

# define model params
TR = params['general']['TR']
hrf = utilities.spm_hrf(0,TR)

fit_model = params['fitting']['prf']['fit_model']

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = (np.array([14,22,25,41,55,71,74,82])-params['processing']['crop_pRF_TR'])*TR


# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['general']['screen_width'],
                         screen_distance_cm = params['general']['screen_distance'],
                         design_matrix = prf_dm,
                         TR = TR)

# and css gridder
if fit_model == 'css':
    gg = CSS_Iso2DGaussianGridder(stimulus = prf_stim,
                                  hrf = hrf,
                                  filter_predictions = True,
                                  window_length = params['processing']['sg_filt_window_length'],
                                  polyorder = params['processing']['sg_filt_polyorder'],
                                  highpass = True,
                                  task_lengths = np.array([prf_dm.shape[-1]]))


else:
	# sets up stimulus and hrf for this gaussian gridder
    gg = Iso2DGaussianGridder(stimulus = prf_stim,
                              hrf = hrf,
                              filter_predictions = True,
                              window_length = params['processing']['sg_filt_window_length'],
                              polyorder = params['processing']['sg_filt_polyorder'],
                              highpass = True,
                              task_lengths = np.array([prf_dm.shape[-1]]))


# Load pRF estimates 
estimates = []
    
for _,field in enumerate(hemi): # each hemi field
    
    # name for 
    gii_file = [x for _,x in enumerate(med_filenames) if field in x][0]
    estimates_combi = os.path.split(gii_file)[-1].replace(params['processing']['extension'],'_'+fit_model+'_estimates.npz')

    estimates_pth = os.path.join(fits_pth,fit_model)

    if os.path.isfile(os.path.join(estimates_pth,estimates_combi)): # if combined estimates exists
        
        print('loading %s'%os.path.join(estimates_pth,estimates_combi))
        estimates.append(np.load(os.path.join(estimates_pth,estimates_combi))) #save both hemisphere estimates in same array

    else: # if not join chunks and save file
        estimates.append(join_chunks(estimates_pth, estimates_combi, field,
                                     chunk_num = total_chunks,fit_model = fit_model))
        
# mask estimates
print('masking estimates')
masked_estimates = mask_estimates(estimates, sj, params, ROI = 'None', fit_model = fit_model)


# plot for two ROIs of interest

ROIs = ['V1','sPCS']
roi_verts = {} #empty dictionary 
for i,val in enumerate(ROIs):   
    if type(val)==str: # if string, we can directly get the ROI vertices  
        roi_verts[val] = cortex.get_roi_verts(params['processing']['space'],val)[val]

red_color = ['#591420','#d12e4c']
data_color = ['#262626','#8a8a8a']

fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)

for idx,roi in enumerate(ROIs):

    new_rsq = masked_estimates['rsq'][roi_verts[roi]]

    new_xx = masked_estimates['x'][roi_verts[roi]]
    new_yy = masked_estimates['y'][roi_verts[roi]]
    new_size = masked_estimates['size'][roi_verts[roi]]

    new_beta = masked_estimates['beta'][roi_verts[roi]]
    new_baseline = masked_estimates['baseline'][roi_verts[roi]]

    complex_location = new_xx + new_yy * 1j
    new_polar_angle = np.angle(complex_location)
    new_ecc = np.abs(complex_location)
    
    new_ns = masked_estimates['ns'][roi_verts[roi]]

    new_data = data[roi_verts[roi]] # data from ROI

    new_index =np.where(new_rsq==np.nanmax(new_rsq))[0][0]# index for max rsq within ROI

    timeseries = new_data[new_index]

    if 'gauss' in fit_model:
        model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                 new_beta[new_index],new_baseline[new_index])
    else: #css
        model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                 new_beta[new_index],new_baseline[new_index],new_ns[new_index])
        
    print('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))

    # plot data with model
    time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
    
    # instantiate a second axes that shares the same x-axis
    if roi == 'sPCS': axis = axis.twinx() 
    
    # plot data with model
    axis.plot(time_sec,model_it_prfpy,c=red_color[idx],lw=3,label=roi+', R$^2$ = %.2f'%new_rsq[new_index],zorder=1)
    axis.scatter(time_sec,timeseries, marker='v',s=15,c=red_color[idx])#,label=roi)
    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.tick_params(axis='both', labelsize=18)
    axis.tick_params(axis='y', labelcolor=red_color[idx], labelsize=18)
    axis.set_xlim(0,len(timeseries)*TR)
    plt.gca().set_ylim(bottom=0)
    
    # to align axis centering it at 0
    if idx == 0:
        if sj=='11':
            axis.set_ylim(-3,9)#6)
        ax1 = axis
    else:
        if sj=='11':
            axis.set_ylim(-1.5,4.5)
        align_yaxis(ax1, 0, axis, 0)


    if idx == 0:
        handles,labels = axis.axes.get_legend_handles_labels()
    else:
        a,b = axis.axes.get_legend_handles_labels()
        handles = handles+a
        labels = labels+b

    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(4):
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor=red_color[idx], alpha=0.1)
        ax_count += 2

    
axis.legend(handles,labels,loc='upper left',fontsize=15)  # doing this to guarantee that legend is how I want it   

fig.savefig(os.path.join(figures_pth,'pRF_singvoxfit_timeseries_%s_rsq-%0.2f.svg'%(str(ROIs),rsq_threshold)), dpi=100,bbox_inches = 'tight')
