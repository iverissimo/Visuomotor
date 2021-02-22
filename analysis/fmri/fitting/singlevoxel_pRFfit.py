
################################################
#      Do pRF fit on single voxel, 
#    by loading estimates, getting fit and 
#    saving plot of fit on timeseries
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

elif len(sys.argv)<3:   
    raise NameError('Please add ROI name (ex: V1) or "None" if looking at vertex from no specific ROI  '	
                    'as 2nd argument in the command line!')	

elif len(sys.argv)<4:   
    raise NameError('Please vertex index number of that ROI (or from whole brain)'	
                    'as 3rd argument in the command line!'
                    '(can also be "max" or "min" to fit vertex of max or min RSQ)')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    roi = str(sys.argv[2]) # ROI or 'None'

    if str(sys.argv[3]) != 'max' and str(sys.argv[3]) != 'min': # if we actually get a number for the vertex
    
        vertex = int(sys.argv[3]) # vertex number

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'


# number of chunks that data was split in
total_chunks = params['fitting']['prf']['total_chunks']

# use smoothed data?
fit_smooth = params['fitting']['prf']['fit_smooth']

# file extension
file_extension = params['fitting']['prf']['extension']
if fit_smooth:
    new_ext = '_smooth%d'%params['processing']['smooth_fwhm']+params['processing']['extension']
    file_extension = file_extension.replace(params['processing']['extension'],new_ext)

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

if roi != 'None':
    print('masking data for ROI %s'%roi)
    roi_ind = cortex.get_roi_verts(params['processing']['space'],roi) # get indices for that ROI
    data = data[roi_ind[roi]]

# load DM and create pRF stim object

# get screenshot pngs
png_path = os.path.join(os.getcwd(),'prf_stimuli')
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
hrf = spm_hrf(0,TR)

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = (np.array([14,22,25,41,55,71,74,82])-params['processing']['crop_pRF_TR'])*TR


# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['general']['screen_width'],
                         screen_distance_cm = params['general']['screen_distance'],
                         design_matrix = prf_dm,
                         TR = TR)

# sets up stimulus and hrf for this gaussian gridder
gg = Iso2DGaussianGridder(stimulus = prf_stim,
                          hrf = hrf,
                          filter_predictions = True,
                          window_length = params['processing']['sg_filt_window_length'],
                          polyorder = params['processing']['sg_filt_polyorder'],
                          highpass = True,
                          add_mean = True,
                          task_lengths = np.array([prf_dm.shape[-1]]))

# and css gridder
gg_css = CSS_Iso2DGaussianGridder(stimulus = prf_stim,
                                  hrf = hrf,
                                  filter_predictions = True,
                                  window_length = params['processing']['sg_filt_window_length'],
                                  polyorder = params['processing']['sg_filt_polyorder'],
                                  highpass = True,
                                  add_mean = True,
                                  task_lengths = np.array([prf_dm.shape[-1]]))



# Load pRF estimates 

# models to get single voxel and compare fits
models = ['gauss','iterative_gauss','css']

for _,model in enumerate(models): # run through each model
    
    estimates = []
    
    for _,field in enumerate(hemi): # each hemi field
        
        # name for 
        gii_file = [x for _,x in enumerate(med_filenames) if field in x][0]
        estimates_combi = os.path.split(gii_file)[-1].replace(params['processing']['extension'],'_'+model+'_estimates.npz')
    
        estimates_pth = os.path.join(fits_pth,model)
 
        if os.path.isfile(os.path.join(estimates_pth,estimates_combi)): # if combined estimates exists
            
            print('loading %s'%os.path.join(estimates_pth,estimates_combi))
            estimates.append(np.load(os.path.join(estimates_pth,estimates_combi))) #save both hemisphere estimates in same array

        else: # if not join chunks and save file
            
            estimates.append(join_chunks(estimates_pth,estimates_combi,field,
                                         chunk_num = total_chunks,fit_model = model))
            
    # mask estimates
    print('masking estimates')
    masked_estimates = mask_estimates(estimates, sj, params, ROI = roi, fit_model = model)
    
    # set timeseries of that vertex to plot
    if str(sys.argv[3]) == 'max':
        vertex = np.where(masked_estimates['rsq']==np.nanmax(masked_estimates['rsq']))[0][0]
        
    elif str(sys.argv[3]) == 'min':
        vertex = np.where(masked_estimates['rsq']==np.nanmin(masked_estimates['rsq']))[0][0]
        
    timeseries = data[vertex]
        
    if model == 'css':
        model_fit = gg_css.return_single_prediction(masked_estimates['x'][vertex], masked_estimates['y'][vertex],
                                                    masked_estimates['size'][vertex], masked_estimates['beta'][vertex],
                                                    masked_estimates['baseline'][vertex], masked_estimates['ns'][vertex])
    else:
        model_fit = gg.return_single_prediction(masked_estimates['x'][vertex], masked_estimates['y'][vertex],
                                                masked_estimates['size'][vertex], masked_estimates['beta'][vertex],
                                                masked_estimates['baseline'][vertex])

    print('vertex %d of ROI %s , rsq of %s fit is %.3f' %(vertex, roi, model, masked_estimates['rsq'][vertex]))
    
    
    # plot data with model
    time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
    fig = plt.figure(figsize=(15,7.5),dpi=100)
    plt.plot(time_sec, model_fit, c='#db3050', lw=3, label='prf model',zorder=1)
    plt.scatter(time_sec,timeseries, marker='v',c='k',label='data')
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('BOLD signal change (%)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,len(timeseries)*TR)
    
    if model == 'css':
        plt.title('vertex %d of ROI %s , rsq of %s fit is %.3f, n=%.2f' %(vertex, roi, model, masked_estimates['rsq'][vertex], masked_estimates['ns'][vertex]))
    else:
        plt.title('vertex %d of ROI %s , rsq of %s fit is %.3f' %(vertex, roi, model, masked_estimates['rsq'][vertex]))
        

    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(4):
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='r', alpha=0.1)
        ax_count += 2

    plt.legend(loc=0)

    filename = estimates_combi.replace('_estimates.npz','_ROI-%s_vertex-%s.svg'%(roi,vertex))
    fig.savefig(os.path.join(figures_pth,filename), dpi=100,bbox_inches = 'tight')





