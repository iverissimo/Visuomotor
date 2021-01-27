
# extra processing after fmriprep, for all tasks
# outputs relevant files in derivatives folder
# for further analysis

import os, yaml
import sys, glob
import re 

import numpy as np
import pandas as pd

sys.path.append(os.path.split(os.getcwd())[0]) # so script it finds util.py
from utils import * #import script to use relevante functions


# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number and open json parameter file
if len(sys.argv)<2:
    raise NameError('Please add subject number (ex:01) or "all" '
                    'as 1st argument in the command line!')
    
else:
    if sys.argv[1] == 'all': # process all subjects  
        sj = ['sub-'+str(x).zfill(2) for x in params['general']['subs']]
    else:     
        sj = ['sub-'+str(sys.argv[1]).zfill(2)] #fill subject number with 0 in case user forgets


# define paths and list of files
deriv_pth = params['general']['paths']['data']['derivatives'] # path to derivatives folder
fmriprep_pth = os.path.join(deriv_pth,'fmriprep') # path to fmriprep files

# run for all tasks
for t,cond in enumerate(params['general']['tasks']):
    
    # and all subjects selected
    for _,s in enumerate(sj):
    
        # list of functional files
        filename = [run for run in glob.glob(os.path.join(fmriprep_pth,s,'*','func/*')) if cond in run and params['processing']['space'] in run and run.endswith(params['processing']['extension'])]
        filename.sort()
        
        if not filename: # if list empty
            print('Subject %s has no files for %s' %(s,cond))
            
        else:
    
            TR = params['general']['TR']

            # set output path for processed files
            out_pth = os.path.join(deriv_pth,'post_fmriprep',cond,s)
            
            if not os.path.exists(out_pth): # check if path to save processed files exist
                os.makedirs(out_pth) 
                
            for _,file in enumerate(filename):

                # define hemisphere to plot
                hemi = 'left' if '_hemi-L' in file else 'right'
                
                if cond in ('prf'): # if pRF we cut out first 7TRs from "raw file" to make further analysis better

                    file = crop_gii(file,params['processing']['crop_pRF_TR'],out_pth,extension = params['processing']['extension'])
                
                # high pass filter all runs (savgoy-golay)
                _ ,filt_gii_pth = highpass_gii(file,params['processing']['sg_filt_polyorder'],params['processing']['sg_filt_deriv'],
                                                         params['processing']['sg_filt_window_length'],out_pth, extension = params['processing']['extension'])

                
                # do PSC
                _ , psc_data_pth = psc_gii(filt_gii_pth, out_pth, method = 'median', extension = params['processing']['extension']) 


