
# extra processing after fmriprep, for all tasks
# outputs relevant files in derivatives folder
# for further analysis

import os, yaml
import os.path as op
import sys, glob
import re 
import numpy as np

sys.path.append(op.split(os.getcwd())[0]) # so script it finds visuomotor_utils.py
from visuomotor_utils import crop_gii, dc_filter_gii, highpass_gii, psc_gii, median_gii, mean_gii, leave_one_out #import script to use relevant functions


# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number 
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
fmriprep_pth = op.join(deriv_pth,'fmriprep') # path to fmriprep files

# general settings
TR = params['general']['TR']

# run for all tasks
for t,cond in enumerate(params['general']['tasks']):
    
    # and all subjects selected
    for _,s in enumerate(sj):
    
        # list of functional files
        filename = [run for run in glob.glob(op.join(fmriprep_pth,s,'*','func/*')) if cond in run and params['processing']['space'] in run and run.endswith(params['processing']['extension'])]
        
        print('%i files found for %s task %s'%(len(filename), s, cond))
        
        if not filename: # if list empty
            print('Subject %s has no files for %s' %(s,cond))
            
        else:
            # set output path for processed files
            out_pth = op.join(deriv_pth,'post_fmriprep',cond,s)
            
            # make path to save processed files 
            os.makedirs(out_pth, exist_ok=True) 
                 
            # store all psc absolute filename to use later   
            all_psc_absfile = [] 
            
            for _,file in enumerate(filename):

                # define hemisphere to plot
                hemi = 'left' if '_hemi-L' in file else 'right'
                
                if cond in ('prf'): # if pRF we cut out first 7TRs from "raw file" to make further analysis better

                    file = crop_gii(file,params['processing']['crop_pRF_TR'],out_pth,extension = params['processing']['extension'])
                
                    # high pass filter all runs (with discrete cosine set)
                    _ ,filt_gii_pth = dc_filter_gii(file, out_pth, extension = params['processing']['extension'],
                                                                    first_modes_to_remove = params['processing']['first_modes_to_remove'])
                
                if cond in ('soma'): # different type of filtering for soma

                    # copy raw files (to use for glm single)
                    os.system('cp {og_file} {new_file}'. format(og_file = file, 
                                                                new_file = op.join(out_pth, op.split(file)[-1])))

                    # high pass filter all runs (savgoy-golay)
                    _ ,filt_gii_pth = highpass_gii(file,params['processing']['sg_filt_polyorder'],params['processing']['sg_filt_deriv'],
                                                         params['processing']['sg_filt_window_length'],out_pth, extension = params['processing']['extension'])


                # do PSC
                _ , psc_data_pth = psc_gii(filt_gii_pth, out_pth, method = 'mean', extension = params['processing']['extension']) 

                all_psc_absfile.append(psc_data_pth)
            
            # make mean/median file or LOO file, depending on what we're going to fit
            
            file_path = op.split(psc_data_pth)[0]
            fit_type = params['fitting'][cond]['type']
            
            # make directory to save average run files
            average_out = op.join(file_path,fit_type) 

            # check if path to save processed files exist
            os.makedirs(average_out, exist_ok=True)

            if  fit_type=='median':
                # make average of all runs
                
                med_gii = []
                for field in ['hemi-L', 'hemi-R']:
                    hemi = [h for h in all_psc_absfile if field in h and 'run-median' not in h]  #we don't want to average median run if already in original dir

                    # set name for median run (now numpy array)
                    med_file = op.join(average_out, re.sub('run-\d{2}_', 'run-median_', op.split(hemi[0])[-1]))

                    # if file doesn't exist
                    if not os.path.exists(med_file):
                        med_gii.append(median_gii(hemi, average_out))  # create it
                        print('computed %s' % (med_gii))
                    else:
                        med_gii.append(med_file)
                        print('median file %s already exists, skipping' % (med_gii))

            elif fit_type=='mean':
                # make average of all runs
                
                med_gii = []
                for field in ['hemi-L', 'hemi-R']:
                    hemi = [h for h in all_psc_absfile if field in h and 'run-mean' not in h]  #we don't want to average median run if already in original dir

                    # set name for median run (now numpy array)
                    med_file = op.join(average_out, re.sub('run-\d{2}_', 'run-mean_', op.split(hemi[0])[-1]))

                    # if file doesn't exist
                    if not os.path.exists(med_file):
                        med_gii.append(mean_gii(hemi, average_out))  # create it
                        print('computed %s' % (med_gii))
                    else:
                        med_gii.append(med_file)
                        print('median file %s already exists, skipping' % (med_gii))
                        
            elif fit_type=='loo_run':
                # leave-one-run-out
                
                for field in ['hemi-L', 'hemi-R']:
                    hemi = [h for h in all_psc_absfile if field in h and 'run-leave' not in h]  #we don't want to average over wrong runs

                    # make list of run numbers
                    if cond == 'prf':
                        run_numbers = [x[-53:-47] for _,x in enumerate(hemi)] 
                    elif cond == 'soma':
                        run_numbers = [x[-45:-39] for _,x in enumerate(hemi)]
                    
                    loo_lists = leave_one_out(hemi) # subdivide files into lists where one run is left out
                    
                    for r,ll in enumerate(loo_lists):
                        
                        print('averaging %s'%str(ll))
                        
                        # set name for median run (now numpy array)
                        med_file = op.join(average_out, re.sub('run-\d{2}_', 'run-leave_%s_out_'%(run_numbers[r][-2:]), os.path.split(ll[0])[-1]))
                    
                        # if file doesn't exist
                        if not op.exists(med_file):
                            med_gii = median_gii(ll, average_out,run_name='leave_%s_out'%(run_numbers[r][-2:]))  # create it
                            print('computed %s' % (med_gii))
                        else:
                            print('median file %s already exists, skipping' % (med_file))

    
    

                   
