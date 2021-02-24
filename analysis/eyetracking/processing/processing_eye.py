
################################################
#      Load, process and convert EDF files, 
#           save hdf5 and pandas DF
# NOTE: hedfpy (installed in my local eye conda env)
# uses python 2 - need to swap conda env to run
################################################

import numpy as np
import glob, os, sys
import yaml
import pandas as pd
import hedfpy

sys.path.append(os.path.split(os.getcwd())[0]) # so script it finds util.py
from utils import * #import script to use relevante functions


# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01)'	
                    'as 1st argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	


# set paths
sourcedata_pth = params['general']['paths']['data']['sourcedata'] # path to sourcedata folder (where eyetrackin files are)
deriv_pth = params['general']['paths']['data']['derivatives'] # path to derivatives folder


# run for all tasks
for t,cond in enumerate(params['general']['tasks']):
    #list of absolute paths to all edf files for that task and subject
    edf_filenames = glob.glob(os.path.join(sourcedata_pth,'sub-{sj}'.format(sj=sj),'*','eyetrack/*'))
    edf_filenames = [x for _,x in enumerate(edf_filenames) if cond in x and x.endswith('.edf')]; edf_filenames.sort()

    if not edf_filenames:
        print('no eyetracking on %s runs, for subject %s'%(cond,sj))
    else:
        print('%d eyetracking edf files found, for subject %s. converting...'%(len(edf_filenames),sj))
        
        # set output path for processed files
        out_pth = os.path.join(deriv_pth, 'eyetracking', cond, 'sub-{sj}'.format(sj=sj))
        
        if not os.path.exists(out_pth): # check if path to save processed files exist
                os.makedirs(out_pth) 
                
        #single hdf5 file that contains all eye data for the runs of that task
        hdf_file = os.path.join(out_pth, 'sub-{sj}_task-{cond}_eyetrack.h5'.format(sj=sj, cond=cond))  
        
        # convert
        alias_list = edf2h5(edf_filenames, hdf_file, 
                           pupil_hp = params['eyetracking']['HP_pupil_f'], 
                           pupil_lp = params['eyetracking']['LP_pupil_f'])
        
        ho = hedfpy.HDFEyeOperator(hdf_file)
        
        for _,al in enumerate(alias_list):
            # load table with timestamps for run
            with pd.HDFStore(ho.input_object) as h5_file:
                # load table with timestamps for run
                table_timestamps = h5_file['%s/trials'%al] 

                # get start time for run
                run_start_smp = int(np.array(table_timestamps[table_timestamps['trial_start_index'] == 0]['trial_start_EL_timestamp'])[0])

                # load table with positions for run
                period_block_nr = ho.sample_in_block(sample = run_start_smp, block_table = h5_file['%s/blocks'%al]) 
                table_pos = h5_file['%s/block_%i'%(al, period_block_nr)]

            # save timestampts with output name
            table_timestamps.to_csv(os.path.join(out_pth,'timestamps_%s.csv'%al), sep="\t")

            # save gaze position etc 
            table_pos.to_csv(os.path.join(out_pth,'gaze_%s.csv'%al), sep="\t")



