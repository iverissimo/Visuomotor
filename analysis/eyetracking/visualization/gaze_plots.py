
################################################
#        Make gaze plots for each run
#              (sanity check)
################################################


import numpy as np
import glob, os, sys
import yaml
import pandas as pd

import matplotlib.pyplot as plt

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
deriv_pth = params['general']['paths']['data']['derivatives'] # path to derivatives folder

# subjects that did pRF task with linux computer, so res was full HD
HD_subs = [str(num).zfill(2) for num in params['general']['HD_screen_subs']] 
res = params['general']['screenRes_HD'] if sj in HD_subs else params['general']['screenRes']


# run for all tasks
for _,cond in enumerate(params['general']['tasks']):
    
    
    # load csvs with gaze data and timing info
    csv_pth = os.path.join(deriv_pth,'eyetracking', cond,'sub-{sj}'.format(sj=sj))

    if os.path.isdir(csv_pth):
        gaze_filenames = [os.path.join(csv_pth,x) for _,x in enumerate(os.listdir(csv_pth)) 
                          if 'gaze' in x and x.endswith('.csv')]
        
        timestamps_filenames = [os.path.join(csv_pth,x) for _,x in enumerate(os.listdir(csv_pth)) 
                              if 'timestamps' in x and x.endswith('.csv')]
    else:
        gaze_filenames = []
        timestamps_filenames = []
        
    if not gaze_filenames:
        
        print('no gaze files for %s runs of subject %s'%(cond,sj))
        
    else:
        # sort so runs are the same
        gaze_filenames.sort()
        timestamps_filenames.sort()

    	# set output path for plots
        out_pth = os.path.join(deriv_pth, 'plots','gaze', cond, 'sub-{sj}'.format(sj=sj))

        if not os.path.exists(out_pth): # check if path to save processed files exist
                os.makedirs(out_pth) 

        # for each run
        for ind, run in enumerate(gaze_filenames):
            
            # load timsetamps
            timestamps_pd = pd.read_csv(timestamps_filenames[ind],sep = '\t')
            
            # load gaze
            gaze_pd = pd.read_csv(run,sep = '\t')
            
            # compute array with all gaze x,y positions of each trial, for whole run

            # get start and end time for run
            run_start_smp = int(np.array(timestamps_pd[timestamps_pd['trial_start_index'] == 0]['trial_start_EL_timestamp'])[0])
            run_end_smp = int(np.array(timestamps_pd[timestamps_pd['trial_start_index'] == len(timestamps_pd['trial_start_index'])-1]['trial_end_EL_timestamp'])[0])

            # gaze can be from right or left eye, so set string first
            # using gaze_x/y_int - interpolated gaze
            xgaz_srt = gaze_pd.columns[[i for i in range(len(gaze_pd.keys().values)) if 'gaze_x_int' in gaze_pd.keys().values[i]][0]]
            ygaz_str = gaze_pd.columns[[i for i in range(len(gaze_pd.keys().values)) if 'gaze_y_int' in gaze_pd.keys().values[i]][0]]

            # get x and y positions, within run time
            x_pos = np.array(gaze_pd[np.logical_and(run_start_smp<=gaze_pd['time'],gaze_pd['time']<=run_end_smp)][xgaz_srt])
            y_pos = np.array(gaze_pd[np.logical_and(run_start_smp<=gaze_pd['time'],gaze_pd['time']<=run_end_smp)][ygaz_str])

            gaze_alltrl = np.array([x_pos.squeeze(),y_pos.squeeze()], dtype=object)
            
            # plot gaze!
            plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

            plt.plot(gaze_alltrl[0][:],c='k')
            plt.plot(gaze_alltrl[1][:],c='orange')

            plt.xlabel('Samples',fontsize=18)
            plt.ylabel('Position',fontsize=18)
            plt.legend(['xgaze','ygaze'], fontsize=10)
            plt.title('Gaze run-%s' %str(ind+1).zfill(2), fontsize=20)

            # add lines for each trial start
            # subtract start time to make it valid
            for k in range(len(timestamps_pd['trial_start_EL_timestamp'])):

                plt.axvline(x = timestamps_pd['trial_start_EL_timestamp'][k]-run_start_smp,c='r',linestyle='--',alpha=0.5) #start recording

            plt.savefig(os.path.join(out_pth,'gaze_xydata_run-%s.png' %str(ind+1).zfill(2)))
            plt.close()




