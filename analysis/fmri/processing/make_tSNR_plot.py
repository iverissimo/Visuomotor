
################################################
#  Create tSNR plots for a specific surface file
################################################


import os, yaml
import sys, glob
import re 

import matplotlib.colors as colors

from utils import *

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm
import matplotlib.pyplot as plt
from distutils.util import strtobool

from nilearn import surface

import cortex


if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '
                    'as 1st argument in the command line!')
                    
elif len(sys.argv)<3:
    raise NameError('Please select task to compute tSNR plots ' 
                    'as 2nd argument in the command line!')
                    
elif len(sys.argv)<4:
    raise NameError('Please select if tSNR plot for median run (median) ' 
                    'or single runs (single) as 3rd argument in the command line!')
                             
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets
    
    task = str(sys.argv[2])

    # load settings from yaml
    with open(os.path.join(os.path.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
        params = yaml.safe_load(f_in)	
                
    run_type = str(sys.argv[3])

# define paths and list of files
deriv_pth = params['general']['paths']['data']['derivatives'] # path to derivatives folder
fmriprep_pth = os.path.join(deriv_pth,'fmriprep') # path to fmriprep files
post_fmriprep_pth = os.path.join(deriv_pth,'post_fmriprep') # path to post_fmriprep files


# path to save plots
out_dir = os.path.join(deriv_pth,'tSNR', task,'sub-{sj}'.format(sj=sj))

if not os.path.exists(out_dir):  # check if path exists
    os.makedirs(out_dir)


# list of fmriprep functional files
orig_filename = [run for run in glob.glob(os.path.join(fmriprep_pth,'sub-{sj}'.format(sj=sj),'*','func/*')) if 'task-'+task in run and params['processing']['space'] in run and run.endswith(params['processing']['extension'])]
orig_filename.sort()

# list of post fmriprep functional files
# last part of filename to use
if task == 'prf':
    file_extension = 'cropped_sg.func.gii'
else:
    file_extension = '_sg.func.gii'

post_filepath = [run for run in glob.glob(os.path.join(post_fmriprep_pth,'sub-{sj}'.format(sj=sj),'*','func/*')) if 'task-'+task in run and params['processing']['space'] in run and run.endswith(file_extension)]
post_filepath.sort()


# do same plots for pre and post processed files
for files in ['pre','post']:
    
    filename = orig_filename.copy() if files=='pre' else post_filename.copy() # choose correct list with absolute filenames
    
    gii_files = []
    if run_type == 'single':
        for run,_ in enumerate(filename):
            gii_files.append([r for r in filename if 'run-'+str(run).zfill(2) in r])
            #print(gii_files)

    elif run_type == 'median':
    
        for field in ['hemi-L', 'hemi-R']: #choose one hemi at a time
            hemi = [h for h in filename if field in h and 'run-median' not in h] # make median run in output dir, 
                                                                        # but we don't want to average median run if already in original dir
            # set name for median run (now numpy array)
            med_file = os.path.join(out_dir, re.sub(
                'run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
            # if file doesn't exist
            if not os.path.exists(med_file):
                gii_files.append(median_gii(hemi, out_dir))  # create it
                print('computed %s' % (gii_files))
            else:
                gii_files.append(med_file)
                print('median file %s already exists, skipping' % (gii_files))
        gii_files = [gii_files] # then format identical

    # load and combine both hemispheres
    for indx,list_pos in enumerate(gii_files):
        data_array = []
        if not list_pos:
            print('no files for run-%s'%str(indx).zfill(2))
        else:
            for val in list_pos :
                data_array.append(np.array(surface.load_surf_data(val))) #save both hemisphere estimates in same array
            data_array = np.vstack(data_array)
            
            new_filename = os.path.split(val)[-1].replace('hemi-R','hemi-both')
            print('making tSNR flatmap for %s'%str(new_filename))
            
            # make tsnr map
            stat_map = np.mean(data_array, axis = 1)/np.std(data_array, axis = 1)
            
            # save histogram of values
            fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
            plt.hist(stat_map)
            plt.title('Histogram of tSNR values for %s run-%s of sub-%s'%(task,str(indx).zfill(2),sj))
            fig.savefig(os.path.join(out_dir,('histogram_'+new_filename).replace(params['processing']['extension'],'.png')), dpi=100)
            
            up_lim = 200
            low_lim = 0
            colormap = 'viridis'

            # and plot it
            tsnr_flat = cortex.dataset.Vertex(stat_map.T, params['processing']['space'],
                                 vmin=low_lim, vmax=up_lim, cmap=colormap)
            
            _ = cortex.quickflat.make_png(os.path.join(out_dir,('flatmap_'+new_filename.replace(params['processing']['extension'],'.png'))),
                                          tsnr_flat, recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)

    
    


