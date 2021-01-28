
################################################
#      run prf fit jobs through SLURM 
# (to use in cartesius or similar server)
################################################

import re, os
import glob, yaml
import sys

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

# define participant number 
if len(sys.argv)<2:
    raise NameError('Please add subject number (ex:01) or "all" '
                    'as 1st argument in the command line!')
    
else:
    if sys.argv[1] == 'all': # process all subjects  
        subjects = ['sub-'+str(x).zfill(2) for x in params['general']['subs']]
    else:     
        subjects = ['sub-'+str(sys.argv[1]).zfill(2)] #fill subject number with 0 in case user forgets

# number of chunks to split data in (makes fitting faster)
total_chunks = params['fitting']['prf']['total_chunks']

batch_string = """#!/bin/bash
#SBATCH -t 100:00:00
#SBATCH -N 1
#SBATCH -v
#SBATCH -c 24

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

python $PY_CODE $SJ_NR $CHUNK_NR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

# base directory in cartesius
base_dir = params['general']['paths']['fitting']['cartesius']['basedir']
# to save .sh files, for each chunk
batch_dir = os.path.join(base_dir,'batch')
# path to python fitting script  
pycode_dir = os.path.join(base_dir,'Visuomotor','analysis','fmri','fitting','pRF_fitmodel.py')

os.chdir(batch_dir)


for sub in subjects:

    for _,chu in enumerate(range(total_chunks)): # submit job for each chunk

        working_string = batch_string.replace('$SJ_NR', str(sub).zfill(2))
        working_string = working_string.replace('$CHUNK_NR', str(chu+1).zfill(3))
        working_string = working_string.replace('$PY_CODE', pycode_dir)

        js_name = os.path.join(batch_dir, 'pRF_' + sub + '_chunk-%s_of_%s'%(str(chu+1).zfill(3),str(total_chunks).zfill(3)) + '_iterative.sh')
        of = open(js_name, 'w')
        of.write(working_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        print(working_string)
        os.system('sbatch ' + js_name)

