
################################################
#      run prf fit jobs through SLURM 
# (to use in cartesius or similar server)
################################################

import os
import os.path as op
import yaml
import sys

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

subjects = ['01'] #['01','02', '03','04', '05', '07','08','09','11','12', '13'] #['01','02','04','08','09','11','12'] # subjects
base_dir = 'lisa' # where we are running the scripts
# number of chunks to split data in (makes fitting faster)
total_chunks = params['fitting']['prf']['total_chunks']

runs2fit = ['mean'] #['leave_01_out','leave_02_out','leave_03_out','leave_04_out','leave_05_out']

batch_string = """#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -v
#SBATCH -c 16
#SBATCH --output=$BATCHDIR/slurm_Visuomotor_PRF-%A.out

# call the programs

source activate i38

cp -r $DATADIR $TMPDIR

wait

python $PY_CODE $SJ_NR $CHUNK_NR $RUN_TYPE

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/Visuomotor_data/ $DATADIR

wait

"""

# base directory 
base_dir = params['general']['paths']['fitting']['lisa']['basedir']
# to save .sh files, for each chunk
batch_dir = op.join(base_dir,'batch')
# path to python fitting script  
pycode_dir = op.join(base_dir,'Visuomotor','analysis','fmri','fitting','pRF_fitmodel.py')
# directory with all data
data_dir = params['general']['paths']['fitting']['lisa']['data_dir']

#os.chdir(batch_dir)


for sub in subjects:

    for rt in runs2fit:

        for _,chu in enumerate(range(total_chunks)): # submit job for each chunk

            working_string = batch_string.replace('$SJ_NR', str(sub).zfill(2))
            working_string = working_string.replace('$CHUNK_NR', str(chu+1).zfill(3))
            working_string = working_string.replace('$PY_CODE', pycode_dir)
            working_string = working_string.replace('$BATCHDIR', batch_dir)
            working_string = working_string.replace('$RUN_TYPE', rt)
            working_string = working_string.replace('$DATADIR', data_dir)

            js_name = os.path.join(batch_dir, 'Visuomotor_pRF_sub-' + str(sub).zfill(2) + '_chunk-%s_of_%s'%(str(chu+1).zfill(3),str(total_chunks).zfill(3)) + '_iterative.sh')
            of = open(js_name, 'w')
            of.write(working_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            print(working_string)
            os.system('sbatch ' + js_name)

