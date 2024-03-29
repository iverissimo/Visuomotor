
################################################
#      run prf fit jobs through SLURM 
#      FOR MISSING FILES OF INITIAL RUN
################################################

import os
import os.path as op
import yaml
import pandas as pd

# load settings from yaml
with open(os.path.join(os.path.split(os.path.split(os.getcwd())[0])[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

subjects = ['01','02','04','08','09','11','12'] # subjects
base_dir = 'lisa' # where we are running the scripts
# number of chunks to split data in (makes fitting faster)
total_chunks = params['fitting']['prf']['total_chunks']

runs2fit = ['mean'] #['leave_01_out','leave_02_out','leave_03_out','leave_04_out','leave_05_out']

batch_string = """#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -v
#SBATCH -c 16
#SBATCH --output=$BATCHDIR/slurm_PRF-%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

source activate i36

cp -r $DATADIR/ $TMPDIR

wait

python $PY_CODE $SJ_NR $CHUNK_NR $RUN_TYPE

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/Visuomotor_data/ $DATADIR

wait

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

# base directory 
base_dir = params['general']['paths']['fitting']['lisa']['basedir']
# to save .sh files, for each chunk
batch_dir = op.join(base_dir,'batch')
# path to python fitting script  
pycode_dir = op.join(base_dir,'Visuomotor','analysis','fmri','fitting','pRF_fitmodel.py')
# directory with all data
data_dir = params['general']['paths']['fitting']['lisa']['data_dir']

# load missing files
missing_files = pd.read_csv(op.join(op.split(pycode_dir)[0],'missing_files.csv'))

#os.chdir(batch_dir)

for i in range(len(missing_files)):

    working_string = batch_string.replace('$SJ_NR', str(missing_files.iloc[i]['sub']).zfill(2))
    working_string = working_string.replace('$CHUNK_NR', str(int(missing_files.iloc[i]['chunk'])).zfill(3))
    working_string = working_string.replace('$PY_CODE', pycode_dir)
    working_string = working_string.replace('$BATCHDIR', batch_dir)
    working_string = working_string.replace('$RUN_TYPE', missing_files.iloc[i]['run_type'])
    working_string = working_string.replace('$DATADIR', data_dir)

    js_name = os.path.join(batch_dir, 'pRF_sub-' + str(missing_files.iloc[i]['sub']).zfill(2) + '_chunk-%s_of_%s'%(str(int(missing_files.iloc[i]['chunk'])).zfill(3),str(total_chunks).zfill(3)) + '_iterative.sh')
    of = open(js_name, 'w')
    of.write(working_string)
    of.close()

    print('submitting ' + js_name + ' to queue')
    print(working_string)
    os.system('sbatch ' + js_name)

