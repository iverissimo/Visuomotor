
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
        subjects = [str(x) for x in params['general']['subs']]
    else:     
        subjects = [str(sys.argv[1])] # make it string

# number of chunks to split data in (makes fitting faster)
total_chunks = params['fitting']['prf']['total_chunks']


batch_string = """#!/bin/sh

conda activate i36

cd $CODE_DIR

python $PY_CODE $SJ_NR $CHUNK_NR


"""

# base directory in cartesius
base_dir = params['general']['paths']['fitting']['local']['basedir']
# to save .sh files, for each chunk
batch_dir = os.path.join(base_dir,'batch')
if not os.path.exists(batch_dir): # check if path to save processed files exist
    os.makedirs(batch_dir) 
# path to python fitting script  
pycode_dir = os.path.join(base_dir,'analysis','fmri','fitting')

os.chdir(batch_dir)


for sub in subjects:

    for _,chu in enumerate(range(total_chunks)): # submit job for each chunk

        working_string = batch_string.replace('$SJ_NR', str(sub).zfill(2))
        working_string = working_string.replace('$CHUNK_NR', str(chu+1).zfill(3))
        working_string = working_string.replace('$CODE_DIR', pycode_dir)
        working_string = working_string.replace('$PY_CODE', 'pRF_fitmodel.py')

        js_name = os.path.join(batch_dir, 'pRF_sub-' + str(sub).zfill(2) + '_chunk-%s_of_%s'%(str(chu+1).zfill(3),str(total_chunks).zfill(3)) + '_iterative.sh')
        of = open(js_name, 'w')
        of.write(working_string)
        of.close()

        print('running ' + js_name)
        print(working_string)
        os.system('cd ' + batch_dir + '; chmod u+x ' + os.path.split(js_name)[-1])
        os.system(os.path.join('.',os.path.split(js_name)[-1]))

