import re, os
import glob, yaml
import sys
import os.path as op
import numpy as np

import argparse

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

from preproc_mridata import MRIData
from soma_model import GLMsingle_Model, GLM_Model
from prf_model import prfModel

from viewer import somaViewer

# defined command line options
# this also generates --help and error handling
CLI = argparse.ArgumentParser()

CLI.add_argument("--subject",
                nargs="*",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'Subject number (ex:1). If "all" will run for all participants. If list of subs, will run only those (ex: 1 2 3 4)'
                )

CLI.add_argument("--system",
                type=str,  # any type/callable can be used here
                default = 'lisa',
                help = 'Where are we running analysis? (ex: local, lisa). Default lisa'
                )

CLI.add_argument("--exclude",  # name on the CLI - drop the `--` for positional/required parameters
                nargs="*",  # 0 or more values expected => creates a list
                type=int,
                default=[],  # default if nothing is provided
                help = 'List of subs to exclude (ex: 1 2 3 4). Default []'
                )

CLI.add_argument("--cmd",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'What analysis to run?\n Options: fit_prf, '
                )

CLI.add_argument("--task",
                type=str,  # any type/callable can be used here
                default = 'pRF',
                help = 'What task we want to run? pRF [Default] vs soma'
                )

# only relevant for pRF fitting
CLI.add_argument("--prf_model_name", 
                type = str, 
                default = 'gauss',
                help="Type of pRF model to fit: gauss [default], css, dn, etc...")
CLI.add_argument("--fit_hrf", 
                type = int, 
                default = 0,
                help="1/0 - if we want to fit hrf on the data or not [default]")
CLI.add_argument("--run_type", 
                default = 'mean_run',
                help="Type of run to fit (mean of runs [default], 1, loo_run, ...)")

# if we want to divide data in batches (chunks)
CLI.add_argument("--chunk_data", 
                type = int, 
                default = 1,
                help="1/0 - if we want to divide the data into chunks [default] or not")

# nr jobs for parallel
CLI.add_argument("--n_jobs", 
                type = int, 
                default = 16,
                help="number of jobs for parallel")

# only relevant for LISA/slurm system 
CLI.add_argument("--node_name", 
                type = str, 
                help="Node name, to send job to [default None]")
CLI.add_argument("--partition_name", 
                type = str, 
                help="Partition name, to send job to [default None]")
CLI.add_argument("--batch_mem_Gib", 
                type = int, 
                default = 90,
                help="Node memory limit [default 90]")
CLI.add_argument("--email", 
                type = int, 
                default = 0,
                help="Send job email 1/0 [default 0]")
CLI.add_argument("--hours", 
                type = int, 
                default = 10,
                help="Number of hours to set as time limit for job")


# parse the command line
args = CLI.parse_args()

# access CLI options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
system_dir = args.system
exclude_sj = args.exclude

py_cmd = args.cmd

task = args.task
model2fit = args.prf_model_name

fit_hrf = args.fit_hrf
run_type = args.run_type
chunk_data = args.chunk_data

n_jobs = args.n_jobs

node_name = args.node_name
partition_name = args.partition_name
batch_mem_Gib = args.batch_mem_Gib

hours = args.hours
run_time = '{h}:00:00'.format(h = str(hours)) #'10:00:00'
send_email = bool(args.email)


## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj)

## set start of slurm command

slurm_cmd = """#!/bin/bash
#SBATCH -t {rtime}
#SBATCH -N 1
#SBATCH -v
#SBATCH --cpus-per-task=16
#SBATCH --output=$BD/slurm_Visuomotor_{task}_{model}_fit_%A.out\n""".format(rtime=run_time, task = task, model = model2fit)
                    
if partition_name is not None:
    slurm_cmd += '#SBATCH --partition {p}\n'.format(p=partition_name)
if node_name is not None:
    slurm_cmd += '#SBATCH -w {n}\n'.format(n=node_name)

# add memory for node
slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=batch_mem_Gib)

# set fit folder name
if task == 'pRF':
    fitfolder = 'pRF_fit'
elif task == 'soma':
    fitfolder = 'somaRF_fits'

# batch dir to save .sh files
batch_dir = Visuomotor_data.params['general']['paths']['lisa']['batch']

# loop over participants
for pp in Visuomotor_data.sj_num:

    if task == 'pRF':

        # if we're chunking the data, then need to submit each chunk at a time
        if chunk_data:
            # total number of chunks
            total_ch = Visuomotor_data.params['fitting']['prf']['total_chunks']
            ch_list = np.arange(total_ch)
        else:
            ch_list = [None]

        # set fitting model command 
        for ch in ch_list:

            if ch is None:

                fit_cmd = """python run_analysis.py --subject {pp} --system {system} --cmd fit_prf \
    --run_type {rt} --model2fit {prf_mod} --n_jobs {nj} --fit_hrf {ft} --wf_dir $TMPDIR\n\n
    """.format(pp = pp,
            system = system_dir,
            rt = run_type,
            prf_mod = model2fit,
            ft = fit_hrf,
            nj = n_jobs)

            else:
                fit_cmd = """python run_analysis.py --subject {pp} --system {system} --cmd fit_prf \
    --chunk_num {ch} --run_type {rt} --model2fit {prf_mod} --n_jobs {nj} --fit_hrf {ft} --wf_dir $TMPDIR\n\n
    """.format(pp = pp,
            system = system_dir,
            ch = ch,
            rt = run_type,
            prf_mod = model2fit,
            ft = fit_hrf,
            nj = n_jobs)

            slurm_cmd = slurm_cmd + """# call the programs
    $START_EMAIL

    # make derivatives dir in node and sourcedata because we want to access behav files
    mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/prf/sub-$SJ_NR
    mkdir -p $TMPDIR/derivatives/$FITFOLDER/sub-$SJ_NR
    # mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR

    wait
    cp -r $DERIV_DIR/post_fmriprep/$SPACE/prf/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep/$SPACE/prf/

    wait

    # cp -r $SOURCE_DIR/sub-$SJ_NR $TMPDIR/sourcedata/

    wait

    if [ -d "$DERIV_DIR/$FITFOLDER/sub-$SJ_NR" ] 
    then
        cp -r $DERIV_DIR/$FITFOLDER/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER
    fi

    wait

    """
            ### update slurm job script
            batch_string =  slurm_cmd + """$PY_CMD

    wait          # wait until programs are finished

    rsync -chavzP --exclude=".*" $TMPDIR/derivatives/ $DERIV_DIR

    wait          # wait until programs are finished

    $END_EMAIL
    """

            ### if we want to send email
            if send_email == True:
                batch_string = batch_string.replace('$START_EMAIL', 'echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"')
                batch_string = batch_string.replace('$END_EMAIL', 'echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"')

            ## replace other variables
            working_string = batch_string.replace('$SJ_NR', str(pp).zfill(2))
            working_string = working_string.replace('$SPACE', Visuomotor_data.sj_space)
            working_string = working_string.replace('$FITFOLDER', fitfolder)
            working_string = working_string.replace('$PY_CMD', fit_cmd)
            working_string = working_string.replace('$BD', batch_dir)
            working_string = working_string.replace('$DERIV_DIR', Visuomotor_data.derivatives_pth)
            working_string = working_string.replace('$SOURCE_DIR', Visuomotor_data.sourcedata_pth)

            print(working_string)

            # run it
            js_name = op.join(batch_dir, '{fname}_sub-{sj}_chunk-{ch}_run-{r}_Visuomotor.sh'.format(fname=fitfolder,
                                                                                    ch=ch,
                                                                                    sj=pp,
                                                                                    r=run_type))
            of = open(js_name, 'w')
            of.write(working_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            os.system('sbatch ' + js_name)


    elif task == 'soma':
        fit_cmd = """python run_analysis.py --subject {pp} --system {system} --cmd fit_RF \
--run_type {rt} --task soma --model2fit somaRF --fit_hrf {ft} --wf_dir $TMPDIR\n\n
""".format(pp = pp,
        system = system_dir,
        rt = run_type,
        ft = fit_hrf)

        slurm_cmd = slurm_cmd + """# call the programs
$START_EMAIL

# make derivatives dir in node and sourcedata because we want to access behav files
mkdir -p $TMPDIR/derivatives/{glm_fits,$FITFOLDER}/sub-$SJ_NR
mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/soma/sub-$SJ_NR
mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR

wait

cp -r $DERIV_DIR/glm_fits/sub-$SJ_NR $TMPDIR/derivatives/glm_fits/

wait

wait

cp -r $DERIV_DIR/post_fmriprep/$SPACE/soma/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep/$SPACE/soma/

wait

cp -r $SOURCE_DIR/sub-$SJ_NR $TMPDIR/sourcedata/

wait

if [ -d "$DERIV_DIR/$FITFOLDER/sub-$SJ_NR" ] 
then
    cp -r $DERIV_DIR/$FITFOLDER/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER
fi

wait

"""
        ### update slurm job script
        batch_string =  slurm_cmd + """$PY_CMD

wait          # wait until programs are finished

rsync -chavzP --exclude=".*" $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

$END_EMAIL
"""

        ### if we want to send email
        if send_email == True:
            batch_string = batch_string.replace('$START_EMAIL', 'echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"')
            batch_string = batch_string.replace('$END_EMAIL', 'echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"')

        ## replace other variables
        working_string = batch_string.replace('$SJ_NR', str(pp).zfill(2))
        working_string = working_string.replace('$FITFOLDER', fitfolder)
        working_string = working_string.replace('$PY_CMD', fit_cmd)
        working_string = working_string.replace('$BD', batch_dir)
        working_string = working_string.replace('$DERIV_DIR', Visuomotor_data.derivatives_pth)
        working_string = working_string.replace('$SOURCE_DIR', Visuomotor_data.sourcedata_pth)
        working_string = working_string.replace('$SPACE', Visuomotor_data.sj_space)

        print(working_string)

        # run it
        js_name = op.join(batch_dir, '{fname}_sub-{sj}_run-{r}_Visuomotor.sh'.format(fname=fitfolder,
                                                                                sj=pp,
                                                                                r=run_type))
        of = open(js_name, 'w')
        of.write(working_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        os.system('sbatch ' + js_name)