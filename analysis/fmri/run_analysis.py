import re, os
import glob, yaml
import sys
import os.path as op

import argparse

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

from preproc_mridata import MRIData
from soma_model import GLM_Model, somaRF_Model
from prf_model import prfModel

# defined command line options
# this also generates --help and error handling
CLI = argparse.ArgumentParser()

CLI.add_argument("--subject",
                nargs="*",
                type=str,  # any type/callable can be used here
                default=[],
                required=True,
                help = 'Subject number (ex:1). If "all" will run for all participants. If list of subs, will run only those (ex: 1 2 3 4)'
                )
CLI.add_argument("--system",
                #nargs="*",
                type=str,  # any type/callable can be used here
                default = 'local',
                help = 'Where are we running analysis? (ex: local, lisa). Default local'
                )
CLI.add_argument("--exclude",  # name on the CLI - drop the `--` for positional/required parameters
                nargs="*",  # 0 or more values expected => creates a list
                type=int,
                default=[],  # default if nothing is provided
                help = 'List of subs to exclude (ex: 1 2 3 4). Default []'
                )
CLI.add_argument("--cmd",
                #nargs="*",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'What analysis to run?\n Options: postfmriprep, fit_prf, fit_glm, etc'
                )
CLI.add_argument("--task",
                type=str,  # any type/callable can be used here
                default = 'pRF',
                help = 'What task we want to run? pRF [Default] vs soma'
                )
CLI.add_argument("--wf_dir", 
                    type = str, 
                    help="Path to workflow dir, if such if not standard root dirs(None [default] vs /scratch)")

CLI.add_argument("--n_jobs", 
                type = int, 
                default = 8,
                help="number of jobs for parallel")

# options for pRF fitting
CLI.add_argument("--run_type", 
                default = 'mean_run',
                help="Type of run to fit (mean_run [default], 1, loo_run, ...)")
CLI.add_argument("--fit_hrf", 
                type = int, 
                default = 0,
                help="1/0 - if we want to fit hrf on the data or not [default] - for prf fitting")
CLI.add_argument("--chunk_num", 
                type = int, 
                default = None,
                help="Chunk number to fit - for prf fitting")
CLI.add_argument("--vertex", 
                type = int, 
                default = None,
                help="Vertex number to fit")
CLI.add_argument("--ROI", 
                type = str, 
                default = None,
                help="ROI name to fit")
CLI.add_argument("--model2fit", 
                type = str, 
                default = 'gauss',
                help="pRF/soma model name to fit [gauss, css, glm, glmsingle, somaRF]")

# other options
CLI.add_argument("--data_type", 
                type = str, 
                default = 'anat',
                help="Data type to run fmriprep/mriqc")

# options for SLURM systems
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

wf_dir = args.wf_dir
n_jobs = args.n_jobs

# task and model to analyze
task = args.task
model2fit = args.model2fit
run_type = args.run_type

## prf options
fit_hrf = args.fit_hrf
chunk_num = args.chunk_num
vertex = args.vertex
ROI = args.ROI

## slurm options
node_name = args.node_name
partition_name = args.partition_name
batch_mem_Gib = args.batch_mem_Gib
hours = args.hours
run_time = '{h}:00:00'.format(h = str(hours)) #'10:00:00'
send_email = bool(args.email)

## fmriprep/mriqc options
data_type = args.data_type

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj, wf_dir = wf_dir)

## Run specific command

# task agnostic
if py_cmd == 'postfmriprep':

    Visuomotor_data.post_fmriprep_proc()

elif py_cmd == 'fmriprep':

    Visuomotor_data.call_fmriprep(data_type = data_type,
                     partition_name = partition_name, node_name = node_name,
                     root_folder = Visuomotor_data.params['general']['paths'][Visuomotor_data.base_dir]['test'],
                     node_mem = 5000, batch_mem_Gib = batch_mem_Gib, run_time = run_time)

# if running prf analysis
if task in ['pRF', 'prf']:

    # load data model
    data_model = prfModel(Visuomotor_data)

    # call command
    if py_cmd == 'fit_prf':

        # if we want to fit HRF params
        data_model.fit_hrf = True if bool(fit_hrf) == True else False

        # get participant models, which also will load DM
        pp_prf_models = data_model.set_models(participant_list = data_model.MRIObj.sj_num, filter_predictions = True)

        for pp in data_model.MRIObj.sj_num:

            data_model.fit_data(pp, pp_prf_models = pp_prf_models, 
                                fit_type = run_type, 
                                chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                model2fit = model2fit, outdir = None, save_estimates = True,
                                xtol = 1e-3, ftol = 1e-4, n_jobs = n_jobs)

# motor task analysis
elif task == 'soma':

    # load data model
    data_model = GLM_Model(Visuomotor_data)

    # if we want nilearn dm or custom 
    custom_dm = True if Visuomotor_data.params['fitting']['soma']['use_nilearn_dm'] == False else False 
    
    # call command
    if py_cmd == 'fit_glm':
        
        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            data_model.fit_data(pp, fit_type = run_type, custom_dm = custom_dm, 
                                    keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'])

    elif py_cmd == 'stats_glm':

        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            data_model.contrast_regions(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'],
                                            custom_dm = custom_dm, fit_type = run_type,
                                            keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'])

    elif py_cmd == 'fixed_effects':
        
        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            data_model.fixed_effects_contrast_regions(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'],
                                                        custom_dm = custom_dm, fit_type = run_type,
                                                        keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'])
            
    elif py_cmd == 'Fstat':

        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            data_model.f_goodness_of_fit(pp,custom_dm = custom_dm, fit_type = run_type, 
                                         keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'],
                                         alpha_fdr = 0.01)

    elif py_cmd == 'fit_RF':

        ## make RF model object
        data_RFmodel = somaRF_Model(Visuomotor_data)

        if Visuomotor_data.params['fitting']['soma']['keep_b_evs']:
            keep_b_evs = True
            region_keys = ['face', 'right_hand', 'left_hand', 'both_hand']
        else:
            keep_b_evs = False
            region_keys = ['face', 'right_hand', 'left_hand']

        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            data_RFmodel.fit_data(pp, somaModelObj = data_model, betas_model = 'glm',
                                    fit_type = run_type, nr_grid = 100, n_jobs = n_jobs,
                                    region_keys = region_keys, keep_b_evs = keep_b_evs,
                                    custom_dm = custom_dm)






