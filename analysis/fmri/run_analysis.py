import re, os
import glob, yaml
import sys
import os.path as op

import argparse

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

from preproc_mridata import MRIData
from soma_model import GLMsingle_Model, GLM_Model, somaRF_Model
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
                help = 'What analysis to run?\n Options: postfmriprep, '
                )

CLI.add_argument("--wf_dir", 
                    type = str, 
                    help="Path to workflow dir, if such if not standard root dirs(None [default] vs /scratch)")

# options for pRF fitting
CLI.add_argument("--run_type", 
                default = 'mean_run',
                help="Type of run to fit (mean of runs [default], 1, loo_run, ...)")
CLI.add_argument("--fit_hrf", 
                type = int, 
                default = 0,
                help="1/0 - if we want to fit hrf on the data or not [default]")
CLI.add_argument("--chunk_num", 
                type = int, 
                default = None,
                help="Chunk number to fit")
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
                help="pRF model name to fit")

# parse the command line
args = CLI.parse_args()

# access CLI options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
system_dir = args.system
exclude_sj = args.exclude
py_cmd = args.cmd

wf_dir = args.wf_dir

## prf options
run_type = args.run_type
fit_hrf = args.fit_hrf
chunk_num = args.chunk_num
vertex = args.vertex
ROI = args.ROI
model2fit = args.model2fit

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj, wf_dir = wf_dir)

## Run specific command
if py_cmd == 'postfmriprep':

    Visuomotor_data.post_fmriprep_proc()

elif py_cmd == 'fit_prf':

    data_model = prfModel(Visuomotor_data)

    if bool(fit_hrf) == True:
        data_model.fit_hrf = True
    else:
        data_model.fit_hrf = False

    # get participant models, which also will load 
    # DM and mask it according to participants behavior
    pp_prf_models = data_model.set_models(participant_list = data_model.MRIObj.sj_num)

    for pp in data_model.MRIObj.sj_num:

        data_model.fit_data(pp, pp_prf_models = pp_prf_models, 
                            fit_type = run_type, 
                            chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                            model2fit = model2fit, outdir = None, save_estimates = True,
                            xtol = 1e-3, ftol = 1e-4, n_jobs = 8)


elif py_cmd == 'fit_glmsingle':

    data_model = GLMsingle_Model(Visuomotor_data)

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.fit_data(pp)

elif py_cmd == 'fit_glm':

    data_model = GLM_Model(Visuomotor_data)

    # if we want nilearn dm or custom 
    custom_dm = True if Visuomotor_data.params['fitting']['soma']['use_nilearn_dm'] == False else False 

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.fit_data(pp, fit_type = 'mean_run', custom_dm = custom_dm)

elif py_cmd == 'stats_glmsingle':

    data_model = GLMsingle_Model(Visuomotor_data)

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.compute_roi_stats(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'])

elif py_cmd == 'stats_glm':

    data_model = GLM_Model(Visuomotor_data)

    # if we want nilearn dm or custom 
    custom_dm = True if Visuomotor_data.params['fitting']['soma']['use_nilearn_dm'] == False else False 

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.contrast_regions(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'],
                                        custom_dm = custom_dm)









