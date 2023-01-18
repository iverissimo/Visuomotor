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

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj, wf_dir = wf_dir)

## Run specific command

# task agnostic
if py_cmd == 'postfmriprep':

    Visuomotor_data.post_fmriprep_proc()

# if running prf analysis
if task in ['pRF', 'prf']:

    # load data model
    data_model = prfModel(Visuomotor_data)

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

elif task == 'soma':

    # if standard GLM model or RF model that uses GLM betas
    if (model2fit == 'glm') or \
        ((model2fit == 'somaRF') and (Visuomotor_data.params['fitting']['soma']['somaRF']['beta_model'] == 'glm')):

        # load data model
        data_model = GLM_Model(Visuomotor_data)

        # if we want nilearn dm or custom 
        custom_dm = True if Visuomotor_data.params['fitting']['soma']['use_nilearn_dm'] == False else False 

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

        elif py_cmd == 'fit_RF':

            ## make RF model object
            data_RFmodel = somaRF_Model(Visuomotor_data)

            ## loop over all subjects 
            for pp in Visuomotor_data.sj_num:
                data_RFmodel.fit_data(pp, somaModelObj = data_model, betas_model = 'glm',
                                        fit_type = 'mean_run', nr_grid = 100, n_jobs = n_jobs,
                                        region_keys = ['face', 'right_hand', 'left_hand'],
                                        custom_dm = custom_dm)


    # if standard GLM model or RF model that uses GLM betas
    elif (model2fit == 'glmsingle') or \
        ((model2fit == 'somaRF') and (Visuomotor_data.params['fitting']['soma']['somaRF']['beta_model'] == 'glmsingle')):

        # load data model
        data_model = GLMsingle_Model(Visuomotor_data)

        if py_cmd == 'fit_glmsingle':

            ## loop over all subjects 
            for pp in Visuomotor_data.sj_num:
                data_model.fit_data(pp)

        elif py_cmd == 'stats_glmsingle':
             
            ## loop over all subjects 
            for pp in Visuomotor_data.sj_num:
                data_model.compute_roi_stats(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'])

        elif py_cmd == 'fit_RF':

            print('Not implemented')





