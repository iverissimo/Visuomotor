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

from viewer import somaViewer, pRFViewer

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
                help = 'What plot to make?\n Options: COM_maps, glmsing_tc, glmsing_hex  '
                )

CLI.add_argument("--model",
                #nargs="*",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'What model to use?\n Options: glm, glmsingle, somaRF, gauss '
                )

CLI.add_argument("--task",
                type=str,  # any type/callable can be used here
                default = 'soma',
                help = 'What task we want to run? pRF vs soma [Default]'
                )

CLI.add_argument("--run_type", 
                default = 'mean_run',
                help="Type of run to fit (mean_run [default], 1, loo_run, ...)")

# parse the command line
args = CLI.parse_args()

# access CLI options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
system_dir = args.system
exclude_sj = args.exclude
py_cmd = args.cmd
model_name = args.model
task = args.task
run_type = args.run_type

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj)

# if running prf analysis
if task in ['pRF', 'prf']:

    # load data model
    data_model = prfModel(Visuomotor_data)

    # set model to be used
    data_model.model_type = model_name

    ## initialize plotter
    plotter = pRFViewer(data_model)

    if py_cmd == 'show_click':

        plotter.open_click_viewer(Visuomotor_data.sj_num[0], fit_type = run_type, 
                                            prf_model_name = model_name, 
                                            mask_arr = True, rsq_threshold = .2)
        
    elif py_cmd == 'prf_estimates':

        plotter.plot_prf_results(participant_list = Visuomotor_data.sj_num, 
                                fit_type = run_type, prf_model_name = model_name,
                                mask_arr = True, rsq_threshold = .2, iterative = True)

elif task == 'soma':

    # load data model
    data_model = GLM_Model(Visuomotor_data)

    # if we want nilearn dm or custom 
    custom_dm = True if Visuomotor_data.params['fitting']['soma']['use_nilearn_dm'] == False else False 

    if model_name == 'somaRF':
        ## make RF model object
        data_RFmodel = somaRF_Model(Visuomotor_data)
    else:
        data_RFmodel = None


    ## initialize plotter
    plotter = somaViewer(data_model)

    ## run command
    if py_cmd == 'COM_maps': # center of mass maps for standard GLM over average of runs

        region = ''
        while len(region) == 0:
            region = input("Which region to plot? (ex: face, upper_limb): ")

        ## loop over all subjects 
        for pp in Visuomotor_data.sj_num:
            plotter.plot_COM_maps(pp, region = region, custom_dm = custom_dm, fit_type = run_type,
                                        all_rois = Visuomotor_data.params['plotting']['soma']['roi2plot'],
                                        keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'])

    elif py_cmd == 'rsq':

        plotter.plot_rsq(Visuomotor_data.sj_num, fit_type = run_type,
                            all_rois = Visuomotor_data.params['plotting']['soma']['roi2plot'])

    elif py_cmd == 'beta_y_dist':

        plotter.plot_betas_over_y(Visuomotor_data.sj_num, fit_type = run_type,
                                keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'],
                                roi2plot_list = list(Visuomotor_data.params['plotting']['soma']['roi2plot'].keys()),
                                n_bins = 150, all_rois = Visuomotor_data.params['plotting']['soma']['roi2plot'], 
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'])
    
    elif py_cmd == 'COM_y_dist':

        plotter.plot_COM_over_y(Visuomotor_data.sj_num, fit_type = run_type,
                                keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'],
                                roi2plot_list = ['M1', 'S1', 'CS'], n_bins = 50,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'])

    elif py_cmd == 'RF_y_dist':

        plotter.plot_RF_over_y(Visuomotor_data.sj_num, fit_type = run_type, data_RFmodel = data_RFmodel,
                                keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'],
                                roi2plot_list = ['M1', 'S1', 'CS'], n_bins = 50,
                                all_regions = ['face', 'left_hand', 'right_hand', 'both_hand'])

    elif py_cmd == 'show_click':

        plotter.open_click_viewer(Visuomotor_data.sj_num[0], custom_dm = custom_dm, model2plot = model_name, 
                                            data_RFmodel = data_RFmodel, keep_b_evs = Visuomotor_data.params['fitting']['soma']['keep_b_evs'])



    

    