import re, os
import glob, yaml
import sys
import os.path as op

import argparse

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

from preproc_mridata import MRIData
from soma_model import GLMsingle_Model

from viewer import somaViewer

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

# parse the command line
args = CLI.parse_args()

# access CLI options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
system_dir = args.system
exclude_sj = args.exclude
py_cmd = args.cmd

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj)

## Run specific command
if py_cmd == 'postfmriprep':

    Visuomotor_data.post_fmriprep_proc()

elif py_cmd == 'fit_glmsingle':

    data_model = GLMsingle_Model(Visuomotor_data)

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.fit_data(pp)

elif py_cmd == 'stats_glmsingle':

    data_model = GLMsingle_Model(Visuomotor_data)

    ## loop over all subjects 
    for pp in Visuomotor_data.sj_num:
        data_model.compute_roi_stats(pp, z_threshold = Visuomotor_data.params['fitting']['soma']['z_threshold'])

elif py_cmd == 'plot_tc':

    data_model = GLMsingle_Model(Visuomotor_data)

    plotter = somaViewer(data_model)

    vertex = ''
    while len(vertex) == 0:
        vertex = input("Vertex number to plot?: ")

    plotter.plot_glmsingle_tc(Visuomotor_data.sj_num[0], int(vertex))

elif py_cmd == 'plot_hex':

    data_model = GLMsingle_Model(Visuomotor_data)

    plotter = somaViewer(data_model)

    plotter.plot_glmsingle_roi_betas(Visuomotor_data.sj_num)





