import re, os
import glob, yaml
import sys
import os.path as op

import argparse

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

from preproc_mridata import MRIData


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

CLI.add_argument("--exclude_sj",  # name on the CLI - drop the `--` for positional/required parameters
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
sj = args.subject
system_dir = args.system
exclude_sj = args.exclude_sj
py_cmd = args.cmd

## Load MRI object
Visuomotor_data = MRIData(params, sj, 
                        base_dir = system_dir, 
                        exclude_sj = exclude_sj)

## Run specific command

if py_cmd == 'postfmriprep':

    Visuomotor_data.post_fmriprep_proc()
