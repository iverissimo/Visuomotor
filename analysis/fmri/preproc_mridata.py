
import numpy as np
import os.path as op
import os
import yaml
import glob
import re
from visuomotor_utils import crop_gii, dc_filter_gii, highpass_gii, psc_gii, median_gii, mean_gii, leave_one_out

class VisuomotorData:
    
    """VisuomotorData
    Class that loads relevant paths and settings for Visuomotor data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], base_dir = 'local', wf_dir = None):
        
        """__init__
        constructor for class, takes experiment params and subject num as input
        
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str/list/arr
            participant number(s)
        exclude_sj: list/arr
            list with subject numbers to exclude
            
        """
        
        # set params
        
        if isinstance(params, str):
            # load settings from yaml
            with open(params, 'r') as f_in:
                self.params = yaml.safe_load(f_in)
        else:
            self.params = params

        # relevant tasks
        self.tasks = self.params['general']['tasks']

        # timing
        self.TR = self.params['general']['TR']
            
        # excluded participants
        self.exclude_sj = exclude_sj
        if len(self.exclude_sj)>0:
            self.exclude_sj = [str(val).zfill(2) for val in exclude_sj]

        ## set some paths
        # which machine we run the data
        self.base_dir = base_dir
        
        # project root folder
        self.proj_root_pth = self.params['general']['paths'][self.base_dir]['root']

        # in case we are computing things in a different worflow dir
        # useful when fitting models in /scratch node
        if wf_dir is not None:
            self.proj_root_pth = wf_dir
        
        # sourcedata dir
        self.sourcedata_pth = op.join(self.proj_root_pth,'sourcedata')
        
        # derivatives dir
        self.derivatives_pth = op.join(self.proj_root_pth,'derivatives')
        
        ## set sj number
        if sj_num in ['group', 'all']: # if we want all participants in sourcedata folder
            sj_num = [op.split(val)[-1].zfill(2)[4:] for val in glob.glob(op.join(self.sourcedata_pth, 'sub-*'))]
            self.sj_num = [val for val in sj_num if val not in self.exclude_sj ]
        
        elif isinstance(sj_num, list) or isinstance(sj_num, np.ndarray): # if we provide list of sj numbers
            self.sj_num = [str(s).zfill(2) for s in sj_num if str(s).zfill(3) not in self.exclude_sj ]
        
        else:
            self.sj_num = [str(sj_num).zfill(2)] # if only one participant, put in list to make life easier later
        
        ## get session number (can be more than one)
        self.session = {}
        for s in self.sj_num:
            if wf_dir is not None:
                print('WARNING, working in a temp dir so sourcedata might not be there')
            else:
                self.session['sub-{sj}'.format(sj=s)] = [op.split(val)[-1] for val in glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=s), 'ses-*')) if 'anat' not in val] 
        

class BehData(VisuomotorData):

    """BehData
    Class that loads relevant paths and settings for behavioral data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], base_dir = None, wf_dir = None):  # initialize child class

        """ Initializes MRIData object. 
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str/list/arr
            participant number(s)
        exclude_sj: list/arr
            list with subject numbers to exclude
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj, base_dir = base_dir, wf_dir = wf_dir)


        ## some pRF params relevant for setting task
        self.pRF_bar_pass_direction = self.params['prf']['bar_pass_direction']
        self.pRF_bar_pass_in_TRs = self.params['prf']['bar_pass_in_TRs'] 
        self.pRF_ITI_in_TR = self.params['prf']['PRF_ITI_in_TR'] 

        ## some SOMA params relevant for setting task
        self.soma_trial_order = self.params['soma']['trial_order']
        self.soma_event_time_in_sec = self.params['soma']['event_dur_in_sec']


class MRIData(BehData):
    
    """MRIData
    Class that loads relevant paths and settings for (f)MRI data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], base_dir = None, wf_dir = None):  # initialize child class

        """ Initializes MRIData object. 
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str/list/arr
            participant number(s)
        exclude_sj: list/arr
            list with subject numbers to exclude
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj, base_dir = base_dir, wf_dir = wf_dir)

        ## some paths
        # path to freesurfer
        self.freesurfer_pth = op.join(self.derivatives_pth, 'freesurfer')
        
        # path to fmriprep
        self.fmriprep_pth = op.join(self.derivatives_pth, 'fmriprep')

        # path to postfmriprep
        self.postfmriprep_pth = op.join(self.derivatives_pth, 'post_fmriprep')

        ## some relevant params
        self.sj_space = self.params['processing']['space'] # subject space
        self.file_ext = self.params['processing']['file_ext'] # file extension
        self.crop_TR = self.params['processing']['crop_TR']
        self.hemispheres = ['hemi-L','hemi-R']
 

    def post_fmriprep_proc(self):

        """
        Run final processing steps on functional data (after fmriprep)
        """ 

        # loop over participants
        for pp in self.sj_num:

            # and over sessions (if more than one)
            for ses in self.session['sub-{sj}'.format(sj=pp)]:

                # get list of functional files to process, per task
                sub_fmriprep_pth = op.join(self.fmriprep_pth, 'sub-{sj}'.format(sj=pp), ses, 'func')

                for tsk in self.tasks:
                    print('Processing bold files from task-{t}'.format(t=tsk))

                    # bold files for participant and task
                    bold_files = [op.join(sub_fmriprep_pth,run) for run in os.listdir(sub_fmriprep_pth) if 'space-{sp}'.format(sp=self.sj_space) in run \
                        and 'task-{t}'.format(t=tsk) in run and run.endswith(self.file_ext)]

                    if not bold_files: # if list empty
                        print('Subject %s has no files for %s' %(pp, tsk))

                    else:
                        output_pth = op.join(self.postfmriprep_pth, self.sj_space, tsk, 'sub-{sj}'.format(sj=pp))
                        # if output path doesn't exist, create it
                        os.makedirs(output_pth, exist_ok = True)
                        print('saving files in %s'%output_pth)

                        if tsk == 'soma':
                            # copy raw files (to use for glm single)
                            [os.system('cp {og_file} {new_file}'. format(og_file = file, 
                                                                        new_file = op.join(output_pth, op.split(file)[-1]))) for file in bold_files]

                        ### crop files, if we want to
                        proc_files = [crop_gii(file,
                                                self.crop_TR[tsk],
                                                output_pth, 
                                                extension = self.file_ext) for file in bold_files]

                        ## filter
                        if self.params['processing']['filter'][tsk] == 'dc':
                            # high pass filter all runs (with discrete cosine set)
                            filt_files = [dc_filter_gii(file, 
                                                        output_pth, 
                                                        extension = self.file_ext,
                                                        first_modes_to_remove = self.params['processing']['first_modes_to_remove'])[-1] for file in proc_files]

                        elif self.params['processing']['filter'][tsk] == 'sg':
                            # high pass filter all runs (savgoy-golay)
                            filt_files = [highpass_gii(file,
                                                        self.params['processing']['sg_filt_polyorder'],
                                                        self.params['processing']['sg_filt_deriv'],
                                                        self.params['processing']['sg_filt_window_length'],
                                                        output_pth, 
                                                        extension = self.file_ext)[-1] for file in proc_files]

                        ## PSC
                        psc_files = [psc_gii(file, 
                                            output_pth, 
                                            method = 'mean', 
                                            extension = self.file_ext)[-1] for file in filt_files]

                        ## make mean/median file or LOO file
                        # depending on what we're going to fit
                        output_pth = op.join(output_pth, self.params['fitting'][tsk]['type'])
                        # if output path doesn't exist, create it
                        os.makedirs(output_pth, exist_ok = True)

                        if self.params['fitting'][tsk]['type'] == 'loo_run':

                            for field in self.hemispheres:

                                hemi = [h for h in psc_files if field in h and 'run-leave' not in h ]  #we don't want to average median run if already in original dir

                                # make list of run numbers
                                run_numbers = [re.findall('run-\d{2}', file)[0] for file in hemi]

                                # subdivide files into lists where one run is left out
                                loo_lists = leave_one_out(hemi) 

                                for r,ll in enumerate(loo_lists):
                                    print('averaging %s'%str(ll))
                                    
                                    # set name for median run (now numpy array)
                                    med_file = op.join(output_pth, re.sub('run-\d{2}_', 'run-leave_%s_out_'%(run_numbers[r][-2:]), op.split(ll[0])[-1]))
                                
                                    # if file doesn't exist
                                    if not op.exists(med_file):
                                        med_gii = mean_gii(ll, output_pth,
                                                                run_name='leave_%s_out'%(run_numbers[r][-2:]))  # create it
                                        print('computed %s' % (med_gii))
                                    else:
                                        print('file %s already exists, skipping' % (med_file))

                        else: # assumes fit type median or mean

                            for field in self.hemispheres:

                                hemi = [h for h in psc_files if field in h and 'median' not in h and 'mean' not in h]  #we don't want to average median run if already in original dir

                                # set name for median run (now numpy array)
                                med_file = op.join(output_pth, re.sub('run-\d{2}_', 
                                                                'run-{ft}_'.format(ft = self.params['fitting'][tsk]['type']), op.split(hemi[0])[-1]))

                                # if file doesn't exist
                                if not op.exists(med_file):
                                    
                                    if self.params['fitting'][tsk]['type'] == 'median':
                                        median_gii(hemi, output_pth)  # create it
                                    elif self.params['fitting'][tsk]['type'] == 'mean':
                                        mean_gii(hemi, output_pth)  # create it
                                    
                                    print('computed %s' % (med_file))
                                else:
                                    print('file % already exists, skipping' %med_file)


