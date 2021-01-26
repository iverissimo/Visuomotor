from __future__ import division

from exptools.core.session import EyelinkSession
from trial import PRFTrial

from psychopy import clock
from psychopy.visual import ImageStim, MovieStim, GratingStim
import numpy as np
import os
import exptools
import json
import glob



class PRFSession(EyelinkSession):
    def __init__(self, *args, **kwargs):

        super(PRFSession, self).__init__(*args, **kwargs)

        self.response_button_signs = dict(zip(self.config.get('buttons', 'keys'), range(len(self.config.get('buttons', 'keys')))))

         # Set arguments from config file or kwargs
        for argument in ['PRF_ITI_in_TR', 'TR', 'task_rate', 'task_rate_offset',
                         'vertical_bar_pass_in_TR', 'horizontal_bar_pass_in_TR', 'empty_bar_pass_in_TR']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)
        for argument in ['mask_type', 'vertical_stim_size', 'horizontal_stim_size',
                         'bar_width_ratio', 'num_elements', 'color_ratio', 'element_lifetime',
                         'stim_present_booleans', 'stim_direction_indices',
                         'fixation_outer_rim_size', 'fixation_rim_size', 'fixation_size',
                         'fast_speed', 'slow_speed', 'element_size', 'element_spatial_frequency']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)

        # trials can be set up independently of the staircases that support their parameters
        self.create_trials(**kwargs)
        self.stopped = False

        self.transition_list = []


    def create_trials(self, **kwargs):
        """docstring for create_trials(self):"""

        self.directions = np.linspace(0, 2.0 * np.pi, 8, endpoint=False)
        # Set arguments from config file or kwargs

        if self.mask_type == 0:
            self.horizontal_stim_size = self.size[1]/self.size[0]


        # orientations, bar moves towards:
        # 0: S      3: NW   6: E
        # 1: SW     4: N    7: SE
        # 2: W      5: NE

        self.bar_pass_durations = []
        for i in range(len(self.stim_present_booleans)):
            if self.stim_present_booleans[i] == 0:
                self.bar_pass_durations.append(
                    self.empty_bar_pass_in_TR * self.TR)
            else:
                if self.stim_direction_indices[i] in (2, 6):  # EW-WE:
                    self.bar_pass_durations.append(
                        self.horizontal_bar_pass_in_TR * self.TR)
                elif self.stim_direction_indices[i] in (0, 4):  # NS-SN:
                    self.bar_pass_durations.append(
                        self.vertical_bar_pass_in_TR * self.TR)

        # nostim-top-left-bottom-right-nostim-top-left-bottom-right-nostim
        # nostim-bottom-left-nostim-right-top-nostim
        self.trial_array = np.array(
            [[self.stim_direction_indices[i], self.stim_present_booleans[i]] for i in range(len(self.stim_present_booleans))])

        self.RG_color = 1/self.color_ratio
        self.BY_color = 1

        self.phase_durations = np.array([[
            -0.001,  # instruct time
            180.0,  # wait for scan pulse
            self.bar_pass_durations[i],
            self.PRF_ITI_in_TR * self.TR] for i in range(len(self.stim_present_booleans))])    # ITI

        self.total_duration = np.sum(np.array(self.phase_durations))
        self.phase_durations[0,0] = 1800

        # fixation point
        self.fixation_outer_rim = GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_outer_rim_size),
                                                   pos=np.array((self.x_offset, 0.0)), color=self.background_color, maskParams={'fringeWidth': 0.4})
        self.fixation_rim = GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_rim_size),
                                             pos=np.array((self.x_offset, 0.0)), color=(-1.0, -1.0, -1.0), maskParams={'fringeWidth': 0.4})
        self.fixation = GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_size),
                                         pos=np.array((self.x_offset, 0.0)), color=self.background_color, opacity=1.0, maskParams={'fringeWidth': 0.4})

        # mask
        if self.mask_type == 1:
            draw_screen_space = [self.screen_pix_size[0]*self.horizontal_stim_size,
                                 self.screen_pix_size[1]*self.vertical_stim_size]
            mask = np.ones(
                (self.screen_pix_size[1], self.screen_pix_size[0]))*-1
            x_edge = int(
                np.round((self.screen_pix_size[0]-draw_screen_space[0])/2))
            y_edge = int(
                np.round((self.screen_pix_size[1]-draw_screen_space[1])/2))
            if x_edge > 0:
                mask[:, :x_edge] = 1
                mask[:, -x_edge:] = 1
            if y_edge > 0:
                mask[-y_edge:, :] = 1
                mask[:y_edge, :] = 1
            import scipy
            mask = scipy.ndimage.filters.gaussian_filter(mask, 5)
            self.mask_stim = GratingStim(self.screen, mask=mask, tex=None, size=[self.screen_pix_size[0], self.screen_pix_size[1]],
                                              pos=np.array((self.x_offset, 0.0)), color=self.screen.background_color)
        elif self.mask_type == 0:
            mask = filters.makeMask(matrixSize=self.screen_pix_size[0], shape='raisedCosine', radius=self.vertical_stim_size *
                                    self.screen_pix_size[1]/self.screen_pix_size[0]/2, center=(0.0, 0.0), range=[1, -1], fringeWidth=0.1)
            self.mask_stim = GratingStim(self.screen, mask=mask, tex=None, 
                size=[self.screen_pix_size[0]*2, self.screen_pix_size[0]*2], 
                pos=np.array((self.x_offset, 0.0)), 
                color=self.screen.background_color)

        # fixation task timing
        self.fix_task_frame_values = self._get_frame_values(framerate=self.framerate, 
                                trial_duration=self.total_duration, 
    
                                safety_margin=3000.0)


    def run(self):
        """docstring for fname"""
        # cycle through trials
        for i in range(len(self.trial_array)):
            # prepare the parameters of the following trial based on the shuffled trial array
            this_trial_parameters = {}
            this_trial_parameters['stim_duration'] = self.phase_durations[i, -2]
            this_trial_parameters['orientation'] = self.directions[self.trial_array[i, 0]]
            this_trial_parameters['stim_bool'] = self.trial_array[i, 1]

            # these_phase_durations = self.phase_durations.copy()
            these_phase_durations = self.phase_durations[i]

            this_trial = PRFTrial(parameters=this_trial_parameters, phase_durations=these_phase_durations,
                                  session=self, screen=self.screen, tracker=self.tracker)

            # run the prepared trial
            this_trial.run(ID=i)
            if self.stopped == True:
                break
        self.close()

    def close(self):
        np.savetxt(self.output_file + '_trans.tsv', np.array(self.transition_list), delimiter='\t', fmt='%4.4f')
        super(PRFSession, self).close()

    def _get_frame_values(self,
                          framerate=60,
                          trial_duration=3000,
                          min_value=1,
                          exp_scale=1,
                          values=[-1, 1],
                          safety_margin=None):

        if safety_margin is None:
            safety_margin = 5

        n_values = len(values)

        total_duration = trial_duration + safety_margin
        total_n_frames = total_duration * framerate

        result = np.zeros(int(total_n_frames))

        n_samples = np.ceil(total_duration * 2 /
                            (exp_scale + min_value)).astype(int)
        durations = np.random.exponential(exp_scale, n_samples) + min_value

        frame_times = np.linspace(
            0, total_duration, total_n_frames, endpoint=False)

        first_index = np.random.randint(n_values)

        result[frame_times < durations[0]] = values[first_index]

        for ix, c in enumerate(np.cumsum(durations)):
            result[frame_times > c] = values[(first_index + ix) % n_values]

        return result
