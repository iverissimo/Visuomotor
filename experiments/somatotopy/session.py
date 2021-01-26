from exptools.core.session import EyelinkSession
from trial import MSTrial
from psychopy import clock
from psychopy.visual import ImageStim, MovieStim
import numpy as np
import os
import exptools
import json
import glob


class MSSession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(MSSession, self).__init__(*args, **kwargs)

        for argument in ['size_fixation_deg', 'language']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)
        for argument in ['pre_post_fixation_time', 'inter_trial_fixation_time',  'stimulus_time']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)
        self.create_trials()

        self.stopped = False

    def create_trials(self):
        """creates trials by loading a list of jpg files from the img/ folder"""
        self.movies = [
        'eyebrows.avi',
        'eyes.avi',
        'mouth.avi',
        'tongue.avi',
        'lhand_fing1.avi', 
        'lhand_fing2.avi', 
        'lhand_fing3.avi', 
        'lhand_fing4.avi', 
        'lhand_fing5.avi', 
        'lleg.avi', 
        'eyebrows.avi',
        'eyes.avi',
        'mouth.avi',
        'tongue.avi',
        'bhand_fing1.avi', 
        'bhand_fing2.avi', 
        'bhand_fing3.avi', 
        'bhand_fing4.avi', 
        'bhand_fing5.avi', 
        'bleg.avi', 
        'tongue.avi',
        'mouth.avi',
        'eyes.avi',
        'eyebrows.avi',
        'rhand_fing1.avi', 
        'rhand_fing2.avi', 
        'rhand_fing3.avi', 
        'rhand_fing4.avi', 
        'rhand_fing5.avi', 
        'rleg.avi',
        'eyebrows.avi',
        'eyes.avi',
        'mouth.avi',
        'tongue.avi',
        'lhand_fing5.avi', 
        'lhand_fing4.avi', 
        'lhand_fing3.avi', 
        'lhand_fing2.avi', 
        'lhand_fing1.avi', 
        'lleg.avi', 
        'eyebrows.avi',
        'eyes.avi',
        'mouth.avi',
        'tongue.avi',
        'bhand_fing5.avi', 
        'bhand_fing4.avi', 
        'bhand_fing3.avi', 
        'bhand_fing2.avi', 
        'bhand_fing1.avi', 
        'bleg.avi', 
        'tongue.avi',
        'mouth.avi',
        'eyes.avi',
        'eyebrows.avi',
        'rhand_fing5.avi', 
        'rhand_fing4.avi', 
        'rhand_fing3.avi', 
        'rhand_fing2.avi', 
        'rhand_fing1.avi', 
        'rleg.avi']
        movie_files = [os.path.join(os.path.abspath(os.getcwd()), 'imgs', 'op',  '%s'%m) for m in self.movies]
        # print(movie_files)
        self.movie_stims = [MovieStim(self.screen, filename=imf.replace('.avi', '_small.avi'), size=self.screen.size) for imf in movie_files]
        # print(self.movie_stims)

        self.trial_order = np.arange(len(self.movie_stims))


    def run(self):
        """docstring for fname"""
        # cycle through trials

        for ti in np.arange(len(self.movie_stims)):

            parameters = {'stimulus': self.trial_order[ti], 'movie':self.movies[self.trial_order[ti]]}

            # parameters.update(self.config)
            if (ti == 0):
                phase_durations = [1800, self.pre_post_fixation_time, self.stimulus_time, self.inter_trial_fixation_time]
            elif (ti == len(self.movies)-1):
                phase_durations = [-0.001, self.inter_trial_fixation_time, self.stimulus_time, self.pre_post_fixation_time]
            else:
                phase_durations = [-0.001, self.inter_trial_fixation_time, self.stimulus_time, self.inter_trial_fixation_time]


            trial = MSTrial(phase_durations=phase_durations,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run(ID=ti)

            if self.stopped == True:
                break

        self.close()
