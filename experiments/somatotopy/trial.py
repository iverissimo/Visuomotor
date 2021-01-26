from exptools.core.trial import Trial
import os
import exptools
import json
from psychopy import logging, visual, event
import numpy as np


class MSTrial(Trial):

    def __init__(self, parameters = {}, phase_durations = [], session = None, screen = None, tracker = None):

        super(
            MSTrial,
            self).__init__(
            phase_durations=phase_durations,
            session=session,
            screen=screen,
            parameters=parameters,
            tracker=tracker)

        self.movie_stim = self.session.movie_stims[self.parameters['stimulus']]
        size_fixation_pix = self.session.deg2pix(self.session.size_fixation_deg)

        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='raisedCos',
                                           size=size_fixation_pix,
                                           texRes=512,
                                           color='white',
                                           sf=0,
                                           maskParams={'fringeWidth': 0.4})

    def draw(self, *args, **kwargs):

        # if self.phase in  (0,1,3):
        if self.phase == 2:
            self.movie_stim.draw()
        self.fixation.draw()


        super(MSTrial, self).draw()

    def event(self):

        for ev in event.getKeys():
            if len(ev) > 0:
                if ev in ['esc', 'escape', 'q']:
                    self.events.append(
                        [-99, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                    print 'run canceled by user'
                if ev in ['space', ' ', 't']:
                    if (self.phase == 0) and (self.ID == 0):
                        self.phase_forward()

            super(MSTrial, self).key_event(ev)
