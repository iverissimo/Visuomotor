from __future__ import division
from exptools.core.trial import Trial

from psychopy import visual, core, misc, event
import numpy as np
from numpy.random import random, shuffle #we only need these two commands from this lib
# from IPython import embed as shell
from math import *
import random, sys

# sys.path.append( 'exp_tools' )
# sys.path.append( os.environ['EXPERIMENT_HOME'] )

from stim import PRFStim

class PRFTrial(Trial):
    def __init__(self, parameters = {}, phase_durations = [], session = None, screen = None, tracker = None):
        super(PRFTrial, self).__init__(parameters = parameters, phase_durations = phase_durations, session = session, screen = screen, tracker = tracker)
        
        self.stim = PRFStim(self.screen, self, self.session, orientation = self.parameters['orientation'])
        
        this_instruction_string = '\t\t\t  Index\t\t/\tMiddle:\n\nColor\t\t-\tB\t\t/\t\tW'# self.parameters['task_instruction']
        self.instruction = visual.TextStim(self.screen, text = this_instruction_string, font = 'Helvetica Neue', pos = (0, 0), italic = True, height = 30, alignHoriz = 'center')
        self.instruction.setSize((1200,50))

        self.run_time = 0.0
        self.instruct_time = self.t_time = self.fix_time = self.stimulus_time = self.post_stimulus_time = 0.0
        self.instruct_sound_played = False

        
    def draw(self):
        """docstring for draw"""
        old_color = self.session.fixation.color[0]
        self.session.fixation.color = [self.session.fix_task_frame_values[self.session.frame_nr], self.session.fix_task_frame_values[self.session.frame_nr], self.session.fix_task_frame_values[self.session.frame_nr]]
        if (old_color != self.session.fixation.color[0]) and hasattr(self.session, 'scanner_start_time'):
            self.session.transition_list.append([self.session.clock.getTime() - self.session.scanner_start_time, self.session.fixation.color[0]])

        if self.phase == 0:
            if self.ID == 0:
                self.instruction.draw()
        elif self.phase == 2:
            self.stim.draw(phase = np.max([(self.phase_times[self.phase] - self.phase_times[self.phase-1]) / self.stim.period,0]))

        self.session.fixation_outer_rim.draw()
        self.session.fixation_rim.draw()
        self.session.fixation.draw()

        super(PRFTrial, self).draw()

    def event(self):
        for ev in event.getKeys():
            if len(ev) > 0:
                if ev in ['esc', 'escape']:
                    self.events.append([-99,self.session.clock.getTime()-self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                    print 'run canceled by user'
                # it handles both numeric and lettering modes 
                elif ev == ' ':
                    self.events.append([0,self.session.clock.getTime()-self.start_time])
                    if self.phase == 0:
                        self.phase_forward()
                    else:
                        self.events.append([-99,self.session.clock.getTime()-self.start_time])
                        self.stopped = True
                        print 'trial canceled by user'
                elif ev == 't': # TR pulse
                    self.events.append([99,self.session.clock.getTime()-self.start_time])
                    if (self.phase == 0) and (self.ID == 0): 
                        # first trial, first phase receives the first 't' of the experiment
                        self.session.scanner_start_time = self.session.clock.getTime()
                    if (self.phase == 0) + (self.phase==1):
                        self.phase_forward()

                event_msg = 'trial ' + str(self.ID) + ' key: ' + str(ev) + ' at time: ' + str(self.session.clock.getTime())
                self.events.append(event_msg)
                print(event_msg + ' ' + str(self.phase))
        
            super(PRFTrial, self).key_event( ev )

    # def run(self, ID = 0):
    #     self.ID = ID
    #     super(PRFTrial, self).run()
        
    #     while not self.stopped:
    #         self.run_time = self.session.clock.getTime() - self.start_time
    #         # Only in trial 1, phase 0 represents the instruction period.
    #         # After the first trial, this phase is skipped immediately
    #         if self.phase == 0:
    #             self.instruct_time = self.session.clock.getTime()
    #             if self.ID != 0:
    #                 self.phase_forward()
    #         # In phase 1, we wait for the scanner pulse (t)
    #         if self.phase == 1:
    #             self.t_time = self.session.clock.getTime()
    #             if self.session.scanner == 'n':
    #                 self.phase_forward()
    #         # In phase 2, the stimulus is presented
    #         if self.phase == 2:
    #             self.stimulus_time = self.session.clock.getTime()
    #             if ( self.stimulus_time - self.t_time ) > self.stim.period:
    #                 self.phase_forward()
    #         # Phase 3 reflects the ITI
    #         if self.phase == 3:
    #             self.post_stimulus_time = self.session.clock.getTime()
    #             if ( self.post_stimulus_time  - self.stimulus_time ) > self.phase_durations[2]:
    #                 self.stopped = True
        
    #         # events and draw
    #         self.event()
    #         self.draw()
    
    #     self.stop()
        
