from __future__ import division
from psychopy import visual, core, misc, event
try:
    from psychopy.visual import filters
except ImportError:
    from psychopy import filters
import numpy as np
from scipy.signal import convolve2d
from IPython import embed as shell
from math import *
import random, sys

# sys.path.append( 'exp_tools' )
# sys.path.append( os.environ['EXPERIMENT_HOME'] )


class PRFStim(object):
    def __init__(self, screen, trial, session, orientation):
        # parameters

        self.trial = trial
        self.session = session
        self.screen = screen
        self.orientation = orientation    # convert to radians immediately, and use to calculate rotation matrix
        self.rotation_matrix = np.matrix([[cos(self.orientation), -sin(self.orientation)],[sin(self.orientation), cos(self.orientation)]])
        # self.refresh_frequency = session.redraws_per_TR / session.standard_parameters['TR

        self.RG_color=session.RG_color
        self.BY_color=session.BY_color

        self.fast_speed = session.fast_speed
        self.slow_speed = session.slow_speed
        
        # print(self.orientation)

        if self.orientation in self.session.directions[[0,4]]:
            self.bar_width = self.screen.size[1] * session.vertical_stim_size * session.bar_width_ratio
            self.bar_length = self.screen.size[0]
        elif self.orientation in self.session.directions[[2,6]]:
            self.bar_width = self.screen.size[0] * session.horizontal_stim_size * session.bar_width_ratio
            self.bar_length = self.screen.size[1]
        self.num_elements = session.num_elements

        # change n_elements, sizes and bar width ratio for horizontal / vertical passes
        if self.trial.parameters['orientation'] in [0,np.pi]: # these are the horizontal passes (e.g. top-bottom)
            self.size_pix = [self.screen.size[1]*session.vertical_stim_size,
                            self.screen.size[0]*session.horizontal_stim_size]
        else: # vertical bar passes:
            self.size_pix = [self.screen.size[0]*session.horizontal_stim_size,
                            self.screen.size[1]*session.vertical_stim_size]
        
        self.period = self.trial.parameters['stim_duration']

        self.full_width = self.size_pix[0] + self.bar_width + self.session.element_size
        self.midpoint = 0

        # this is for determining ecc, which we make dependent on largest screen dimension

        self.phase = 0
        # bookkeeping variables
        self.eccentricity_bin = -1
        self.redraws = 0
        self.frames = 0

        # psychopy stimuli
        self.populate_stimulus()

        # create the stimulus
        self.session.element_array = visual.ElementArrayStim(screen, nElements = self.session.num_elements, sizes = self.element_sizes, sfs = self.element_sfs, 
            xys = self.element_positions, colors = self.colors, colorSpace = 'rgb') 


    def convert_sample(self,in_sample):
        return 1 - (1/(np.e**in_sample+1))
    
    def populate_stimulus(self):

        RG_ratio = 0.5
        BY_ratio = 0.5
        fast_ratio = 0.5
        slow_ratio = 0.5

        # set the default colors
        self.colors = np.ones((self.num_elements,3)) * 0.5
        self.fix_gray_value = self.session.background_color

        # and change them if a pulse is wanted

        # Now set the actual stimulus parameters
        self.colors = np.concatenate((np.ones((int(np.round(self.num_elements*RG_ratio/2.0)),3)) * np.array([1,-1,0]) * self.RG_color,  # red/green - red
                                    np.ones((int(np.round(self.num_elements*RG_ratio/2.0)),3)) * np.array([-1,1,0]) * self.RG_color,  # red/green - green
                                    np.ones((int(np.round(self.num_elements*BY_ratio/2.0)),3)) * np.array([-1,-1,1]) * self.BY_color,  # blue/yellow - blue
                                    np.ones((int(np.round(self.num_elements*BY_ratio/2.0)),3)) * np.array([1,1,-1]) * self.BY_color))  # blue/yellow - yellow

    
        np.random.shuffle(self.colors)

        # but do update all other stim parameters (regardless of pulse)
        self.element_speeds = np.concatenate((np.ones(int(np.round(self.num_elements*fast_ratio))) * self.session.fast_speed,
                                            np.ones(int(np.round(self.num_elements*slow_ratio))) * self.session.slow_speed))
        np.random.shuffle(self.element_speeds)

        self.element_positions = np.random.rand(self.num_elements, 2) * np.array([self.bar_length, self.bar_width]) - np.array([self.bar_length/2.0, self.bar_width/2.0])
        # self.element_sfs = np.ones((self.num_elements)) * self.session.element_spatial_frequency']
        self.element_sfs = np.random.rand(self.num_elements)*7+0.25
        self.element_sizes = np.ones((self.num_elements)) * self.session.element_size
        self.element_phases = np.zeros(self.num_elements)
        self.element_orientations = np.random.rand(self.num_elements) * 720.0 - 360.0

        self.lifetimes = np.random.rand(self.num_elements) * self.session.element_lifetime

    def draw(self, phase = 0):

        self.phase = phase
        self.frames += 1

        to_be_redrawn = self.lifetimes < phase
        self.element_positions[to_be_redrawn] = np.random.rand(to_be_redrawn.sum(), 2) \
                * np.array([self.bar_length, self.bar_width]) \
                - np.array([self.bar_length/2.0, self.bar_width/2.0])     
        self.lifetimes[to_be_redrawn] += np.random.rand(to_be_redrawn.sum()) * self.session.element_lifetime    

        # define midpoint
        self.midpoint = phase * self.full_width - 0.5 * self.full_width #+ self.session.x_offset']

        self.session.element_array.setSfs(self.element_sfs)
        self.session.element_array.setSizes(self.element_sizes)
        self.session.element_array.setColors(self.colors)
        self.session.element_array.setOris(self.element_orientations)
        if self.trial.parameters['orientation'] == np.pi/2:
            draw_midpoint = self.midpoint - self.session.x_offset
        elif self.trial.parameters['orientation'] == 3*(np.pi/2):
            draw_midpoint = self.midpoint + self.session.x_offset
        else:
            draw_midpoint = self.midpoint 
        self.session.element_array.setXYs(np.array(np.matrix(self.element_positions + np.array([0, -draw_midpoint])) * self.rotation_matrix)) 
            
        self.session.element_array.setPhases(self.element_speeds * self.phase * self.period + self.element_phases)

        if self.trial.parameters['stim_bool'] == 1:
            self.session.element_array.draw()
        
        self.session.mask_stim.draw()      
        
