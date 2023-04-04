import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob
from nilearn import surface
import nibabel as nib

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from statsmodels.stats.multitest import fdrcorrection

import datetime

from visuomotor_utils import COM

import scipy

import cortex
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm


class Model:

    def __init__(self, MRIObj = None, outputdir = None):

        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in preproc_mridata
        outputdir: str
            absolute path to save fits
            
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        self.outputdir = outputdir 


    def get_proc_file_list(self, participant, file_ext = 'sg_psc.func.gii'):

        """
        Helper function to get list of bold file names
        to then be loaded and used

        Parameters
        ----------
        participant: str
            participant ID
        file_ext: str
            bold file extension identifier 
        """

        ## get list of possible input paths
        # (sessions)
        input_list = op.join(self.proc_file_pth, 'sub-{sj}'.format(sj = participant))

        # list with absolute file names to be fitted
        bold_filelist = [op.join(input_list, file) for file in os.listdir(input_list) if file.endswith(file_ext)]

        return bold_filelist

    def get_run_list(self, file_list):

        """
        Helper function to get unique run number from list of strings (filenames)

        Parameters
        ----------
        file_list: list
            list with file names
        """
        return np.unique([int(re.findall(r'run-\d{1,3}', op.split(input_name)[-1])[0][4:]) for input_name in file_list])
    
    def load_data4fitting(self, file_list, average = False):

        """
        Helper function to load data for fitting

        Parameters
        ----------
        file_list: list
            list with file names
        average: bool
            if we return average across files or all runs stacked
        """

        # get run IDs
        run_num_list = self.get_run_list(file_list)

        ## load data of all runs
        all_data = []
        for run_id in run_num_list:
            
            run_data = []
            for hemi in self.MRIObj.hemispheres:
                
                hemi_file = [file for file in file_list if 'run-{r}'.format(r=str(run_id).zfill(2)) in file and hemi in file][0]
                print('loading %s' %hemi_file)    
                run_data.append(np.array(surface.load_surf_data(hemi_file)))
                
            all_data.append(np.vstack(run_data)) 

        # if we want to average 
        if average:
            return np.nanmean(all_data, axis = 0) # [vertex, TR]
        else:
            return all_data # [runs, vertex, TR]

    def piecewise_linear(self, x, x0, y0, k1, k2):

        """
        Calculate piecewise-defined function. 
        Restricted to two linear segments, that intersect at x0
        
        Parameters
        ----------
        x : arr
            input array of x-axis coordinates
        x0 : float
            x-position where segments intersect
        y0 : float
            y-position where segments intersect
        k1 : float
            slope for first segment
        k2 : float
            slope for second segment
        """ 

        return np.piecewise(x, [x < x0], # x >= x0],
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

    def fit_piecewise(self, x_data = None, y_data = None, 
                            x0 = .1, y0 = 4, k1 = 2, k2 = -2, 
                            bounds = (-np.inf, np.inf), sigma = None, abs_sigma = False):

        """
        Fit piecewise-defined function on data.
        [Restricted to two linear segments, that intersect at x0]
        
        Parameters
        ----------
        x_data : arr
            x input values to fit
        y_data : arr
            y input values to fit
        bounds: tuple
            Lower and upper bounds on parameters - see scipy optimize bounds for more info
        x0 : float
            initial guess - x-position where segments intersect
        y0 : float
            initial guess - y-position where segments intersect
        k1 : float
            initial guess - slope for first segment
        k2 : float
            initial guess - slope for second segment
        sigma: arr
            1D array of uncertainty in ydata, should contain values of standard deviations of errors in ydata
        """ 

        popt_piecewise, pcov = scipy.optimize.curve_fit(self.piecewise_linear, 
                                                x_data, y_data, 
                                                p0 = [x0, y0, k1, k2],
                                               bounds = bounds,
                                               sigma = sigma, absolute_sigma = abs_sigma)
        
        # get R2 of fit
        pred_arr = self.piecewise_linear(x_data, *popt_piecewise)
        r2 = np.nan_to_num(1 - (np.nansum((y_data - pred_arr)**2, axis=0)/ np.nansum(((y_data - np.mean(y_data))**2), axis=0)))
        
        return popt_piecewise, pcov, r2

    def linear_func(self, dm = None, betas = None):
        return dm.dot(betas)

    def fit_linear(self, data, dm, add_intercept = True):

        """
        helper func to fit linear function on data.
        """
        
        # check DM shape
        if len(dm.shape) == 1: 
            dm = dm.reshape(-1,1)
        elif dm.shape[-1] > dm.shape[0]: # we want [npoints, betas]
            dm = dm.T

        # stack intercept
        if add_intercept: 
            dm = np.hstack((dm, np.ones(dm.shape)))

        # actually fit
        betas = np.linalg.lstsq(dm, data, rcond = -1)[0]
        
        # get R2 of fit
        pred_arr = self.linear_func(dm = dm, betas = betas)
        r2 = np.nan_to_num(1 - (np.nansum((data - pred_arr)**2, axis=0)/ np.nansum(((data - np.mean(data))**2), axis=0)))

        return betas, dm, r2

    def fdr_correct(self, alpha_fdr = 0.01, p_values = None, stat_values = None):

        """
        Calculate False Discovery Rate for a given statistic and p-values
        returns corrected statistic (masking non-significant values with nan)

        Parameters
        ----------
        alpha_fdr : float
            alpha level [default 0.01 (1%)] 
        p_values: arr
            p-values
        stat_values: arr
            statistic value to be corrected      
        """

        # The fdrcorrection function already returns a "mask"
        fdr_mask = fdrcorrection(p_values, alpha=alpha_fdr)[0]
        fdr_mask = fdr_mask.reshape(p_values.shape)

        fdr_stat_vals = stat_values.copy() 
        fdr_stat_vals[~fdr_mask] = np.nan

        return fdr_stat_vals

    def get_Fstat(self, data, rmodel_pred = None, fmodel_pred = None, num_regs = None):

        """
        Calculate goodness of fit F stat for full vs simple model
        
        Parameters
        ----------
        data : arr
            data
        rmodel_pred: arr
            reduced model prediction (model with intercept only)
        fmodel_pred: arr
            full model prediction (all regressors + intercept) 
        num_regs: int
            number of regressors of full model       
        """

        # calculate sum of squared residuals
        rss_1 = np.nansum((data - rmodel_pred) ** 2, axis = -1) # simple model
        rss_2 = np.nansum((data - fmodel_pred) ** 2, axis = -1) # complex model

        # F-statistic
        # model 1 is nested in model 2, so F will always be positive
        F_stat = ((rss_1 - rss_2)/(num_regs - 1))/(rss_2/(data.shape[-1] - num_regs))

        # get corresponing p values
        p_values_F = 1 - scipy.stats.f.cdf(F_stat, dfn = num_regs - 1, dfd = data.shape[-1] - num_regs)

        return F_stat, p_values_F

    def calc_chisq(self, data, pred, error = None):

        """
        Calculate model fit  chi-square
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        """ 
    
        # residuals
        resid = data - pred 
        
        # if not providing uncertainty in ydata
        if error is None:
            error = np.ones(len(data))
        
        chisq = sum((resid/ error) ** 2)
        
        return chisq

    def calc_reduced_chisq(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Reduced chi-square
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        return self.calc_chisq(data, pred, error = error) / (len(data) - n_params)

    def calc_AIC(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Akaike Information Criterion,
        which measures of the relative quality for a fit, 
        trying to balance quality of fit with the number of variable parameters used in the fit
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        chisq = self.calc_chisq(data, pred, error = error)
        n_obs = len(data) # number of data points
        
        return n_obs * np.log(chisq/n_obs) + 2 * n_params
    
    def calc_BIC(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Bayesian information criterion,
        which measures of the relative quality for a fit, 
        trying to balance quality of fit with the number of variable parameters used in the fit
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        chisq = self.calc_chisq(data, pred, error = error)
        n_obs = len(data) # number of data points
        
        return n_obs * np.log(chisq/n_obs) + np.log(n_obs) * n_params

    def leave_one_out(self, input_list):

        """ make list of lists, by leaving one out

        Parameters
        ----------
        input_list : list/arr
            list of items

        Outputs
        -------
        out_lists : list/arr
            list of lists, with each element
            of the input_list left out of the returned lists once, in order

        
        """

        out_lists = []
        for x in input_list:
            out_lists.append([y for y in input_list if y != x])

        return out_lists

    def normalize(self, M):
        """
        normalize data array
        """
        return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))


    def corr2zFischer(self, rho1, alpha = .05, n_obs = None):

        """
        Convert correlation coefficients with Fisher's z-transformation
        and calculate confidence intervals

        Parameters
        ----------
        rho1: arr
            correlation coeficient(s) 
        alpha: float
            alpha level for confidence interval
        n_obs: arr
            number of observations used to calculate the correlation coefficient
        """

        # correlation coefficients r are transformed using Fisher z transformation
        z1 = .5 * np.log((1+rho1)/(1-rho1)) # np.arctanh(rho1)

        ## calculate confidence intervals (95%)
        se = 1/np.sqrt(n_obs - 3)

        z = scipy.stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = z1-z*se, z1+z*se

        # reverse the transformation
        lo, hi = np.tanh((lo_z, hi_z))

        return z1, se, lo, hi

    