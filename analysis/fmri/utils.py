
# useful functions to use in other scripts

import os, re
import numpy as np

import shutil

import pandas as pd
import nibabel as nb

from nilearn import surface
import nistats

from scipy.signal import savgol_filter
import nipype.interfaces.freesurfer as fs

import cv2

from skimage import color
import cv2
from skimage.transform import rescale
from skimage.filters import threshold_triangle

from PIL import Image, ImageOps

import math
import cortex

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors

from scipy.stats import pearsonr, t, norm
from scipy import ndimage

from scipy.integrate import trapz
from scipy import stats
from scipy import misc


def median_gii(files,outdir):

    """ make median gii file (over runs)

    Parameters
    ----------
    files : List/arr
        list of absolute filenames to do median over
    outdir : str
        path to save new files
    

    Outputs
    -------
    median_file: str
        absolute output filename
    
    """


    img = []
    for i,filename in enumerate(files):
        img_load = nb.load(filename)
        img.append([x.data for x in img_load.darrays]) #(runs,TRs,vertices)

    median_img = np.median(img, axis = 0)

    darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in median_img]
    median_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                           extra = img_load.extra,
                                           darrays = darrays) # need to save as gii again

    median_file = os.path.join(outdir,re.sub('run-\d{2}_','run-median_',os.path.split(files[0])[-1]))
    nb.save(median_gii,median_file)

    return median_file


def crop_gii(gii_path,num_TR,outpath, extension = '.func.gii'):

    """ crop gii file

    Parameters
    ----------
    gii_path : str
        absolute filename for gii file
    num_TR : int
        number of TRs to remove from beginning of file
    outpath : str
        path to save new file
    extension: str
        file extension
    

    Outputs
    -------
    crop_gii_path: str
        absolute filename for cropped gii file
    
    """
    
    outfile = os.path.split(gii_path)[-1].replace(extension,'_cropped'+extension)
    crop_gii_path = os.path.join(outpath,outfile)

    if os.path.isfile(crop_gii_path): # if cropped file exists, skip
        print('File {} already in folder, skipping'.format(crop_gii_path))

    else:
        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(gii_path)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        crop_data = data_in[num_TR:,:] # crop initial TRs
        print('original file with %d TRs, now cropped and has %d TRs' %(data_in.shape[0],crop_data.shape[0]))

        # save as gii again
        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in crop_data]
        gii_out = nb.gifti.gifti.GiftiImage(header = gii_in.header, extra = gii_in.extra, darrays = darrays)

        nb.save(gii_out,crop_gii_path) # save as gii file
        print('new file saved in %s'%crop_gii_path)

    return crop_gii_path



def highpass_gii(filename,polyorder,deriv,window,outpth, extension = '.func.gii'):

    """ highpass filter gii file

    Parameters
    ----------
    filename : List/array
        list of absolute filename for gii file
    polyorder : int
        order of the polynomial used to fit the samples - must be less than window_length.
    deriv : int
        order of the derivative to compute - must be a nonnegative integer
    window: int
        length of the filter window (number of coefficients) - must be a positive odd integer
    outpth: str
        path to save new files
    extension: str
        file extension
    

    Outputs
    -------
    filename_sg: arr
        np array with filtered run
    filepath_sg: str
        filtered filename
    
    """

    outfile = os.path.split(filename)[-1].replace(extension,'_sg'+extension)
    filepath_sg = os.path.join(outpth,outfile)

    if not os.path.isfile(filename): # check if original file exists
            print('no file found called %s' %filename)
            filename_sg = []
            filepath_sg = []

    elif os.path.isfile(filepath_sg): # if filtered file exists, skip
        print('File {} already in folder, skipping'.format(filepath_sg))
        filename_sg = nb.load(filepath_sg)
        filename_sg = np.array([filename_sg.darrays[i].data for i in range(len(filename_sg.darrays))]) #load surface data

    else:

        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(filename)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        print('filtering run %s' %filename)
        data_in_filt = savgol_filter(data_in, window, polyorder, axis = 0, deriv = deriv, mode = 'nearest')
        filename_sg = data_in - data_in_filt + data_in_filt.mean(axis = 0) # add mean image back to avoid distribution around 0

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in filename_sg]
        gii_out = nb.gifti.gifti.GiftiImage(header = gii_in.header, extra = gii_in.extra, darrays = darrays)

        filename_sg = np.array(filename_sg)
        nb.save(gii_out,filepath_sg) # save as gii file


    return filename_sg,filepath_sg



def psc_gii(gii_file,outpth, method='median', extension = '.func.gii'):

    """ percent signal change gii file

    Parameters
    ----------
    gii_file : str
        absolute filename for gii
    outpth: str
        path to save new files
    method: str
        do median vs mean 
    extension: str
        file extension

    Outputs
    -------
    psc_gii: arr
        np array with percent signal changed file
    psc_gii_pth: List/arr
        list with absolute filenames for saved giis
    
    """

    outfile = os.path.split(gii_file)[-1].replace(extension,'_psc'+extension)
    psc_gii_pth = os.path.join(outpth,outfile)

    if not os.path.isfile(gii_file): # check if file exists
            print('no file found called %s' %gii_file)
            psc_gii = []
            psc_gii_pth = []

    elif os.path.isfile(psc_gii_pth): # if psc file exists, skip
        print('File {} already in folder, skipping'.format(psc_gii_pth))
        psc_gii = nb.load(psc_gii_pth)
        psc_gii = np.array([psc_gii.darrays[i].data for i in range(len(psc_gii.darrays))]) #load surface data

    else:

        # load with nibabel instead to save outputs always as gii
        img_load = nb.load(gii_file)
        data_in = np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]) #load surface data

        print('PSC run %s' %gii_file)

        if method == 'mean':
            data_m = np.mean(data_in,axis=0)
        elif method == 'median':
            data_m = np.median(data_in, axis=0)

        psc_gii = 100.0 * (data_in - data_m)/data_m#np.abs(data_m)

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in psc_gii]
        new_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                           extra = img_load.extra,
                                           darrays = darrays) # need to save as gii again

        psc_gii = np.array(psc_gii)
        print('saving %s' %psc_gii_pth)
        nb.save(new_gii,psc_gii_pth) #save in correct path


    return psc_gii, psc_gii_pth



def screenshot2DM(filenames,scale,screen,outfile,dm_shape = 'rectangle'):

    """ percent signal change gii file

    Parameters
    ----------
    filenames : List/arr
        list of absolute filenames of pngs
    scale : float
        scaling factor, to downsample images
    screen : list/arr
        list of screen resolution [hRes,vRes]
    outdir: str
        path to save new files
    
    Outputs
    -------
    DM : str
        absolute output design matrix filename
    
    """
    
    hRes = int(screen[0])
    vRes = int(screen[1])
    
    if dm_shape == 'square': # make square dm, using max screen dim
        dim1 = hRes
        dim2 = hRes
        
    else:
        dim1 = hRes
        dim2 = vRes
        
    im_gr_resc = np.zeros((len(filenames),int(dim2*scale),int(dim1*scale)))
    
    for i, png in enumerate(filenames): #rescaled and grayscaled images
        image = Image.open(png).convert('RGB')
        
        if dm_shape == 'square': # add padding (top and bottom borders)
            #padded_img = Image.new(image.mode, (hRes, hRes), (255, 255, 255))
            #padded_img.paste(image, (0, ((hRes - vRes) // 2)))
            padding = (0, (hRes - vRes)//2, 0, (hRes - vRes)-((hRes - vRes)//2))
            image = ImageOps.expand(image, padding, fill=(255, 255, 255))
            #plt.imshow(image)
            
        image = image.resize((dim1,dim2), Image.ANTIALIAS)
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(np.asarray(image)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]

    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))



def shift_DM(prf_dm):

    """
    function to shift bars in pRF DM, getting the average bar position 
    
    Note: Very clunky and non-generic function, but works.
     should optimize eventually
    """

    # initialize a new DM with zeros, same shape as initial DM
    avg_prf_dm = np.zeros(prf_dm.shape)

    vert_bar_updown = range(13,22) #[13-21]
    vert_bar_downup = range(73,82) #[73-81]
    hor_bar_rightleft = range(24,41) #[24-40]
    hor_bar_leftright = range(54,71) #[54-70]

    # set vertical axis limits, to not plot above or below that
    # use first and last TR from initial bar pass (vertical up->down)
    vert_min_pix = np.where(prf_dm[0,:,vert_bar_updown[0]]==255)[0][0] # minimum vertical pixel index, below that should be empty (because no more display)
    vert_max_pix = np.where(prf_dm[0,:,vert_bar_updown[-1]]==255)[0][-1] # maximum vertical pixel index, above that should be empty (because no more display)

    # first get median width (grossura) of vertical and horizontal bars at a TR where full bar on screen
    length_vert_bar = int(np.median([len(np.where(prf_dm[x,:,vert_bar_updown[2]]==255)[0]) for x in range(prf_dm[:,:,vert_bar_updown[2]].shape[0])]))
    length_hor_bar = int(np.median([len(np.where(prf_dm[:,x,hor_bar_rightleft[2]]==255)[0]) for x in range(prf_dm[:,:,hor_bar_rightleft[2]].shape[1])]))

    # amount of pixel indexs I should shift bar forward in time -> (TR2 - TR1)/2
    shift_increment = math.ceil((np.median([np.where(prf_dm[x,:,vert_bar_updown[1]]==255)[0][-1] for x in range(prf_dm[:,:,vert_bar_updown[1]].shape[0])]) - \
        np.median([np.where(prf_dm[x,:,vert_bar_updown[0]]==255)[0][-1] for x in range(prf_dm[:,:,vert_bar_updown[0]].shape[0])]))/2)


    for j in range(prf_dm.shape[-1]): # FOR ALL TRs (j 0-89)

        # FOR VERTICAL BAR PASSES
        if j in vert_bar_updown or j in vert_bar_downup: 

            # loop to fill pixels that belong to the new bar position at that TR
            for i in range(length_vert_bar):
                if j in vert_bar_downup: 

                    if j==vert_bar_downup[-1]:

                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[0,:,j]==255)[0][-1]-shift_increment

                        if avg_end_pos-i>=vert_min_pix: # if bigger that min pix index, which means it's within screen
                            avg_prf_dm[:,avg_end_pos-i,j]=255

                    else:
                        # shift start position and fill screen horizontally to make new bar
                        avg_start_pos = np.where(prf_dm[0,:,j]==255)[0][0]-shift_increment

                        if avg_start_pos+i<=vert_max_pix: # if lower that max pix index, which means it's within screen
                            avg_prf_dm[:,avg_start_pos+i,j]=255

                elif j in vert_bar_updown: #or j==vert_bar_downup[-1]:

                    if j==vert_bar_updown[-1]:

                        # shift start position and fill screen horizontally to make new bar
                        avg_start_pos = np.where(prf_dm[0,:,j]==255)[0][0]+shift_increment

                        if avg_start_pos+i<=vert_max_pix: # if lower that max pix index, which means it's within screen
                            avg_prf_dm[:,avg_start_pos+i,j]=255

                    else:
                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[0,:,j]==255)[0][-1]+shift_increment

                        if avg_end_pos-i>=vert_min_pix: # if bigger that min pix index, which means it's within screen
                            avg_prf_dm[:,avg_end_pos-i,j]=255

        # FOR HORIZONTAL BAR PASSES
        if j in hor_bar_rightleft or j in hor_bar_leftright: 

            # loop to fill pixels that belong to the new bar position at that TR
            for i in range(length_hor_bar):

                if j in hor_bar_rightleft:
                    if j in hor_bar_rightleft[-2:]: # last two TRs might already be in limit, so fill based on other bar side

                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][-1]-shift_increment

                        if avg_end_pos-i>=0: # if bigger than 0 (min x index), which means it's within screen
                            avg_prf_dm[avg_end_pos-i,vert_min_pix:vert_max_pix,j]=255

                    else:
                        avg_start_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][0]-shift_increment

                        if avg_start_pos+i<=prf_dm.shape[0]-1: # if lower than 168 (max x index), which means it's within screen
                            avg_prf_dm[avg_start_pos+i,vert_min_pix:vert_max_pix,j]=255

                elif j in hor_bar_leftright:
                    if j in hor_bar_leftright[-2:]: # last two TRs might already be in limit, so fill based on other bar side

                        avg_start_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][0]+shift_increment

                        if avg_start_pos+i<=prf_dm.shape[0]-1: # if lower than 168 (max x index), which means it's within screen
                            avg_prf_dm[avg_start_pos+i,vert_min_pix:vert_max_pix,j]=255

                    else:                    
                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][-1]+shift_increment

                        if avg_end_pos-i>=0: # if bigger than 0 (min x index), which means it's within screen
                            avg_prf_dm[avg_end_pos-i,vert_min_pix:vert_max_pix,j]=255

    return avg_prf_dm #(x,y,t)


def join_chunks(path, out_name, hemi, chunk_num = 83, fit_model = 'css'):
    """ combine all chunks into one single estimate numpy array (per hemisphere)

    Parameters
    ----------
    path : str
        path to files
    out_name: str
        output name of combined estimates
    hemi : str
        'hemi_L' or 'hemi_R' hemisphere
    chunk_num : int
        total number of chunks to combine (per hemi)
    fit_model: str
        fit model of estimates
    
    Outputs
    -------
    estimates : npz 
        numpy array of estimates
    
    """
    print(hemi)
    
    for ch in range(chunk_num):
        
        chunk_name = [x for _,x in enumerate(os.listdir(path)) if hemi in x and fit_model in x and 'chunk-%s'%str(ch+1).zfill(3) in x][0]
        print('loading chunk %s'%chunk_name)
        chunk = np.load(os.path.join(path, chunk_name)) # load chunk
        
        if ch == 0:
            xx = chunk['x']
            yy = chunk['y']

            size = chunk['size']

            beta = chunk['betas']
            baseline = chunk['baseline']

            if fit_model =='css': 
                ns = chunk['ns']

            rsq = chunk['r2']
        else:
            xx = np.concatenate((xx,chunk['x']))
            yy = np.concatenate((yy,chunk['y']))

            size = np.concatenate((size,chunk['size']))

            beta = np.concatenate((beta,chunk['betas']))
            baseline = np.concatenate((baseline,chunk['baseline']))

            if fit_model =='css': 
                ns = np.concatenate((ns,chunk['ns']))

            rsq = np.concatenate((rsq,chunk['r2']))
    
    print('shape of estimates for hemifield %s is %s'%(hemi,str(xx.shape)))

    # save file
    output = os.path.join(path,out_name)
    print('saving %s'%output)
    
    if fit_model =='css':
        np.savez(output,
              x = xx,
              y = yy,
              size = size,
              betas = beta,
              baseline = baseline,
              ns = ns,
              r2 = rsq)
    else:        
        np.savez(output,
              x = xx,
              y = yy,
              size = size,
              betas = beta,
              baseline = baseline,
              r2 = rsq)
     
            
    return np.load(output)


def dva_per_pix(height_cm,distance_cm,vert_res_pix):

    """ calculate degrees of visual angle per pixel, 
    to use for screen boundaries when plotting/masking

    Parameters
    ----------
    height_cm : int
        screen height
    distance_cm: float
        screen distance (same unit as height)
    vert_res_pix : int
        vertical resolution of screen
    
    Outputs
    -------
    deg_per_px : float
        degree (dva) per pixel
    
    """

    # screen size in degrees / vertical resolution
    deg_per_px = (2.0 * np.degrees(np.arctan(height_cm /(2.0*distance_cm))))/vert_res_pix

    return deg_per_px

    
def mask_estimates(estimates, sub, params, ROI = 'V1', fit_model = 'css'):
    
    """ mask estimates, to be positive RF, within screen limits
    and for a certain ROI (if the case)

    Parameters
    ----------
    estimates : List/arr
        list of estimates.npz for both hemispheres
    sub: str
        sub ID
    params : dict
        parameteres dictionart with relevant settings
    ROI : str
        roi to mask estimates (can also be 'None')
    fit_model: str
        fit model of estimates
    
    Outputs
    -------
    masked_estimates : npz 
        numpy array of masked estimates
    
    """
    
    xx = np.concatenate((estimates[0]['x'],estimates[1]['x']))
    yy = np.concatenate((estimates[0]['y'],estimates[1]['y']))
       
    size = np.concatenate((estimates[0]['size'],estimates[1]['size']))
    
    beta = np.concatenate((estimates[0]['betas'],estimates[1]['betas']))
    baseline = np.concatenate((estimates[0]['baseline'],estimates[1]['baseline']))
    
    if fit_model == 'css': 
        ns = np.concatenate((estimates[0]['ns'],estimates[1]['ns'])) # exponent of css
    else: #if gauss
        ns = np.ones(xx.shape)

    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2']))
    
    # set limits for xx and yy, forcing it to be within the screen boundaries
    
    # subjects that did pRF task with linux computer, so res was full HD
    HD_subs = [str(num).zfill(2) for num in params['general']['HD_screen_subs']] 
    res = params['general']['screenRes_HD'] if str(sub).zfill(2) in HD_subs else params['general']['screenRes']
    
    max_size = params['fitting']['prf']['max_size']
    
    vert_lim_dva = (res[-1]/2) * dva_per_pix(params['general']['screen_width'],params['general']['screen_distance'],res[0])
    hor_lim_dva = (res[0]/2) * dva_per_pix(params['general']['screen_width'],params['general']['screen_distance'],res[0])
    
    
    # make new variables that are masked 
    masked_xx = np.zeros(xx.shape); masked_xx[:]=np.nan
    masked_yy = np.zeros(yy.shape); masked_yy[:]=np.nan
    masked_size = np.zeros(size.shape); masked_size[:]=np.nan
    masked_beta = np.zeros(beta.shape); masked_beta[:]=np.nan
    masked_baseline = np.zeros(baseline.shape); masked_baseline[:]=np.nan
    masked_rsq = np.zeros(rsq.shape); masked_rsq[:]=np.nan
    masked_ns = np.zeros(ns.shape); masked_ns[:]=np.nan

    for i in range(len(xx)): #for all vertices
        if xx[i] <= hor_lim_dva and xx[i] >= -hor_lim_dva: # if x within horizontal screen dim
            if yy[i] <= vert_lim_dva and yy[i] >= -vert_lim_dva: # if y within vertical screen dim
                if beta[i]>=0: # only account for positive RF
                    if size[i]<=max_size: # limit size to max size defined in fit

                        # save values
                        masked_xx[i] = xx[i]
                        masked_yy[i] = yy[i]
                        masked_size[i] = size[i]
                        masked_beta[i] = beta[i]
                        masked_baseline[i] = baseline[i]
                        masked_rsq[i] = rsq[i]
                        masked_ns[i]=ns[i]

    if ROI != 'None':
        
        roi_ind = cortex.get_roi_verts(params['processing']['space'],ROI) # get indices for that ROI
        
        # mask for roi
        masked_xx = masked_xx[roi_ind[ROI]]
        masked_yy = masked_yy[roi_ind[ROI]]
        masked_size = masked_size[roi_ind[ROI]]
        masked_beta = masked_beta[roi_ind[ROI]]
        masked_baseline = masked_baseline[roi_ind[ROI]]
        masked_rsq = masked_rsq[roi_ind[ROI]]
        masked_ns = masked_ns[roi_ind[ROI]]

    masked_estimates = {'x':masked_xx,'y':masked_yy,'size':masked_size,
                        'beta':masked_beta,'baseline':masked_baseline,'ns':masked_ns,
                        'rsq':masked_rsq}
    
    return masked_estimates


def align_yaxis(ax1, v1, ax2, v2):

    """
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    (function to align twin axis in same plot)
    """

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)



def smooth_gii(gii_file, outdir, space = 'fsaverage', fwhm = 2):

    """ takes gifti file and smooths it - saves new smoothed gifti

    Parameters
    ----------
    gii_file : str
        absolute path for gifti file
    outdir: str
        absolute output dir to save smoothed file
    space: str
        subject surface space
    fwhm : int/float
        width of the kernel, at half of the maximum of the height of the Gaussian
    
    Outputs
    -------
    smooth_gii_pth : str 
        absolute path for smoothed file
    
    """

    smooth_gii_pth = []

    if not os.path.isfile(gii_file): # check if file exists
            print('no file found called %s' %gii_file)
    else:

        # load with nibabel to save outputs always as gii
        gii_in = nb.load(gii_file)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) # load surface data

        print('loading file %s' %gii_file)

        # first need to convert to mgz
        # will be saved in output dir
        new_mgz = os.path.join(outdir,os.path.split(gii_file)[-1].replace('.func.gii','.mgz'))

        print('converting gifti to mgz as %s' %(new_mgz))
        os.system('mri_convert %s %s'%(gii_file, new_mgz))

        # now smooth mgz
        smoother = fs.SurfaceSmooth()
        smoother.inputs.in_file = new_mgz
        smoother.inputs.subject_id = space

        # define hemisphere
        smoother.inputs.hemi = 'lh' if '_hemi-L_' in new_mgz else 'rh'
        print('smoothing %s' %smoother.inputs.hemi)
        smoother.inputs.fwhm = fwhm
        smoother.run() # doctest: +SKIP

        new_filename = os.path.split(new_mgz)[-1].replace('.mgz','_smooth%d.mgz'%(smoother.inputs.fwhm))
        smooth_mgz = os.path.join(outdir,new_filename)
        shutil.move(os.path.join(os.getcwd(),new_filename), smooth_mgz) #move to correct dir

        # transform to gii again
        new_data = surface.load_surf_data(smooth_mgz).T

        smooth_gii_pth = smooth_mgz.replace('.mgz','.func.gii')
        print('converting to %s' %smooth_gii_pth)
        os.system('mri_convert %s %s'%(smooth_mgz,smooth_gii_pth))

    return smooth_gii_pth


def smooth_nparray(arr_in, header_filename, out_dir, filestr, sub_space = 'fsaverage', n_TR = 83, smooth_fwhm = 2, sub_ID = 'median'):

    """ takes array with shape (vertices,), with some relevant quantification
    and smooths it - useful for later plotting of surface map

    Parameters
    ----------
    arr_in: array
        numpy array to be smoothed with shape (vertices,)
    header_filename: list
        list of strings with absolute path to gii files to use has header info, should include each hemisphere gii files
    outdir: str
        absolute output dir to save smoothed array
    filestr: str
        identifier string to add to name of new gii/array (ex: '_rsq')
    sub_space: str
        subject surface space
    n_TR: int
        number of TRs of task
    smooth_fwhm : int/float
        width of the kernel, at half of the maximum of the height of the Gaussian
    sub_ID: str
        subject identifier (ex: '01' or 'median')

    Outputs
    -------
    out_array: array
        smoothed numpy array
    
    """
    

    # get mid vertex index (diving hemispheres)
    left_index = cortex.db.get_surfinfo(sub_space).left.shape[0] 

    # temporary output folder, to save files that will be deleted later
    new_out = os.path.join(out_dir,'temp_smooth')

    if not os.path.exists(new_out):  # check if path exists
        os.makedirs(new_out)

    # transform array into gii (to be smoothed)
    smooth_filename = []

    for field in ['hemi-L', 'hemi-R']:

        if field=='hemi-L':
            arr_4smoothing = arr_in[0:left_index]
        else:
            arr_4smoothing = arr_in[left_index::]

        # load hemi just to get header
        filename = [gii for _,gii in enumerate(header_filename) if field in gii]
        img_load = nb.load(filename[0])

        # absolute path of new gii
        out_filename = os.path.join(new_out, os.path.split(filename[0])[-1].replace('.func.gii',filestr+'.func.gii'))

        print('saving %s'%out_filename)
        est_array_tiled = np.tile(arr_4smoothing[np.newaxis,...],(n_TR,1)) # NEED TO DO THIS 4 MGZ to actually be read (header is of func file)
        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in est_array_tiled]
        estimates_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                               extra = img_load.extra,
                                               darrays = darrays) # need to save as gii

        nb.save(estimates_gii,out_filename)
    
        smo_estimates_path = smooth_gii(out_filename, new_out, space = sub_space, fwhm = smooth_fwhm) # smooth it!

        smooth_filename.append(smo_estimates_path)
        print('saving %s'%smo_estimates_path)


    # load both hemis and combine in one array
    smooth_arr = []

    for _,name in enumerate(smooth_filename): # not elegant but works

        img_load = nb.load(name)
        smooth_arr.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))

    out_array = np.concatenate((smooth_arr[0][0],smooth_arr[1][0]))  

    new_filename = os.path.split(smo_estimates_path)[-1].replace('.func.gii','.npy').replace('hemi-R','hemi-both')
    new_filename = re.sub('sub-\d{2}_', 'sub-%s_'%sub_ID.zfill(2), new_filename)

    print('saving smoothed file in %s'%os.path.join(out_dir,new_filename))
    np.save(os.path.join(out_dir,new_filename),out_array)

    if os.path.exists(new_out):  # check if path exists
        print('deleting %s to save memory'%str(new_out))
        shutil.rmtree(new_out)

    return out_array


def normalize(M):
    """
    normalize data array
    """
    return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))


def make_2D_colormap(rgb_color = '101', bins = 50):
    """
    generate 2D basic colormap
    and save to pycortex filestore
    """
    
    ##generating grid of x bins
    x,y = np.meshgrid(
        np.linspace(0,1,bins),
        np.linspace(0,1,bins)) 
    
    # define color combination for plot
    if rgb_color=='101': #red blue
        col_grid = np.dstack((x,np.zeros_like(x), y))
        name='RB'
    elif rgb_color=='110': # red green
        col_grid = np.dstack((x, y,np.zeros_like(x)))
        name='RG'
    elif rgb_color=='011': # green blue
        col_grid = np.dstack((np.zeros_like(x),x, y))
        name='GB'
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(col_grid,
    extent = (0,1,0,1),
    origin = 'lower')
    ax.axis('off')

    rgb_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'costum2D_'+name+'_bins_%d.png'%bins)

    plt.savefig(rgb_fn, dpi = 200)
       
    return rgb_fn  


def median_soma_events(files,outdir):
    
    """ function that makes median event data frame (over runs)

    Parameters
    ----------
    files : List/arr
        list of absolute filenames to do median over
    outdir : str
        path to save new files
    

    Outputs
    -------
    median_file: str
        absolute output filename
    
    """
    
    # set output name
    median_file = os.path.join(outdir, re.sub('run-\d{2}_', 'run-median_', os.path.split(files[0])[-1]))
    
    if os.path.isfile(median_file):
        print('file %s already exists'%median_file)
    else:
        # list of stimulus onsets
        print('averaging %d event files'%len(files))

        all_events = []
        
        for _,val in enumerate(files):

            events_pd = pd.read_csv(val,sep = '\t')

            new_events = []

            for ev in events_pd.iterrows():
                row = ev[1]   
                if row['trial_type'][0] == 'b': # if both hand/leg then add right and left events with same timings
                    new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
                    new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
                else:
                    new_events.append([row['onset'],row['duration'],row['trial_type']])

            df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
            all_events.append(df)

        # make median event dataframe
        onsets = []
        durations = []
        for w in range(len(all_events)):
            onsets.append(all_events[w]['onset'])
            durations.append(all_events[w]['duration'])

        events_avg = pd.DataFrame({'onset':np.median(np.array(onsets),axis=0),
                                   'duration':np.median(np.array(durations),axis=0),
                                   'trial_type':all_events[0]['trial_type']})
        # save with output name
        events_avg.to_csv(median_file, sep="\t")
        
        print('computed median events')

    return median_file    


def fit_glm(voxel, dm):
    
    """ GLM fit on timeseries
    Regress a created design matrix on the input_data.

    Parameters
    ----------
    voxel : arr
        timeseries of a single voxel
    dm : arr
        DM array (#TR,#regressors)
    

    Outputs
    -------
    prediction : arr
        model fit for voxel
    betas : arr
        betas for model
    r2 : arr
        coefficient of determination
    mse : arr
        mean of the squared residuals
    
    """

    if np.isnan(voxel).any():
        betas = np.nan
        prediction = np.nan
        mse = np.nan
        r2 = np.nan

    else:   # if not nan (some vertices might have nan values)
        betas = np.linalg.lstsq(dm, voxel, rcond = -1)[0]
        prediction = dm.dot(betas)

        mse = np.mean((voxel - prediction) ** 2) # calculate mean of squared residuals
        r2 = pearsonr(prediction, voxel)[0] ** 2 # and the rsq
    
    return prediction, betas, r2, mse


def leave_one_out(input_list):

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



def set_contrast(dm_col,tasks,contrast_val=[1],num_cond=1):
    
    """ define contrast matrix

    Parameters
    ----------
    dm_col : list/arr
        design matrix columns (all possible task names in list)
    tasks : list/arr
        list with list of tasks to give contrast value
        if num_cond=1 : [tasks]
        if num_cond=2 : [tasks1,tasks2], contrast will be tasks1 - tasks2     
    contrast_val : list/arr 
        list with values for contrast
        if num_cond=1 : [value]
        if num_cond=2 : [value1,value2], contrast will be tasks1 - tasks2
    num_cond : int
        if one task vs the implicit baseline (1), or if comparing 2 conditions (2)

    Outputs
    -------
    contrast : list/arr
        contrast array

    """
    
    contrast = np.zeros(len(dm_col))

    if num_cond == 1: # if only one contrast value to give ("task vs implicit intercept")

        for j,name in enumerate(tasks[0]):
            for i in range(len(contrast)):
                if dm_col[i] == name:
                    contrast[i] = contrast_val[0]

    elif num_cond == 2: # if comparing 2 conditions (task1 - task2)

        for k,lbl in enumerate(tasks):
            idx = []
            for i,val in enumerate(lbl):
                idx.extend(np.where([1 if val == label else 0 for _,label in enumerate(dm_col)])[0])

            val = contrast_val[0] if k==0 else contrast_val[1] # value to give contrast

            for j in range(len(idx)):
                for i in range(len(dm_col)):
                    if i==idx[j]:
                        contrast[i]=val

    print('contrast for %s is %s'%(tasks,contrast))
    return contrast



def compute_stats(voxel, dm, contrast, betas, pvalue = 'oneside'):
    
    """ compute statistis for GLM

    Parameters
    ----------
    voxel : arr
        timeseries of a single voxel
    dm : arr
        DM array (#TR,#regressors)
    contrast: arr
        contrast vector
    betas : arr
        betas for model at that voxel
    pvalue : str
        type of tail for p-value - 'oneside'/'twoside'

    Outputs
    -------
    t_val : float
        t-statistic for that voxel relative to contrast
    p_val : float
        p-value for that voxel relative to contrast
    z_score : float
        z-score for that voxel relative to contrast
    
    """

    
    def design_variance(X, which_predictor=1):
        
        ''' Returns the design variance of a predictor (or contrast) in X.
        
        Parameters
        ----------
        X : numpy array
            Array of shape (N, P)
        which_predictor : int or list/array
            The index of the predictor you want the design var from.
            Note that 0 refers to the intercept!
            Alternatively, "which_predictor" can be a contrast-vector
            
        Outputs
        -------
        des_var : float
            Design variance of the specified predictor/contrast from X.
        '''
    
        is_single = isinstance(which_predictor, int)
        if is_single:
            idx = which_predictor
        else:
            idx = np.array(which_predictor) != 0

        c = np.zeros(X.shape[1])
        c[idx] = 1 if is_single == 1 else which_predictor[idx]
        des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
        
        return des_var

    
    if np.isnan(voxel).any():
        t_val = np.nan
        p_val = np.nan
        z_score = np.nan

    else:   # if not nan (some vertices might have nan values)
        
        # calculate design variance
        design_var = design_variance(dm, contrast)
        
        # sum of squared errors
        sse = ((voxel - (dm.dot(betas))) ** 2).sum() 
        
        #degrees of freedom = N - P = timepoints - predictores
        df = (dm.shape[0] - dm.shape[1])
        
        # t statistic for vertex
        t_val = contrast.dot(betas) / np.sqrt((sse/df) * design_var)

        if pvalue == 'oneside': 
            # compute the p-value (right-tailed)
            p_val = t.sf(t_val, df) 

            # z-score corresponding to certain p-value
            z_score = norm.isf(np.clip(p_val, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

        elif pvalue == 'twoside':
            # take the absolute by np.abs(t)
            p_val = t.sf(np.abs(t_val), df) * 2 # multiply by two to create a two-tailed p-value

            # z-score corresponding to certain p-value
            z_score = norm.isf(np.clip(p_val/2, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

    return t_val,p_val,z_score


def mask_arr(arr, threshold = 0, side = 'above'):
    
    ''' mask array given a threshold value
        
        Parameters
        ----------
        arr : arr
            array with data to threshold (ex: zscores)
        threshold : int/float
            threshold value
        side : str
            'above'/'below'/'both', indicating if output masked values will be
             above threshold, below -threshold or both
        
        Outputs
        -------
        data_threshed : arr
            thresholded data
    '''

    # set at nan whatever is outside thresh
    data_threshed = np.zeros(arr.shape); data_threshed[:]=np.nan 

    for i,value in enumerate(arr):
        if side == 'above':
            if value > threshold:
                data_threshed[i] = value
        elif side == 'below':
            if value < -threshold:
                data_threshed[i] = value
        elif side == 'both':
            if value < -threshold or value > threshold:
                data_threshed[i] = value

    return data_threshed


def COM(data):
    
    """ given an array of values x vertices, 
    compute center of mass 

    Parameters
    ----------
    data : List/arr
        array with values to COM  (elements,vertices)

    Outputs
    -------
    center_mass : arr
        array with COM for each vertex
    """
    
    # first normalize data, to fix issue of negative values
    norm_data = np.array([normalize(data[...,x]) for x in range(data.shape[-1])])
    norm_data = norm_data.T
    
    #then calculate COM for each vertex
    center_mass = np.array([ndimage.measurements.center_of_mass(norm_data[...,x]) for x in range(norm_data.shape[-1])])

    return center_mass.T[0]



def spm_hrf(delay, TR):
    """ 
    [TAKEN FROM POPEYE 0.5.2 - in new version hrf has diff distribution]
    An implementation of spm_hrf.m from the SPM distribution
    
    Arguments:
    
    Required:
    TR: repetition time at which to generate the HRF (in seconds)
    
    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32
       
    """
    # default settings
    p=[5,15,1,1,6,0,32]
    p=[float(x) for x in p]
    
    # delay variation
    p[0] += delay
    p[1] += delay
    
    fMRI_T = 16.0
    
    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf=stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=np.array(range(np.int(p[6]/TR)))*fMRI_T
    hrf = hrf[good_pts.astype(int)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf /= trapz(hrf)
    return hrf


def add_alpha2colormap(colormap = 'rainbow_r', bins = 256, invert_alpha = False, cmap_name = 'costum',
                      discrete = False):

    """ add alpha channel to colormap,
    and save to pycortex filestore

    Parameters
    ----------
    colormap : str or List/arr
        if string then has to be a matplolib existent colormap
        if list/array then contains strings with color names, to create linear segmented cmap
    bins : int
        number of bins for colormap
    invert_alpha : bool
        if we want to invert direction of alpha channel
        (y can be from 0 to 1 or 1 to 0)
    cmap_name : str
        new cmap filename, final one will have _alpha_#-bins added to it
    discrete : bool
        if we want a discrete colormap or not (then will be continuous)

    Outputs
    -------
    rgb_fn : str
        absolute path to new colormap
    """
    
    if isinstance(colormap, str): # if input is string (so existent colormap)

        # get colormap
        cmap = cm.get_cmap(colormap)

    else: # is list of strings
        cvals  = np.arange(len(colormap))
        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colormap))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        
        if discrete == True: # if we want a discrete colormap from list
            cmap = matplotlib.colors.ListedColormap(colormap)
            bins = int(len(colormap))

    # convert into array
    cmap_array = cmap(range(bins))
    
    # make alpha array
    if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
    else:
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))
    
    # reshape array for map
    new_map = []
    for i in range(cmap_array.shape[-1]):
        new_map.append(np.tile(cmap_array[...,i],(bins,1)))

    new_map = np.moveaxis(np.array(new_map), 0, -1)

    # add alpha channel
    new_map[...,-1] = alpha
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(new_map,
    extent = (0,1,0,1),
    origin = 'lower')
    ax.axis('off')

    rgb_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)

    #misc.imsave(rgb_fn, new_map)
    plt.savefig(rgb_fn, dpi = 200,transparent=True)
       
    return rgb_fn  



