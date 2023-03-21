
# useful functions to use in other scripts

import os, re
import numpy as np
import os.path as op
import shutil

import pandas as pd
import nibabel as nb

from nilearn import surface

from scipy.signal import savgol_filter
import nipype.interfaces.freesurfer as fs

import cv2

from skimage import color
import cv2
from skimage.transform import rescale
from skimage.filters import threshold_triangle

from PIL import Image, ImageOps, ImageDraw

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
from scipy import fft

import itertools
from joblib import Parallel, delayed


def median_gii(files,outdir, run_name='median'):

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
    avg_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                           extra = img_load.extra,
                                           darrays = darrays) # need to save as gii again

    median_file = os.path.join(outdir,re.sub('run-\d{2}_','run-%s_'%(run_name),os.path.split(files[0])[-1]))
    nb.save(avg_gii,median_file)

    return median_file

def mean_gii(files,outdir, run_name='mean'):

    """ make average gii file (over runs)

    Parameters
    ----------
    files : List/arr
        list of absolute filenames to do median over
    outdir : str
        path to save new files
    

    Outputs
    -------
    avg_file: str
        absolute output filename
    
    """


    img = []
    for i,filename in enumerate(files):
        img_load = nb.load(filename)
        img.append([x.data for x in img_load.darrays]) #(runs,TRs,vertices)

    avg_img = np.mean(img, axis = 0)

    darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in avg_img]
    avg_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                           extra = img_load.extra,
                                           darrays = darrays) # need to save as gii again

    avg_file = os.path.join(outdir,re.sub('run-\d{2}_','run-%s_'%(run_name),os.path.split(files[0])[-1]))
    nb.save(avg_gii,avg_file)

    return avg_file


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


def dc_filter_gii(filename, outpth, extension = '.func.gii',
                                first_modes_to_remove=5):
    
    """ 
    High pass discrete cosine filter array
    
    Parameters
    ----------
    filename : List/array
        list of absolute filename for gii file
    outpth: str
        path to save new files
    extension: str
        file extension
    first_modes_to_remove: int
        Number of low-frequency eigenmodes to remove (highpass)
    
    Outputs
    -------
    filename_filt: arr
        np array with filtered run
    filepath_filt: str
        filtered filename
    """ 

    outfile = os.path.split(filename)[-1].replace(extension,'_dc'+extension)
    filepath_filt = os.path.join(outpth,outfile)

    if not os.path.isfile(filename): # check if original file exists
            print('no file found called %s' %filename)
            filename_filt = []
            filepath_filt = []

    elif os.path.isfile(filepath_filt): # if filtered file exists, skip
        print('File {} already in folder, skipping'.format(filepath_filt))
        filename_filt = nb.load(filepath_filt)
        filename_filt = np.array([filename_filt.darrays[i].data for i in range(len(filename_filt.darrays))]) #load surface data

    else:
        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(filename)
        data = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        print('filtering run %s' %filename)

        # get Discrete Cosine Transform
        coeffs = fft.dct(data, norm='ortho', axis=0)
        coeffs[...,:first_modes_to_remove] = 0

        # filter signal
        filtered_signal = fft.idct(coeffs, norm='ortho', axis=0)
        # add mean image back to avoid distribution around 0
        filename_filt = filtered_signal + np.mean(data, axis=0)#[np.newaxis, ...]

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in filename_filt]
        gii_out = nb.gifti.gifti.GiftiImage(header = gii_in.header, extra = gii_in.extra, darrays = darrays)

        filename_filt = np.array(filename_filt)
        nb.save(gii_out,filepath_filt) # save as gii file

    return filename_filt, filepath_filt




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

        psc_gii = 100.0 * ((data_in - data_m)/np.absolute(data_m))

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


def crop_shift_arr(arr, crop_nr = None, shift = 0):
    
    """
    helper function to crop and shift array
    
    Parameters
    ----------
    arr : array
       original array
       assumes time dim is last one (arr.shape[-1])
    crop_nr : None or int
        if not none, expects int with number of FIRST time points to crop
    shift : int
        positive or negative int, of number of time points to shift (if neg, will shift leftwards)
        
    """
        
    # if cropping
    if crop_nr:
        new_arr = arr[...,crop_nr:]
    else:
        new_arr = arr
        
    # if shiftting
    out_arr = new_arr.copy()
    if shift > 0:
        out_arr[...,shift:] = new_arr[..., :-int(shift)]
    elif shift < 0:
        out_arr[...,:shift] = new_arr[..., np.abs(shift):]
        
    return out_arr


def make_prf_dm(res_scaling = .1, screen_res = [1680,1050], dm_shape = 'square',
                bar_width_ratio = 0.125, iti = 0.5,
                bar_pass_in_TRs = {'empty': 10, 'L-R': 18, 'R-L': 18, 'U-D': 10, 'D-U': 10},
                bar_pass_direction = ['empty','U-D','R-L','empty','L-R','D-U','empty'],
                run_lenght_TR = 90, crop_nr = 4, shift = -1):

    """ 

    Make design matrix array for pRF task

    """

    # get array of bar condition label per TR
    condition_per_TR = []
    for cond in bar_pass_direction:
        condition_per_TR += list(np.tile(cond, bar_pass_in_TRs[cond]))
        
        if np.ceil(iti)>0: # if ITI in TR, 
            condition_per_TR += list(np.tile('empty', int(np.ceil(iti))))
        
    # drop last TRs, for DM to have same time lenght as data
    condition_per_TR = condition_per_TR[:run_lenght_TR]

    ## crop and shift if such was the case
    condition_per_TR = crop_shift_arr(np.array(condition_per_TR)[np.newaxis], 
                                        crop_nr = crop_nr, shift = shift)[0]

    # all possible positions in pixels for for midpoint of
    # y position for vertical bar passes, 
    ver_y = screen_res[1]*np.linspace(0,1, bar_pass_in_TRs['U-D'])#+1)
    # x position for horizontal bar passes 
    hor_x = screen_res[0]*np.linspace(0,1, bar_pass_in_TRs['L-R'])#+1)

    # coordenates for bar pass, for PIL Image
    coordenates_bars = {'L-R': {'upLx': hor_x - 0.5 * bar_width_ratio * screen_res[0], 
                                'upLy': np.repeat(screen_res[1], bar_pass_in_TRs['L-R']),
                                'lowRx': hor_x + 0.5 * bar_width_ratio * screen_res[0], 
                                'lowRy': np.repeat(0, bar_pass_in_TRs['L-R'])},
                        'R-L': {'upLx': np.array(list(reversed(hor_x - 0.5 * bar_width_ratio * screen_res[0]))), 
                                'upLy': np.repeat(screen_res[1], bar_pass_in_TRs['R-L']),
                                'lowRx': np.array(list(reversed(hor_x+ 0.5 * bar_width_ratio * screen_res[0]))), 
                                'lowRy': np.repeat(0, bar_pass_in_TRs['R-L'])},
                        'U-D': {'upLx': np.repeat(0, bar_pass_in_TRs['U-D']), 
                                'upLy': ver_y+0.5 * bar_width_ratio * screen_res[1],
                                'lowRx': np.repeat(screen_res[0], bar_pass_in_TRs['U-D']), 
                                'lowRy': ver_y - 0.5 * bar_width_ratio * screen_res[1]},
                        'D-U': {'upLx': np.repeat(0, bar_pass_in_TRs['D-U']), 
                                'upLy': np.array(list(reversed(ver_y + 0.5 * bar_width_ratio * screen_res[1]))),
                                'lowRx': np.repeat(screen_res[0], bar_pass_in_TRs['D-U']), 
                                'lowRy': np.array(list(reversed(ver_y - 0.5 * bar_width_ratio * screen_res[1])))}
                        }

    # save screen display for each TR 
    visual_dm_array = np.zeros((len(condition_per_TR), round(screen_res[1] * res_scaling), round(screen_res[0] * res_scaling)))
    i = 0

    for trl, bartype in enumerate(condition_per_TR): # loop over bar pass directions

        img = Image.new('RGB', tuple(screen_res)) # background image

        if bartype not in np.array(['empty','empty_long']): # if not empty screen

            #print(bartype)

            # set draw method for image
            draw = ImageDraw.Draw(img)
            # add bar, coordinates (upLx, upLy, lowRx, lowRy)
            draw.rectangle(tuple([coordenates_bars[bartype]['upLx'][i],coordenates_bars[bartype]['upLy'][i],
                                coordenates_bars[bartype]['lowRx'][i],coordenates_bars[bartype]['lowRy'][i]]), 
                        fill = (255,255,255),
                        outline = (255,255,255))

            # increment counter
            if trl < (len(condition_per_TR) - 1):
                i = i+1 if condition_per_TR[trl] == condition_per_TR[trl+1] else 0    

        ## save in array
        visual_dm_array[int(trl):int(trl + 1), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]

    # swap axis to have time in last axis [x,y,t]
    visual_dm = visual_dm_array.transpose([1,2,0])

    if dm_shape == 'square':
        ## make it square
        # add padding (top and bottom borders) 
        new_visual_dm = np.zeros((round(np.max(screen_res) * res_scaling), round(np.max(screen_res) * res_scaling),
                                len(condition_per_TR)))

        pad_ind = int(np.ceil((screen_res[0] - screen_res[1])/2 * res_scaling))
        new_visual_dm[pad_ind:int(visual_dm.shape[0]+pad_ind),:,:] = visual_dm.copy()
    else:
        new_visual_dm = visual_dm.copy()
        
    return new_visual_dm


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


def save_estimates(filename, estimates, mask_indices, orig_num_vert = 1974, model_type = 'gauss'):
    
    """
    re-arrange estimates that were masked
    and save all in numpy file
    
    Parameters
    ----------
    filename : str
        absolute filename of estimates to be saved
    estimates : arr
        2d estimates that were obtained (datapoints,estimates)
    mask_indices : arr
        larray with voxel indices that were NOT masked out
    orig_num_vert: int
        original number of datapoints
        
    Outputs
    -------
    out_file: str
        absolute output filename
    
    """ 
    
    final_estimates = np.zeros((orig_num_vert, estimates.shape[-1])); final_estimates[:] = np.nan
    
    counter = 0
    
    for i in range(orig_num_vert):
        if i <= mask_indices[-1]:
            if i == mask_indices[counter]:
                final_estimates[i] = estimates[counter]
                counter += 1
            
    if model_type == 'gauss':
        
        np.savez(filename,
                 x = final_estimates[..., 0],
                 y = final_estimates[..., 1],
                 size = final_estimates[..., 2],
                 betas = final_estimates[...,3],
                 baseline = final_estimates[..., 4],
                 r2 = final_estimates[..., 5])
    
    elif model_type == 'css':
        np.savez(filename,
                 x = final_estimates[..., 0],
                 y = final_estimates[..., 1],
                 size = final_estimates[..., 2],
                 betas = final_estimates[...,3],
                 baseline = final_estimates[..., 4],
                 ns = final_estimates[..., 5],
                 r2 = final_estimates[..., 6])
        
        
def fit_glm_get_t_stat(voxel, dm, contrast):
    
    """ GLM fit on timeseries
    Regress a created design matrix on the input_data.
    +
    and compute simple contrast
    (used to compute run-level analyses)
    
    Parameters
    ----------
    voxel : arr
        timeseries of a single voxel
    dm : arr
        DM array (#TR,#regressors)
    contrast: arr
        contrast vector
    
    Outputs
    -------
    betas : arr
        betas for model
    r2 : arr
        coefficient of determination
    t_val : float
        t-statistic for that voxel relative to contrast
    cb: effect size
        dot product of the contrast vector and the parameters
    
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

    if np.isnan(voxel).any() or np.isnan(dm).any():
        betas = np.repeat(np.nan, dm.shape[-1]-1) # betas from the fit
        r2 = np.nan
        t_val = np.nan # t-statistic 
        cb = np.nan # effect_size => dot product of the contrast vector and the parameters
        effect_var = np.nan

    else:   # if not nan (some vertices might have nan values)
        
        ######### FIT GLM ###########
        betas = np.linalg.lstsq(dm, voxel, rcond = -1)[0]
        prediction = dm.dot(betas)

        mse = np.mean((voxel - prediction) ** 2) # calculate mean of squared residuals
        r2 = pearsonr(prediction, voxel)[0] ** 2 # and the rsq
        
        ##### COMPUTE STATS ON CONTRAST #########
        # calculate design variance
        design_var = design_variance(dm, contrast)
        
        # sum of squared errors
        sse = ((voxel - (dm.dot(betas))) ** 2).sum() 
        
        #degrees of freedom = N - P = timepoints - predictores
        df = (dm.shape[0] - dm.shape[1])
        
        # effect size
        cb = contrast.dot(betas)

        # effect variance
        effect_var = (sse/df) * design_var
        
        # t statistic for vertex
        t_val = cb / np.sqrt(effect_var)
    
    return betas, r2, t_val, cb, effect_var
    

def mean_epi_gii(list_filenames, outdir):

    """ Make mean EPI over the time course
    to make a vasculature map
    (will normalize mean EPI, and average across runs too)
    
    Parameters
    ----------
    list_filenames : list
        list of absolute filenames, for all runs, both hemispheres
    outdir : str
        path to directory where to store median EPI
    
    Outputs
    -------
    out_filenames : list
        list with computed filenames
    
    """
    
    out_filenames = []
    
    # laod each hemi
    for field in ['hemi-L', 'hemi-R']:
        
        # get all func files for that hemi
        hemi_files = [h for h in list_filenames if field in h]
    
        hemi_all = []
        
        for _,file in enumerate(hemi_files):
            
            # load data for a run
            img_load = nb.load(file)
            run_data = np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]) #load surface data

            # average the EPI time course 
            mean_data = np.mean(run_data, axis=0)

            # normalize image by dividing the value of each vertex by the value of the vertex with the maximum intensity
            norm_data = normalize(mean_data)

            # append normalized run
            hemi_all.append(norm_data)
        
        # average al runs
        median_img = np.mean(hemi_all, axis = 0)

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in median_img]
        median_gii = nb.gifti.gifti.GiftiImage(header = img_load.header,
                                               extra = img_load.extra,
                                               darrays = darrays) # need to save as gii again

        median_file = op.join(outdir, re.sub('run-\d{2}_','run-median_',op.split(file)[-1]).replace('.func.gii','_meanEPI.func.gii'))
        nb.save(median_gii,median_file)

        print('computed %s'%median_file)
        out_filenames.append(median_file)
        
    return out_filenames


def weighted_avg(data_arr, weight_arr, main_axis = 0):

    """
    helper function to do a weighted average along an axis
    """
    return np.average(data_arr, axis = main_axis, weights = weight_arr)


def split_half_comb(input_list):

    """ make list of lists, by spliting half
    and getting all unique combinations
    
    Parameters
    ----------
    input_list : list/arr
        list of items
    Outputs
    -------
    unique_pairs : list/arr
        list of tuples
    
    """

    A = list(itertools.combinations(input_list, int(len(input_list)/2)))
    
    combined_pairs = []
    for pair in A:
        combined_pairs.append(tuple([pair, tuple([r for r in input_list if r not in pair])]))

    # get unique pairs
    seen = set()
    unique_pairs = [t for t in combined_pairs if tuple(sorted(t)) not in seen and not seen.add(tuple(sorted(t)))]

    return unique_pairs

def correlate_arrs(data1_arr, data2_arr, n_jobs = 4, weights=[], shuffle_axis = None):
    
    """
    Compute Pearson correlation between two numpy arrays
    
    Parameters
    ----------
    data1 : list/array
        numpy array 
    data2 : list/array
        same as data1
    n_jobs : int
        number of jobs for parallel
    
    """ 
    
    # if we indicate an axis to shuffle, then do so
    if shuffle_axis is not None:

        if shuffle_axis == -1:
            data_shuf1 = data1_arr.T.copy()
            np.random.shuffle(data_shuf1)
            data1_arr = data_shuf1.T.copy()

            data_shuf2 = data2_arr.T.copy()
            np.random.shuffle(data_shuf2)
            data2_arr = data_shuf2.T.copy()

        elif shuffle_axis == 0:
            np.random.shuffle(data1_arr)
            np.random.shuffle(data2_arr)
    
    ## actually correlate
    correlations = np.array(Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(data1_arr[i], data2_arr[i]) for i in np.arange(data1_arr.shape[0])))[...,0,1]
            
    return correlations


