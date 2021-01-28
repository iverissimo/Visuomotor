
# useful functions to use in other scripts

import os, re
import numpy as np

import shutil

import nibabel as nb

from nilearn import surface

from scipy.signal import savgol_filter
import nipype.interfaces.freesurfer as fs

import cv2

from skimage import color
import cv2
from skimage.transform import rescale
from skimage.filters import threshold_triangle

from PIL import Image, ImageOps

import math



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


def smooth_gii(gii_file, outdir, fwhm = 5, extension = '.func.gii'):

    """ percent signal change gii file

    Parameters
    ----------
    gii_file : str
        absolute filename for gii
    outdir: str
        path to save new files
    fwhm: int
        width of the kernel, at half of the maximum of the height of the Gaussian
    extension: str
        file extension

    Outputs
    -------
    smooth_gii: arr
        np array with smoothed file
    smooth_gii_pth: str
        absolute path for smoothed file
    
    """

    outfile = os.path.split(gii_file)[-1].replace(extension,'_smooth%d'%fwhm+extension)
    smooth_gii_pth = os.path.join(outdir,outfile)

    if not os.path.isfile(gii_file): # check if file exists
            print('no file found called %s' %gii_file)
            smooth_gii = []
            smooth_gii_pth = []

    elif os.path.isfile(smooth_gii_pth): # if psc file exists, skip
        print('File {} already in folder, skipping'.format(smooth_gii_pth))
        smooth_gii = nb.load(smooth_gii_pth)
        smooth_gii = np.array([smooth_gii.darrays[i].data for i in range(len(smooth_gii.darrays))]) #load surface data

    else:

        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(gii_file)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        print('loading file %s' %gii_file)

        # first need to convert to mgz
        # will be saved in output dir
        new_mgz = os.path.join(outdir,os.path.split(gii_file)[-1].replace(extension,'.mgz'))

        print('converting gifti to mgz as %s' %(new_mgz))
        os.system('mri_convert %s %s'%(gii_file,new_mgz))

        # now smooth it
        smoother = fs.SurfaceSmooth()
        smoother.inputs.in_file = new_mgz
        smoother.inputs.subject_id = 'fsaverage'

        # define hemisphere
        smoother.inputs.hemi = 'lh' if '_hemi-L' in new_mgz else 'rh'
        print('smoothing %s' %smoother.inputs.hemi)
        smoother.inputs.fwhm = fwhm
        smoother.run() # doctest: +SKIP

        new_filename = os.path.split(new_mgz)[-1].replace('.mgz','_smooth%d.mgz'%(smoother.inputs.fwhm))
        smooth_mgz = os.path.join(outdir,new_filename)
        shutil.move(os.path.join(os.getcwd(),new_filename), smooth_mgz) #move to correct dir

        # transform to gii again
        smooth_gii = surface.load_surf_data(smooth_mgz).T
        smooth_gii = np.array(smooth_gii)

        smooth_gii_pth = smooth_mgz.replace('.mgz',extension)
        print('converting to %s' %smooth_gii_pth)
        os.system('mri_convert %s %s'%(smooth_mgz,smooth_gii_pth))

    return smooth_gii,smooth_gii_pth


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
    # Very clunky and non-generic function, but works.
    # should optimize eventually

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



    