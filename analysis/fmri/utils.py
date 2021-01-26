
# useful functions to use in other scripts

import os, re
import numpy as np

import nibabel as nb


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





