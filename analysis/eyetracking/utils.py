
# useful functions to use in other scripts

import os
import hedfpy


def edf2h5(edf_files, hdf_file, pupil_hp = 0.01, pupil_lp = 6.0):
    
    """ convert edf files (can be several)
    into one hdf5 file, for later analysis
    
    Parameters
    ----------
    edf_files : List/arr
        list of absolute filenames for edf files
    hdf_file : str
        absolute filename of output hdf5 file

    Outputs
    -------
    all_alias: List
        list of strings with alias for each run
    
    """
    
    # first check if hdf5 already exists
    if os.path.isfile(hdf_file):
        print('The file %s already exists, skipping'%hdf_file)
        
    else:
        ho = hedfpy.HDFEyeOperator(hdf_file)

        all_alias = []

        for ef in edf_files:
            alias = os.path.splitext(os.path.split(ef)[1])[0] #name of data for that run
            ho.add_edf_file(ef)
            ho.edf_message_data_to_hdf(alias = alias) #write messages ex_events to hdf5
            ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = pupil_hp, pupil_lp = pupil_lp) #add raw and preprocessed data to hdf5   

            all_alias.append(alias)
    
    return all_alias