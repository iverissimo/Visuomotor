Repository for Visuomotor mapping in frontal cortex

*Under Development*

# Run An Experiment:
To run the experiments, the package exptools needs to be installed (https://github.com/VU-Cog-Sci/exptools).
In the terminal, go to the folder of the desired experiment. Then type ```python main.py <INITIALS>```, replacing `INITIALS` with your own (or the participants') initials.

# Analysis

Structural images were first preprocessed by resorting to https://github.com/VU-Cog-Sci/mp2rage_preprocessing . The resulting anatomical - and functional - images were preprocessed with fMRIprep (https://fmriprep.org/en/latest/index.html)

pRF fitting was done through the package prfpy (https://github.com/VU-Cog-Sci/prfpy) 

Analysis scripts were developed with Python 3.6
