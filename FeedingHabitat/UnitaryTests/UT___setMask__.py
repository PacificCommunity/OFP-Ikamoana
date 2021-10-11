# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:30:17 2021

@author: julesl
"""

#%%
from sys import path
import numpy as np

if not ("./../" in path) :
    path.append("./../")
    
from feedingHabitat import FeedingHabitat

fh = FeedingHabitat()
mask_filepath = './../Data_Test/po_interim_2deg_mask_short_noIO.txt'
fh.__setMask__(from_text=mask_filepath)

#%%
print("Usable cells for mask L1 : %d / %d" % (np.sum(fh.global_mask['mask_L1']), fh.global_mask['mask_L1'].size))
print("Usable cells for mask L2 : %d / %d" % (np.sum(fh.global_mask['mask_L2']), fh.global_mask['mask_L2'].size))
print("Usable cells for mask L3 : %d / %d" % (np.sum(fh.global_mask['mask_L3']), fh.global_mask['mask_L3'].size))