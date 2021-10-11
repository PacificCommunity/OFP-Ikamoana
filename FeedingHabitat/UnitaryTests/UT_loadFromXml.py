# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:52:12 2021

@author: julesl
"""

#%%
from sys import path

if not ("./../" in path) :
    path.append("./../")
    
from feedingHabitat import FeedingHabitat

fh = FeedingHabitat()

#%%
xmlPath = './../Data_Test/skj_po_interim_2deg_configuration_file.xml'

fh.loadFromXml(xmlPath)

#%%
for key, value in fh.variables_dictionary.items() :
    print(key, ' : ' ,type(value))

#%%
print(fh.parameters_dictionary)