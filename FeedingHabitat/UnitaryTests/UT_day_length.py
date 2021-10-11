# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:02:49 2021

@author: julesl
"""

#%%
from sys import path

if not ("./../" in path) :
    path.append("./../")
    
from feedingHabitat import FeedingHabitat

fh = FeedingHabitat()
xmlPath = './../Data_Test/skj_po_interim_2deg_configuration_file.xml'
fh.loadFromXml(xmlPath)

#%%
fh.daysLength()

fh.variables_dictionary['days_length'].sel(time='1979-01-15T12:00:00', lon=131.5)

CBM_lat_05 = fh.variables_dictionary['days_length'].sel(time='1979-01-15T12:00:00', lat=0.5)

#%%
fh.daysLength(PISCES=True)

fh.variables_dictionary['days_length'].sel(time='1979-01-15T12:00:00', lon=131.5)

PISCES_lat_05 = fh.variables_dictionary['days_length'].sel(time='1979-01-15T12:00:00', lat=0.5)

#%%
print(PISCES_lat_05 - CBM_lat_05 )

## There is ~1 hour difference between those day length calculation methods