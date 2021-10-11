# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:41:54 2021

@author: julesl
"""

import xml.etree.ElementTree as ET
tree = ET.parse('./../Data_Test/skj_po_interim_2deg_configuration_file.xml')
root = tree.getroot()


#%% Root Directory

rep_root = root.find('strdir').attrib['value']
print(rep_root)

nb_cohorts = sum([int(x) for x in root.find('nb_cohort_life_stage')[0].text.split(' ')])
nb_layers = int(root.find('nb_layer').attrib['value'])

print(nb_cohorts)
print(nb_layers)


#%% Temperature

for temp_file in root.find('strfile_t').attrib.values() :
    print(rep_root+temp_file)


#%% Oxygen

for oxy_file in root.find('strfile_oxy').attrib.values() :
    print(rep_root+oxy_file)

tmp = root.find('type_oxy').attrib['value']
if tmp == '0' :
    print('time series')
if tmp == '1' :
    print('monthly')

del tmp


#%% Forage

## Order

a = root.find('day_layer').attrib
b = root.find('night_layer').attrib
print(a)
print(b)

tmp_ordered_forage = {}
ordered_forage = []

for k_a, v_a in a.items() :
    for k_b, v_b in b.items() :
        if k_a == k_b :
            tmp_ordered_forage[k_a] = (v_a, v_b)
print(tmp_ordered_forage)

key_list = list(tmp_ordered_forage.keys())
val_list = list(tmp_ordered_forage.values())

for x in range(nb_layers) :
    for y in reversed(range(x+1)) :
        position = val_list.index((str(x),str(y)))
        ordered_forage.append(key_list[position])

del tmp_ordered_forage
print(ordered_forage)

#                                                                            #
#   SHOULD FIND --> ['meso', 'epi', 'hmbathy', 'bathy', 'mbathy', 'mmeso']   #
#                                                                            #

nb_layers = int(root.find('nb_layer').attrib['value'])
tmp_ordered_forage = {}
ordered_forage = []

for k_a, v_a in a.items() :
    for k_b, v_b in b.items() :
        if k_a == k_b :
            tmp_ordered_forage[k_a] = (v_a, v_b)

tmp_ordered_forage['epi'] = ('1','1')
tmp_ordered_forage['meso'] = ('0','0')
tmp_ordered_forage['mmeso'] = ('2','0')
tmp_ordered_forage['hmbathy'] = ('1','0')

print(tmp_ordered_forage)

key_list = list(tmp_ordered_forage.keys())
val_list = list(tmp_ordered_forage.values())

for x in range(nb_layers) :
    for y in reversed(range(x+1)) :
        position = val_list.index((str(x),str(y)))
        ordered_forage.append(key_list[position])

del tmp_ordered_forage
print(ordered_forage)

{'epi': ('1', '1'), 'meso': ('0', '0'), 'mmeso': ('2', '0'), 'bathy': ('2', '2'), 'mbathy': ('2', '1'), 'hmbathy': ('1', '0')}
['meso', 'epi', 'hmbathy', 'bathy', 'mbathy', 'mmeso']

## File

rep_forage = root.find('strdir_forage').attrib['value']
print(rep_forage)

for forage in ordered_forage :
    print(rep_root+rep_forage+'Fbiom_'+forage+".nc")


#%% Mask

rep_root+root.find('str_file_mask').attrib['value']


#%% Output Directory

root.find('strdir_output').attrib['value']


#%% Lengths

buffer_length = []
for length in root.find('length')[0].text.split(' ') :
    if length != '' :
        buffer_length.append(float(length))

print(buffer_length)


#%% Weigths

buffer_weigth = []
for weigth in root.find('weight')[0].text.split(' ') :
    if weigth != '' :
        buffer_weigth.append(float(weigth))

print(buffer_weigth)


#%% Parameters
## eF

eF = root.find('eF_habitat')
eF_list = []

for element in ordered_forage :
    tmp_eF = eF.find(element)
#    print(tmp_eF)
#    print(tmp_eF.attrib,end='\n\n')
    eF_list.append(list(tmp_eF.attrib.values())[0])

print(eF_list)

## Sigma

#sigma 0
print(root.find('a_sst_spawning').attrib)
print(list(root.find('a_sst_spawning').attrib.values())[0],end='\n\n')

#sigma K
print(root.find('a_sst_habitat').attrib)
print(list(root.find('a_sst_habitat').attrib.values())[0],end='\n\n')

## T*

#T*_0
print(root.findall('b_sst_spawning')[0].attrib)
print(list(root.findall('b_sst_spawning')[0].attrib.values())[0],end='\n\n')

#T*_K
print(root.findall('b_sst_habitat')[0].attrib)
print(list(root.findall('b_sst_habitat')[0].attrib.values())[0],end='\n\n')

## bT

print(root.findall('T_age_size_slope')[0].attrib)
print(list(root.findall('T_age_size_slope')[0].attrib.values())[0],end='\n\n')

## Gamma

print(root.findall('a_oxy_habitat')[0].attrib)
print(list(root.findall('a_oxy_habitat')[0].attrib.values())[0],end='\n\n')

## O*

print(root.findall('b_oxy_habitat')[0].attrib)
print(list(root.findall('b_oxy_habitat')[0].attrib.values())[0],end='\n\n')