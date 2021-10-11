# -*- coding: utf-8 -*-
"""
@Author : Jules Lehodey
@Date   : 02/08/2021

Summary
-------
This module is used by the feedingHabitat module to setup all variables and
parameters. It use a XML configuration file that link to all the netCDF used
to compute de feeding habitat.
It also compute the day length in the area using the PISCES method.

See Also
--------
- feedingHabitat : class used to compute feeding habitat

Reference
---------
[1] E. Maier-Reimer (GBC 1993) - Day length

"""

from math import sin, asin, acos, tan, pi
import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
import numpy as np


###############################################################################
# ---------------------------- READ XML FILE -------------------------------- #
###############################################################################



def loadMask(variables_dictionary, from_text=None, expend_time=True) :
    """
    Load a mask file (i.e. texte file) as a dictionary of numpy array. Each
    array corresponds to a pelagic layer (i.e. L1, L2 and L3). Mask file must contains 4
    values :
        - 0 = Ground / Land / etc...
        - 1 = L1 / epipelagic
        - 2 = L2 / mesopelagic superior
        - 3 = L3 / mesopelagic inferior

    Parameters
    ----------
    variables_dictionary : TYPE
        DESCRIPTION.
    from_text : string, optional
        Text file used to compute the mask. If from_text is None, the Nan
        values of the temperature_L(1, 2 and 3) netCDF will be used.
        The default is None.
    expend_time : boolean, optional
        Add a time axis to broadcast mask on variables. Cf. Numpy broadcast.
        The default is True.

    Returns
    -------
    global_mask : TYPE
        DESCRIPTION.

    """

    # Read mask values ####################################################
    if from_text is None :
        tmp_mask = (  np.isfinite(variables_dictionary["temperature_L1"][0,:,:]).astype(np.int)
                    + np.isfinite(variables_dictionary["temperature_L2"][0,:,:]).astype(np.int)
                    + np.isfinite(variables_dictionary["temperature_L3"][0,:,:]).astype(np.int))
    else :
        tmp_mask = np.loadtxt(from_text)

    # Add a time axis to broadcast mask on variables ######################
    # i.e. Numpy broadcast method (Numpy)
    if expend_time :
        tmp_mask = tmp_mask[np.newaxis,:,:]

    # Separate each layer #################################################
    global_mask = {"mask_L1" : ((tmp_mask == 1) + (tmp_mask == 2) + (tmp_mask == 3)),
                   "mask_L2" : ((tmp_mask == 2) + (tmp_mask == 3)),
                   "mask_L3" : (tmp_mask == 3)}

    del tmp_mask

    return global_mask



def __dayLengthPISCES__(jday, lat) :
    """
    Compute the day length depending on latitude and the day. New function
    provided by Laurent Bopp as used in the PISCES model and used by SEAPODYM
    in 2020.

    Parameters
    ----------
    jday : float
        Day of the year.
    lat : float
        Latitude.

    Modification
    ------------
    original       : E. Maier-Reimer (GBC 1993)
	additions      : C. Le Quere (1999)
	modifications  : O. Aumont (2004)
    	Adapted to C      : P. Lehodey (2005)
        Adapted to Python : J. Lehodey (2021)

    Returns
    -------
    float
        The duration of the day (i.e. while the sun is shining) as a ratio in
        range [0,1].

    """

    rum = (jday - 80.0) / 365.25
    delta = sin(rum * pi * 2.0) * sin(pi * 23.5 / 180.0)
    codel = asin(delta)
    phi = lat * pi / 180.0

    argu = tan(codel) * tan(phi)
    argu = min(1.,argu)
    argu = max(-1.,argu)

    day_length = 24.0 - (2.0 * acos(argu) * 180.0 / pi / 15 )
    day_length = max(day_length,0.0)

    return day_length / 24.0



#
#
def daysLength(coords, model=None, float_32=True) :
    """
    Compute the day length.

    Parameters
    ----------
    coords : xarray.DataArray.coords
        DESCRIPTION.
    model : TYPE, optional
        Base the time, latitude and longitude on model.
        If model is None, base on coords (i.e. xarray.DataArray.coords).
        The default is None.
    float_32 : boolean, optional
        Set the time in float 32 or 64. If True, it is float 32.
        The default is True.

    Returns
    -------
    xarray.DataArray
        Contains the day length of each position on the studied area.

    """

    if model is not None :
        days_of_year = pd.DatetimeIndex(model['time'].data.astype('datetime64[D]')).dayofyear
        latitude = model['lat'].data
        longitude = model['lon'].data
    else :
        days_of_year = pd.DatetimeIndex(coords['time'].data.astype('datetime64[D]')).dayofyear
        latitude = coords['lat'].data
        longitude = coords['lon'].data

    buffer_list = []
    for day in days_of_year :
        for lat in latitude :
            day_length = __dayLengthPISCES__(day, lat)
            if float_32 :
                day_length = np.float32(day_length)
            buffer_list.extend([day_length] * len(longitude))

    days_length = np.ndarray((len(days_of_year),len(latitude),len(longitude)),
                                 buffer=np.array(buffer_list),
                                 dtype=(np.float32 if float_32 else np.float64))

    return xr.DataArray(data=days_length,dims=["time", "lat", "lon"],
                        coords=dict(lon=longitude,lat=latitude,time=coords['time']),
                        attrs=dict(description="Day length.",units="hour"))



def readXmlConfigFilepaths(root, root_directory, layers_number) :

    # TEMPERATURE #############################################################
    temperature_filepaths = []
    for temp_file in root.find('strfile_t').attrib.values() :
        temperature_filepaths.append(root_directory +temp_file)

    # OXYGEN ##################################################################
    oxygen_filepaths = []
    for oxy_file in root.find('strfile_oxy').attrib.values() :
        oxygen_filepaths.append(root_directory+oxy_file)

    partial_oxygen_time_axis = int(root.find('type_oxy').attrib['value'])

    # FORAGE ##################################################################
    tmp_day = root.find('day_layer').attrib
    tmp_night = root.find('night_layer').attrib

    tmp_ordered_forage = {}
    ordered_forage = []

    for k_a, v_a in tmp_day.items() :
        for k_b, v_b in tmp_night.items() :
            if k_a == k_b :
                tmp_ordered_forage[k_a] = (v_a, v_b)

    key_list = list(tmp_ordered_forage.keys())
    val_list = list(tmp_ordered_forage.values())

    for x in range(layers_number) :
        for y in reversed(range(x+1)) :
            tmp_position = val_list.index((str(x),str(y)))
            ordered_forage.append(key_list[tmp_position])

    forage_filepaths = []
    forage_directory  = root.find('strdir_forage').attrib['value']
    for forage in ordered_forage :
        forage_filepaths.append(root_directory+forage_directory+'Fbiom_'+forage+".nc")

    # SST #####################################################################
    if root.find('strfile_sst') is None :
        sst_filepath = None
    else :
        sst_filepath = root_directory + root.find('strfile_sst').attrib['value']

    # ZEU #####################################################################
    if root.find('strfile_zeu') is None :
        zeu_filepath = None
    else :
        zeu_filepath = root_directory + root.find('strfile_zeu').attrib['value']

    # MASK ####################################################################
    if root.find('strfile_mask') == None :
        mask_filepath = None
    else :
        mask_filepath = root_directory + root.find('strfile_mask').attrib['value']

    return (temperature_filepaths, oxygen_filepaths, forage_filepaths, sst_filepath,
            zeu_filepath, ordered_forage, mask_filepath, partial_oxygen_time_axis)



def readXmlConfigParameters(root, ordered_forage) :

    parameters_dictionary = {}

    # eF ######################################################################
    eF = root.find('eF_habitat')
    eF_list = []
    for element in ordered_forage :
        tmp_eF = eF.find(element)
        eF_list.append(float(list(tmp_eF.attrib.values())[0]))
    parameters_dictionary["eF_list"] = eF_list

    # SIGMA ###################################################################
    #sigma 0
    parameters_dictionary["sigma_0"] = float(list(root.find('a_sst_spawning').attrib.values())[0])
    #sigma K
    parameters_dictionary["sigma_K"] = float(list(root.find('a_sst_habitat').attrib.values())[0])

    # T* ######################################################################
    #T* 1
    parameters_dictionary["T_star_1"] = float(list(root.find('b_sst_spawning').attrib.values())[0])
    #T* K
    parameters_dictionary["T_star_K"] = float(list(root.find('b_sst_habitat').attrib.values())[0])

    # bT ######################################################################
    parameters_dictionary["bT"] = float(list(root.find('T_age_size_slope').attrib.values())[0])

    # GAMMA ###################################################################
    parameters_dictionary["gamma"] = float(list(root.find('a_oxy_habitat').attrib.values())[0])

    # O* ######################################################################
    parameters_dictionary["o_star"] = float(list(root.find('b_oxy_habitat').attrib.values())[0])

    return parameters_dictionary



def loadVariablesFromFilepaths(root, temperature_filepaths, oxygen_filepaths,
                               forage_filepaths, sst_filepath, zeu_filepath,
                               mask_filepath, float_32=True) :

    variables_dictionary = {}

    # COORDS ##################################################################
    coords = xr.open_dataset(temperature_filepaths[0]).coords

    # OXYGEN ##################################################################
    variables_dictionary["oxygen_L1"] = np.nan_to_num(
        xr.open_dataarray(
            oxygen_filepaths[0]).data)
    variables_dictionary["oxygen_L2"] = np.nan_to_num(
        xr.open_dataarray(
            oxygen_filepaths[1]).data)
    variables_dictionary["oxygen_L3"] = np.nan_to_num(
        xr.open_dataarray(
            oxygen_filepaths[2]).data)

    # TEMPERATURE #############################################################
    variables_dictionary["temperature_L1"] = np.nan_to_num(
        xr.open_dataarray(
            temperature_filepaths[0]).data)
    variables_dictionary["temperature_L2"] = np.nan_to_num(
        xr.open_dataarray(
            temperature_filepaths[1]).data)
    variables_dictionary["temperature_L3"] = np.nan_to_num(
        xr.open_dataarray(
            temperature_filepaths[2]).data)

    # FORAGE ##################################################################
    # WARNING : 0.000001 is added on forage layer to copy SEAPODYM behavior
    # TODO : 0.000001 is added on forage layer to copy SEAPODYM behavior
    variables_dictionary["forage_epi"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[0]).data) + 0.000001
    variables_dictionary["forage_meso"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[1]).data) + 0.000001
    variables_dictionary["forage_mmeso"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[2]).data) + 0.000001
    variables_dictionary["forage_bathy"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[3]).data) + 0.000001
    variables_dictionary["forage_mbathy"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[4]).data) + 0.000001
    variables_dictionary["forage_hmbathy"] = np.nan_to_num(
        xr.open_dataarray(
            forage_filepaths[5]).data) + 0.000001

    # SST #####################################################################
    variables_dictionary["sst"] = np.nan_to_num(
        xr.open_dataarray(sst_filepath).data)

    # ZEU #####################################################################
    variables_dictionary["zeu"] = np.nan_to_num(
        xr.open_dataarray(zeu_filepath).data)

    # LENGTHS #################################################################
    buffer_length = []
    for length in root.find('length')[0].text.replace('\n', ' ' ).replace('\t', ' ' ).split(' ') :
        if length != '' :
            buffer_length.append(float(length))
    variables_dictionary['cohorts_mean_length'] = np.array(buffer_length)

    # WEIGHT ##################################################################
    buffer_weight = []
    for weight in root.find('weight')[0].text.replace('\n', ' ' ).replace('\t', ' ' ).split(' ') :
        if weight != '' :
            buffer_weight.append(float(weight))
    variables_dictionary['cohorts_mean_weight'] = np.array(buffer_weight)

    # MASK ####################################################################
    global_mask = loadMask(variables_dictionary, from_text=mask_filepath)

    # DAY LENGTH ##############################################################
    variables_dictionary['days_length'] = daysLength(coords, float_32=float_32)

    return variables_dictionary, global_mask, coords



###############################################################################
# ----------------------------- MAIN FUNCTION ------------------------------- #
###############################################################################

def loadFromXml(xml_filepath, partial_cohorts_computation=None, float_32=True) :
    """
    This is the main function used by the feedingHabitat module. It is used to
    read the XML configuration file and load all variables and parameters in
    the main class and also compute the day length.

TODO : Complet DocString
    Parameters
    ----------
    xml_filepath : TYPE
        DESCRIPTION.
    partial_cohorts_computation : TYPE, optional
        DESCRIPTION. The default is None.
    float_32 : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    root_directory : TYPE
        DESCRIPTION.
    output_directory : TYPE
        DESCRIPTION.
    layers_number : TYPE
        DESCRIPTION.
    cohorts_number : TYPE
        DESCRIPTION.
    cohorts_to_compute : TYPE
        DESCRIPTION.
    partial_oxygen_time_axis : TYPE
        DESCRIPTION.
    global_mask : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.
    variables_dictionary : TYPE
        DESCRIPTION.
    parameters_dictionary : TYPE
        DESCRIPTION.

    """
    # WARNING : partial_cohorts_computation start with value greater or equal to 0

    # INITIALIZATION ##########################################################
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    root_directory  = root.find('strdir').attrib['value']
    output_directory = root.find('strdir_output').attrib['value']

    cohorts_number = sum([int(x) for x in root.find('nb_cohort_life_stage')[0].text.split(' ')])
    layers_number  = int(root.find('nb_layer').attrib['value'])

    cohorts_to_compute = None
    if partial_cohorts_computation is not None :
        cohorts_to_compute = set(partial_cohorts_computation)


    # Variables Filepaths #####################################################
    (temperature_filepaths,
     oxygen_filepaths,
     forage_filepaths,
     sst_filepath,
     zeu_filepath,
     ordered_forage,
     mask_filepath,
     partial_oxygen_time_axis) = readXmlConfigFilepaths(root, root_directory, layers_number)

    # Parameters ##############################################################
    parameters_dictionary = readXmlConfigParameters(root, ordered_forage)

    # Variables ###############################################################
    variables_dictionary, global_mask, coords = loadVariablesFromFilepaths(root,
                                                                           temperature_filepaths,
                                                                           oxygen_filepaths,
                                                                           forage_filepaths,
                                                                           sst_filepath,
                                                                           zeu_filepath,
                                                                           mask_filepath,
                                                                           float_32)

    return (root_directory, output_directory,  layers_number, cohorts_number,
            cohorts_to_compute, partial_oxygen_time_axis, global_mask, coords,
            variables_dictionary, parameters_dictionary)
