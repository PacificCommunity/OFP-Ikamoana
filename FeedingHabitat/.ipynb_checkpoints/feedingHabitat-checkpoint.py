# -*- coding: utf-8 -*-
"""
@Author : Jules Lehodey
@Date   : 18/05/2021
"""

from math import sin, asin, cos, acos, tan, atan, pi, sqrt
import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
import numpy as np

class FeedingHabitat :
    
###############################################################################
# -------------------------------  ATTRIBUTS  --------------------------------#
###############################################################################
    
    coords             = None
    global_mask        = None
    cohorts_number     = None
    layers_number      = None
    root_repertory     = None
    output_directory   = None
    # First cohort is number 0 and last is number K-1
    cohorts_to_compute = None
    partial_oxygen_time_axis = False
    
    variables_dictionary = {"forage_epi"          : None,
                            "forage_meso"         : None,
                            "forage_mmeso"        : None,
                            "forage_bathy"        : None,
                            "forage_mbathy"       : None,
                            "forage_hmbathy"      : None,
                            "temperature_L1"      : None,
                            "temperature_L2"      : None,
                            "temperature_L3"      : None,
                            "oxygen_L1"           : None,
                            "oxygen_L2"           : None,
                            "oxygen_L3"           : None,
                            "days_length"         : None,
                            "cohorts_mean_length" : None,
                            "cohorts_mean_weight" : None}
    
    parameters_dictionary = {"eF_list"  : None,
                             "sigma_0"  : None,
                             "sigma_K"  : None,
                             "T_star_1" : None,
                             "T_star_K" : None,
                             "bT"       : None,
                             "gamma"    : None,
                             "o_star"   : None}
    
    
###############################################################################    
# -----------------------------  INITIALIZATION  -----------------------------#
###############################################################################

    def __init__(self) :
        pass

###############################################################################
###############################################################################

    def __readXmlConfigFilepaths__(self, root) :

        # TEMPERATURE #########################################################
        temperature_filepaths = []
        for temp_file in root.find('strfile_t').attrib.values() :
            temperature_filepaths.append(self.root_repertory+temp_file)
        
        # OXYGEN ##############################################################
        oxygen_filepaths = []
        for oxy_file in root.find('strfile_oxy').attrib.values() :
            oxygen_filepaths.append(self.root_repertory+oxy_file)
        
        if root.find('type_oxy').attrib['value'] == '1' :
            self.partial_oxygen_time_axis = True
        
        # FORAGE ##############################################################
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
        
        for x in range(self.layers_number) :
            for y in reversed(range(x+1)) :
                tmp_position = val_list.index((str(x),str(y)))
                ordered_forage.append(key_list[tmp_position])
        
        forage_filepaths = []
        forage_repertory = root.find('strdir_forage').attrib['value']
        for forage in ordered_forage :
            forage_filepaths.append(self.root_repertory+forage_repertory+'Fbiom_'+forage+".nc")
            
        # MASK ################################################################
        mask_filepath = self.root_repertory+root.find('str_file_mask').attrib['value']
        
        return temperature_filepaths, oxygen_filepaths, forage_filepaths, ordered_forage, mask_filepath

###############################################################################
###############################################################################

    def __readXmlConfigParameters__(self, root, ordered_forage) :
        
        # LENGTHS #############################################################
        buffer_length = []
        for length in root.find('length')[0].text.split(' ') :
            if length != '' :
                buffer_length.append(float(length))
        self.variables_dictionary['cohorts_mean_length'] = np.array(buffer_length)
        
        # WEIGHT ##############################################################
        buffer_weight = []
        for weight in root.find('weight')[0].text.split(' ') :
            if weight != '' :
                buffer_weight.append(float(weight))
        self.variables_dictionary['cohorts_mean_weight'] = np.array(buffer_weight)
        
        # eF ##################################################################
        eF = root.find('eF_habitat')
        eF_list = []
        for element in ordered_forage :
            tmp_eF = eF.find(element)
            eF_list.append(float(list(tmp_eF.attrib.values())[0]))
        self.parameters_dictionary["eF_list"] = eF_list

        # SIGMA ###############################################################
        #sigma 0
        self.parameters_dictionary["sigma_0"] = float(list(root.find('a_sst_spawning').attrib.values())[0])
        #sigma K
        self.parameters_dictionary["sigma_K"] = float(list(root.find('a_sst_habitat').attrib.values())[0])        
        # T* ##################################################################
        #T* 1
        self.parameters_dictionary["T_star_1"] = float(list(root.find('b_sst_spawning').attrib.values())[0])
        #T* K
        self.parameters_dictionary["T_star_K"] = float(list(root.find('b_sst_habitat').attrib.values())[0])
        
        # bT ##################################################################
        self.parameters_dictionary["bT"] = float(list(root.find('T_age_size_slope').attrib.values())[0])
        
        # GAMMA ###############################################################
        self.parameters_dictionary["gamma"] = float(list(root.find('a_oxy_habitat').attrib.values())[0])
        
        # O* ##################################################################
        self.parameters_dictionary["o_star"] = float(list(root.find('b_oxy_habitat').attrib.values())[0])

###############################################################################
###############################################################################

    def __loadVariablesFromFilepaths__(self, temperature_filepaths, oxygen_filepaths, forage_filepaths, mask_filepath) :
        
        # COORDS ##############################################################
        self.coords = xr.open_dataset(temperature_filepaths[0]).coords
        
        # OXYGEN ##############################################################
        self.variables_dictionary["oxygen_L1"] = np.nan_to_num(
            xr.open_dataarray(
                oxygen_filepaths[0]).data)
        self.variables_dictionary["oxygen_L2"] = np.nan_to_num(
            xr.open_dataarray(
                oxygen_filepaths[1]).data)
        self.variables_dictionary["oxygen_L3"] = np.nan_to_num(
            xr.open_dataarray(
                oxygen_filepaths[2]).data)
        
        # TEMPERATURE #########################################################
        self.variables_dictionary["temperature_L1"] = np.nan_to_num(
            xr.open_dataarray(
                temperature_filepaths[0]).data)
        self.variables_dictionary["temperature_L2"] = np.nan_to_num(
            xr.open_dataarray(
                temperature_filepaths[1]).data)
        self.variables_dictionary["temperature_L3"] = np.nan_to_num(
            xr.open_dataarray(
                temperature_filepaths[2]).data)
        
        # FORAGE ##############################################################
        self.variables_dictionary["forage_epi"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[0]).data)
        self.variables_dictionary["forage_meso"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[1]).data)
        self.variables_dictionary["forage_mmeso"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[2]).data)
        self.variables_dictionary["forage_bathy"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[3]).data)
        self.variables_dictionary["forage_mbathy"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[4]).data)
        self.variables_dictionary["forage_hmbathy"] = np.nan_to_num(
            xr.open_dataarray(
                forage_filepaths[5]).data)
        
        # MASK ################################################################
        self.setMask(from_text=mask_filepath)        
        
        # DAY LENGTH ##########################################################
        # TODO enlever le False
        self.daysLength(float_32=False)
        
###############################################################################
###############################################################################
    
    #TODO : présenter le fonctionnement global
    def helpMe(self) :
        pass

###############################################################################
###############################################################################
    
    def loadFromXml(self, xml_filepath, partial_cohorts_computation=None) :
        
        # WARNING : partial_cohorts_computation start with value greater or equal to 1
        
        # INITIALIZATION ######################################################
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        
        self.root_repertory = root.find('strdir').attrib['value']
        self.output_directory = root.find('strdir_output').attrib['value']
        
        self.cohorts_number = sum([int(x) for x in root.find('nb_cohort_life_stage')[0].text.split(' ')])
        self.layers_number  = int(root.find('nb_layer').attrib['value'])
        
        if partial_cohorts_computation is not None :
            self.cohorts_to_compute = set(partial_cohorts_computation)
        
        #######################################################################
        # Variables Filepaths #################################################
        
        (temperature_filepaths, 
         oxygen_filepaths, 
         forage_filepaths,
         ordered_forage,
         mask_filepath) = self.__readXmlConfigFilepaths__(root)
        
        #######################################################################
        # Parameters ##########################################################
        
        self.__readXmlConfigParameters__(root, ordered_forage)
        
        #######################################################################
        # LOADING #############################################################
        
        self.__loadVariablesFromFilepaths__(temperature_filepaths, oxygen_filepaths,
                                            forage_filepaths,mask_filepath)
        
        

###############################################################################
###############################################################################
    def setMask(self, from_text=None, expend_time=True) :
        # 0 = ground
        # 1 = L1
        # 2 = L2
        # 3 = L3
        
        # Read mask values ####################################################
        if from_text is None :
            tmp_mask = (  np.isfinite(self.variables_dictionary["temperature_L1"][0,:,:]).astype(np.int)
                        + np.isfinite(self.variables_dictionary["temperature_L2"][0,:,:]).astype(np.int)
                        + np.isfinite(self.variables_dictionary["temperature_L3"][0,:,:]).astype(np.int))            
        else :
            tmp_mask = np.loadtxt(from_text)
        
        # Add a time axis to broadcast mask on variables ######################
        # i.e. Numpy broadcast method (Numpy)
        if expend_time :
            tmp_mask = tmp_mask[np.newaxis,:,:]
        
        # Separate each layer #################################################
        self.global_mask = {"mask_L1" : ((tmp_mask == 1) + (tmp_mask == 2) + (tmp_mask == 3)),
                            "mask_L2" : ((tmp_mask == 2) + (tmp_mask == 3)),
                            "mask_L3" : (tmp_mask == 3)}
        del tmp_mask

###############################################################################
###############################################################################
        
    #TODO : non essentiel dans un premier temps, sélectionne une plage de temps
    #       pour limiter le calcul lors de l'optimisation de parametres
    def setTimeSerie(self, end, start=0) :
        pass

###############################################################################
###############################################################################

    #TODO : non essentiel dans un premier temps, limite l'espace de recherche
    #       pour limiter le calcul lors de l'optimisation de parametres
    def setParametersBoundaries(self, boundaries_list) :
        pass

###############################################################################
###############################################################################

    def __dayLength__(self, jday, lat) :
        
        # The CBM model of Forsythe et al, Ecological Modelling 80 (1995) 87-95
        
        # trvolution angle for the day of the year
        theta = 0.2163108 + 2*(atan(0.9671396 * tan(0.00860*(jday-186))))
        
        # sun's declination angle, or the angular distance at solar noon between the 
        # Sun and the equator, from the Earth orbit revolution angle
        phi = asin(0.39795 * cos (theta))
        
        # Angle between the sun position and the horizon, in degrees
        # 6  - civil twilight || 12 - nautical twilight || 18 - astronomical twilight
        p = 6;
        
        #daylength computed according to 'p'
        arg = (sin(pi*p/180)+sin(lat*pi/180)*sin(phi))/(cos(lat*pi/180)*cos(phi))
        if arg > 1.0 :
            arg = 1.0
        if arg < -1.0 :
            arg = -1.0
        day_length = 24.0-(24.0/pi)*acos(arg)

        return day_length / 24.0
    
###############################################################################
###############################################################################

    def __dayLengthPISCES__(self, jday, lat) :
        # -------------------------------------------------------------------
    	#  New function provided by Laurent Bopp as used in the PISCES model
    	#
    	#  PURPOSE :compute the day length depending on latitude and the day
    	#  --------
    	#
    	#   MODIFICATIONS:
    	#   --------------
    	#      original       : E. Maier-Reimer (GBC 1993)
    	#      additions      : C. Le Quere (1999)
    	#      modifications  : O. Aumont (2004)
    	#	Adapted to C      : P. Lehodey (2005)
        #   Adapted to python : J. Lehodey (2021)
    	#-------------------------------------------------------------------
        
        rum = (jday - 80.0) / 365.25
        delta = sin(rum * pi * 2.0) * sin(pi * 23.5 / 180.0)
        codel = asin(delta)
        phi = lat * pi / 180.0
        
        argu = tan(codel) * tan(phi)
        argu = min(1.,argu)
        argu = max(-1.,argu)
        
        DL = 24.0 - (2.0 * acos(argu) * 180.0 / pi / 15 )
        DL = max(DL,0.0)
        
        return DL
    
###############################################################################
###############################################################################

    # Base the time, latitude and longitude on model
    # If model is None, base on self.coords
    def daysLength(self, model=None, float_32=True, PISCES=False) :
        
        if model is not None :
            days_of_year = pd.DatetimeIndex(model['time'].data.astype('datetime64[D]')).dayofyear
            latitude = model['lat'].data
            longitude = model['lon'].data
        else :
            days_of_year = pd.DatetimeIndex(self.coords['time'].data.astype('datetime64[D]')).dayofyear
            latitude = self.coords['lat'].data
            longitude = self.coords['lon'].data
        
        buffer_list = []
        for day in days_of_year :
            for lat in latitude :
                if not PISCES :
                    day_length = self.__dayLength__(day, lat)
                else :
                    day_length = self.__dayLengthPISCES__(day, lat)
                if float_32 :
                    day_length = np.float32(day_length)
                buffer_list.extend([day_length] * len(longitude))
                
        days_length = np.ndarray((len(days_of_year),len(latitude),len(longitude)),
                                     buffer=np.array(buffer_list),
                                     dtype=(np.float32 if float_32 else np.float64))
        
        del buffer_list

        self.variables_dictionary['days_length'] = xr.DataArray(data=days_length,
                    dims=["time", "lat", "lon"],
                    coords=dict(lon=longitude, lat=latitude, time=self.coords['time']),
                    attrs=dict(description="Day length.", units="hour"))

###############################################################################    
# ---------------------------  TRANSFORMATION  ------------------------------ #
###############################################################################

    # Return sigmaStar (the termal tolerance intervals, i.e. standard deviation)
    # for each cohorts.
    def sigmaStar(self, sigma_0, sigma_K) :
        
        cohorts_mean_weight = self.variables_dictionary['cohorts_mean_weight']
        last_cohorts_mean_weight = cohorts_mean_weight[-1]
        
        return sigma_0 + ( (sigma_K - sigma_0) * (cohorts_mean_weight / last_cohorts_mean_weight) )
    
###############################################################################
###############################################################################
    
    # Return T_star (optimal temperature, i.e. mean) for each cohorts
    # Dim = nb_cohorts
    def tStar(self, T_star_1, T_star_K, bT) :
        
        cohorts_mean_length = self.variables_dictionary['cohorts_mean_length']
        last_cohorts_mean_length = cohorts_mean_length[-1]
        
        return T_star_1 - ( (T_star_1 - T_star_K) * ((cohorts_mean_length / last_cohorts_mean_length)**bT) )

###############################################################################
###############################################################################

    # Return accessibility for each cohorts to each layer according to temperature
    # Dim = nb_cohorts * 3 * Time * Latitude * Longitude
    def temperature(self, sigma_0, sigma_K, T_star_1, T_star_K, bT) :

        layers = ["temperature_L1", "temperature_L2","temperature_L3"]
        
        cohort_buffer = []
        sigma_star = self.sigmaStar(sigma_0, sigma_K)
        T_star = self.tStar(T_star_1, T_star_K, bT)
        
        if self.cohorts_to_compute is None :
            cohorts_tab = range(self.cohorts_number)
        else :
            cohorts_tab = np.array(self.cohorts_to_compute)
        
        for cohort in cohorts_tab :
            layer_buffer = []
            for layer, mask in zip(layers, self.global_mask.keys()) :
                sigma_star_a = sigma_star[cohort]
                T_star_a = T_star[cohort]
                variable = self.variables_dictionary[layer]
                
                if sigma_star_a == 0 :
                    layer_buffer.append(np.zeros_like(variable))
                else :
                    layer_buffer.append(
                        np.exp(
                            (- np.power((variable - T_star_a), 2) ) / (2.0 * (sigma_star_a**2)),
                            out=np.zeros_like(variable),
                            where=self.global_mask[mask]
                            )
                        )
            cohort_buffer.append(layer_buffer)
        
        return np.array(cohort_buffer) 

###############################################################################
###############################################################################

    # Return accessibility for the Nth cohort to each layer according to temperature
    # Dim = 3 * Time * Latitude * Longitude
    def temperatureNthCohort(self, sigma_0, sigma_K, T_star_1, T_star_K, bT, Nth_cohort) :

        layers = ["temperature_L1", "temperature_L2","temperature_L3"]

        sigma_star = self.sigmaStar(sigma_0, sigma_K)
        T_star = self.tStar(T_star_1, T_star_K, bT)
        sigma_star_a = sigma_star[Nth_cohort]
        T_star_a = T_star[Nth_cohort]

        layer_buffer = []
        for layer, mask in zip(layers, self.global_mask.keys()) :
            
            variable = self.variables_dictionary[layer]
            if sigma_star_a == 0 :
                layer_buffer.append(np.zeros_like(variable))
            else :
                layer_buffer.append(
                    np.exp(
                          (- np.power((variable - T_star_a), 2) )
                          / (2.0 * (sigma_star_a**2)),
                          out=np.zeros_like(variable),
                          where=self.global_mask[mask])
                    )
                
        return np.array(layer_buffer) 

###############################################################################
###############################################################################

    # Return accessibility for each cohorts to each layer according to oxygen
    # Dim = nb_cohorts * 3 * Time * Latitude * Longitude
    def oxygen(self, gamma, o_star) :

        layers = ["oxygen_L1", "oxygen_L2", "oxygen_L3"]
        cohort_buffer = []
        
        if self.cohorts_to_compute is None :
            cohorts_number = self.cohorts_number
        else :
            cohorts_number = len(self.cohorts_to_compute)
            
        for _ in range(cohorts_number) :
            layer_buffer = []
            for layer, mask in zip(layers, self.global_mask.keys()) :
                variable = self.variables_dictionary[layer]
                layer_buffer.append(
                    np.where(
                        self.global_mask[mask],
                        1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
                        0.0)
                    )                                          
            cohort_buffer.append(layer_buffer)

        return np.array(cohort_buffer)

###############################################################################
###############################################################################

    # Return accessibility for the Nth cohort to each layer according to oxygen
    # Dim = 3 * Time * Latitude * Longitude
    def oxygenNthCohort(self, gamma, o_star) :

        layers = ["oxygen_L1", "oxygen_L2", "oxygen_L3"]

        layer_buffer = []
        for layer, mask in zip(layers, self.global_mask.keys()) :
            variable = self.variables_dictionary[layer]
            #Take into account the case where oxygen time serie is shorter
            layer_buffer.append(
                np.where(
                    self.global_mask[mask],
                    1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
                    0.0)
                )
            
        result = np.array(layer_buffer)
        
        return result 

###############################################################################
###############################################################################

    # Return forage quantity and weight accessibility in each layer 
    # Dim = 2 * 3 * Time * Latitude * Longitude
    def forage(self, forage_preference_coefficients) :
        
        # Initialization
        days_length = self.variables_dictionary['days_length']
        night_length = np.ones_like(days_length) - days_length

        #L1 ###################################################################
        epi     = self.variables_dictionary["forage_epi"]     * forage_preference_coefficients[0]
        mmeso   = self.variables_dictionary["forage_mmeso"]   * forage_preference_coefficients[2]
        hmbathy = self.variables_dictionary["forage_hmbathy"] * forage_preference_coefficients[5]
        
        day_l1  = days_length * epi
        night_l1  = night_length * (epi + mmeso + hmbathy)
        
        tmp_L1 = np.add(day_l1, night_l1, out=np.zeros_like(days_length), where=self.global_mask['mask_L1'])
        del epi, day_l1, night_l1

        #L2 ###################################################################
        meso    = self.variables_dictionary["forage_meso"]    * forage_preference_coefficients[1]
        mbathy  = self.variables_dictionary["forage_mbathy"]  * forage_preference_coefficients[4]
        
        day_l2  = days_length * (meso + mmeso)
        night_l2  = night_length * (meso + mbathy)
        
        tmp_L2 = np.add(day_l2, night_l2, out=np.zeros_like(days_length), where=self.global_mask['mask_L2'])
        del meso, mmeso, day_l2, night_l2

        #L3 ###################################################################
        bathy   = self.variables_dictionary["forage_bathy"]   * forage_preference_coefficients[3]
        
        day_l3  = days_length * (bathy + mbathy + hmbathy)
        night_l3  = night_length * (bathy)
        
        tmp_L3 = np.add(day_l3, night_l3, out=np.zeros_like(days_length), where=self.global_mask['mask_L3'])
        del bathy, mbathy, hmbathy, day_l3, night_l3, night_length

        # Result ##############################################################
        result = np.array([tmp_L1, tmp_L2, tmp_L3])
        del tmp_L1, tmp_L2, tmp_L3
    
        return result

###############################################################################
###############################################################################

    def Ha(self, fpc, gamma, o_star, sigma_0, sigma_K, T_star_1, T_star_K, bT) :
        
        # TEMPERATURE #########################################################
        print('Temperature')
        ha_result = self.temperature(sigma_0, sigma_K, T_star_1, T_star_K, bT)
        
        # OXYGEN ##############################################################
        print('Oxygen')
        ha_oxygen = self.oxygen(gamma, o_star)
        if self.partial_oxygen_time_axis :
            buffer = []
            for time_unit in range(ha_result.shape[2]) :
                buffer.append(ha_result[:,:,time_unit,:,:] * ha_oxygen[:,:,time_unit % ha_oxygen.shape[2],:,:])
            ha_result = np.transpose(np.array(buffer),(1,2,0,3,4)) 
            del buffer
        else :
            ha_result *= ha_oxygen
        
        del ha_oxygen

        # FORAGE ##############################################################    
        print('Forage')
        ha_forage = self.forage(fpc)
        ha_result *= ha_forage[np.newaxis, :, :, :, :]
        
        ha_result = np.sum(ha_result, axis=1)

        del ha_forage

        return np.where(self.global_mask['mask_L1'], ha_result, np.NaN)
    
###############################################################################
###############################################################################

    def saveHaStepByStep(self, verbose=True) :
        
        # FORAGE ##############################################################
        if verbose :
            print('Computing  forage')
        ha_forage = self.forage(self.parameters_dictionary['eF_list'])
        
        # OXYGEN ##############################################################
        if verbose :
            print('Computing oxygen')
        ha_oxygen = self.oxygenNthCohort(self.parameters_dictionary['gamma'], self.parameters_dictionary['o_star'])
        if self.partial_oxygen_time_axis :
            buffer = []
            for time_unit in range(ha_forage.shape[2]) :
                buffer.append(ha_oxygen[:,time_unit % ha_oxygen.shape[1],:,:])
            ha_oxygen = np.transpose(np.array(buffer),(1,0,2,3))
        
        # Select the cohorts to compute the feeding habitat
        if self.cohorts_to_compute is None :
            cohorts_range = range(self.cohorts_number)
        else :
            cohorts_range = self.cohorts_to_compute
        
        # Compute then save in netCDF file
        for cohort in cohorts_range:
            
            # TEMPERATURE #####################################################
            if verbose :
                print('Cohort %d : computing temperature' % cohort)
            ha_temperature = self.temperatureNthCohort(self.parameters_dictionary['sigma_0'],
                                                       self.parameters_dictionary['sigma_K'], 
                                                       self.parameters_dictionary['T_star_1'], 
                                                       self.parameters_dictionary['T_star_K'], 
                                                       self.parameters_dictionary['bT'],
                                                       Nth_cohort=cohort)
            
            # SAVE ############################################################
            if verbose :
                print('Cohort %d : saving Ha in NetCDF file' % cohort)
            # TODO : Retirer l'ajout de +1e-4
            ha_result = np.where(self.global_mask['mask_L1'],
                                 self.scaling(np.sum(ha_forage * (ha_temperature * ha_temperature + 1e-4), axis=0)),
                                 np.NaN)
            
            da_to_save = xr.DataArray(
                data=ha_result,
                dims=["time", "lat", "lon"],
                coords=dict(
                    lon=self.coords['lon'],
                    lat=self.coords['lat'],
                    time=self.coords['time']),
                attrs=dict(description=("Ha_cohort_%d" %cohort)))
            da_to_save.to_netcdf(self.root_repertory + self.output_directory+("ha_cohort_%d.nc" % cohort))
            
        
###############################################################################
# ---------------------------------  TOOLS  --------------------------------- #
###############################################################################        
        
    def scaling(self, data) :
        """
        Data will be rescaled between [0 ; 1].
        Approximation : If data > 1 then 1 Else data

        Parameters
        ----------
        data : Numpy Array
            DESCRIPTION.

        Returns
        -------
        Rescaled Numpy Array

        """
        # parameters of hyperbola
        phi = 22.5*pi/180.0;
        a = 0.07;
        e = 1.0/cos(phi);
        b = a*sqrt(e*e-1.0);
    
        # coordinate center
        # shift is to have all y>=0
        x0 = 1.0-0.00101482322788;
        y0 = 1.0;
    
        # equation for hyperbola
        sinsq = sin(phi)*sin(phi);
        cossq = 1.0-sinsq;
        rasq  = 1.0/(a*a);
        rbsq  = 1.0/(b*b);
        A = sinsq*rasq - cossq*rbsq;
        B = -2.0*(data-x0)*cos(phi)*sin(phi)*(rasq+rbsq);
        C = 1.0-(data-x0)*(data-x0)*(sinsq*rbsq-cossq*rasq);
    
        return(y0+(B+np.sqrt(B*B-4.0*A*C))/(2*A));





