# -*- coding: utf-8 -*-

"""
Summary
-------
This module is implementing the FeedingHabitat class which can simulate the
feeding habitat the same way as the SEAPODYM model (2020-08). This class start
with a initialization using the readFiles module, then performe computation
for each cohort.

"""
import os
from typing import List, Union
from . import feedinghabitatconfigreader as fhcr
from . import habitatdatastructure as hds
from math import sin, cos, pi, sqrt
import xarray as xr
import numpy as np


class FeedingHabitat :
    
    def __init__(self,
                 xml_filepath: str,
                 partial_cohorts_computation: List[int] = None,
                 float_32: bool = True) -> None :
        """
        Initialize the FeedingHabitat instance according to the XML 
        configuration file passed in argument.

        Parameters
        ----------
        xml_filepath : string
            The pass to the XML configuration file.
        partial_cohorts_computation : list of int, optional
            If you want to perform a partial feeding habitat computation, you
            can  specify a group of cohort using a number corresponding to the
            position in the cohort list.
            Warning : The first cohort is number 0.
            For example, if you want to compute the feeding habitat of the
            second and third cohort : partial_cohorts_computation = [1,2].
            The default is None.
        float_32 : boolean, optional
            Specify if the data in NetCDF files are in float 32 (True) or float
            64 (False).
            The default is True.
        
        Returns
        -------
        None.
        """

        self.data_structure = hds.HabitatDataStructure(
            fhcr.loadFromXml(xml_filepath,partial_cohorts_computation,float_32)
        )

###############################################################################
    
    def __sigmaStar__(self, sigma_0, sigma_K) :
        """Return sigmaStar (the termal tolerance intervals, i.e. standard
        deviation) for each cohorts."""
        
        cohorts_mean_weight = self.data_structure.species_dictionary['cohorts_mean_weight']
        max_weight = np.max(cohorts_mean_weight)

        return sigma_0 + ((sigma_K - sigma_0)
                          * (cohorts_mean_weight / max_weight))
    
    def __tStar__(self, T_star_1, T_star_K, bT) :
        """Return T_star (optimal temperature, i.e. mean) for each cohorts"""
        
        cohorts_mean_length = self.data_structure.species_dictionary['cohorts_mean_length']
        max_length = np.max(cohorts_mean_length)
        
        return T_star_1 - ((T_star_1 - T_star_K)
                           * ((cohorts_mean_length / max_length)**bT))

# TODO : Supprimer si plus utile
    # def __temperatureNthCohort__(
    #         self, sigma_0, sigma_K, T_star_1,
    #         T_star_K, bT, Nth_cohort) :
    #     """Return accessibility for the Nth cohort to each layer according to
    #     temperature"""
        
    #     layers = ["temperature_L1", "temperature_L2","temperature_L3"]

    #     sigma_star = self.__sigmaStar__(sigma_0, sigma_K)
    #     T_star = self.__tStar__(T_star_1, T_star_K, bT)
    #     sigma_star_a = sigma_star[Nth_cohort]
    #     T_star_a = T_star[Nth_cohort]

    #     layer_buffer = []
    #     for layer, mask in zip(layers, self.data_structure.global_mask.keys()) :
            
    #         variable = self.data_structure.variables_dictionary[layer]
    #         layer_buffer.append(
    #             np.exp(
    #                   (- np.power((variable - T_star_a), 2) )
    #                   / (2.0 * (sigma_star_a**2)),
    #                   out=np.zeros_like(variable),
    #                   where=self.data_structure.global_mask[mask])
    #             )
                
    #     return np.array(layer_buffer) 
    
    # def __temperatureNthCohortAtDate__(
    #         self, position, sigma_0, sigma_K, T_star_1,
    #         T_star_K, bT, Nth_cohort) :
    #     """Return accessibility for the Nth cohort at date to each layer
    #     according to temperature"""
        
    #     layers = ["temperature_L1", "temperature_L2","temperature_L3"]

    #     sigma_star = self.__sigmaStar__(sigma_0, sigma_K)
    #     T_star = self.__tStar__(T_star_1, T_star_K, bT)
    #     sigma_star_a = sigma_star[Nth_cohort]
    #     T_star_a = T_star[Nth_cohort]

    #     layer_buffer = []
    #     for layer, mask in zip(layers, self.data_structure.global_mask.keys()) :
            
    #         variable = self.data_structure.variables_dictionary[layer][position]
    #         layer_buffer.append(
    #             np.exp(
    #                   (- np.power((variable - T_star_a), 2) )
    #                   / (2.0 * (sigma_star_a**2)),
    #                   out=np.zeros_like(variable),
    #                   where=self.data_structure.global_mask[mask][0])
    #             )
                
    #     return np.array(layer_buffer)
    
    # def __oxygen__(self, gamma, o_star) :
    #     """Return accessibility for the Nth cohort to each layer
    #     according to oxygen"""
        
    #     layers = ["oxygen_L1", "oxygen_L2", "oxygen_L3"]

    #     layer_buffer = []
    #     for layer, mask in zip(layers, self.data_structure.global_mask.keys()) :
    #         variable = self.data_structure.variables_dictionary[layer]
    #         #Take into account the case where oxygen time serie is shorter
    #         layer_buffer.append(
    #             np.where(
    #                 self.data_structure.global_mask[mask],
    #                 1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
    #                 0.0)
    #             )
            
    #     result = np.array(layer_buffer)
        
    #     return result 

    # def __oxygenAtDate__(self, position, gamma, o_star) :
    #     """Return accessibility for the Nth cohort at a date to each
    #     layer according to oxygen"""
        
    #     layers = ["oxygen_L1", "oxygen_L2", "oxygen_L3"]

    #     if self.data_structure.partial_oxygen_time_axis :
    #         position %= 12

    #     layer_buffer = []
    #     for layer, mask in zip(layers, self.data_structure.global_mask.keys()) :
    #         variable = self.data_structure.variables_dictionary[layer][position] 
    #         layer_buffer.append(
    #             np.where(
    #                 self.global_mask[mask][0],
    #                 1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
    #                 0.0)
    #             )
        
    #     return np.array(layer_buffer)

    # def __forage__(self, forage_preference_coefficients) :
    #     """Return forage quantity and weight accessibility in each layer"""
        
    #     # Initialization
    #     days_length = self.data_structure.variables_dictionary['days_length'].data
    #     night_length = np.ones_like(days_length) - days_length

    #     #L1 ###################################################################
    #     epi     = (self.data_structure.variables_dictionary["forage_epi"]
    #                * forage_preference_coefficients[0]).data
    #     mumeso   = (self.data_structure.variables_dictionary["forage_mumeso"]
    #                * forage_preference_coefficients[2]).data
    #     hmlmeso = (self.data_structure.variables_dictionary["forage_hmlmeso"]
    #                * forage_preference_coefficients[5]).data
        
    #     day_l1  = days_length * epi
    #     night_l1  = night_length * (epi + mumeso + hmlmeso)

    #     tmp_L1 = np.add(day_l1, night_l1, out=np.zeros_like(days_length),
    #                     where=self.data_structure.global_mask['mask_L1'])
    #     del epi, day_l1, night_l1

    #     #L2 ###################################################################
    #     umeso    = (self.data_structure.variables_dictionary["forage_umeso"]
    #                * forage_preference_coefficients[1]).data
    #     mlmeso  = (self.data_structure.variables_dictionary["forage_mlmeso"]
    #                * forage_preference_coefficients[4]).data
        
    #     day_l2  = days_length * (umeso + mumeso)
    #     night_l2  = night_length * (umeso + mlmeso)
        
    #     tmp_L2 = np.add(day_l2, night_l2, out=np.zeros_like(days_length),
    #                     where=self.data_structure.global_mask['mask_L2'])
    #     del umeso, mumeso, day_l2, night_l2

    #     #L3 ###################################################################
    #     lmeso   = (self.data_structure.variables_dictionary["forage_lmeso"]
    #                * forage_preference_coefficients[3]).data
        
    #     day_l3  = days_length * (lmeso + mlmeso + hmlmeso)
    #     night_l3  = night_length * (lmeso)
        
    #     tmp_L3 = np.add(day_l3, night_l3, out=np.zeros_like(days_length),
    #                     where=self.data_structure.global_mask['mask_L3'])
    #     del lmeso, mlmeso, hmlmeso, day_l3, night_l3, night_length

    #     # Result ##############################################################
    #     result = np.array([tmp_L1, tmp_L2, tmp_L3])
    #     del tmp_L1, tmp_L2, tmp_L3
    
    #     return result
    
    # def __forageAtDate__(self, date, position, forage_preference_coefficients) :
        """Return forage quantity and weight accessibility in each layer """
        
        # Initialization
        days_length = self.data_structure.variables_dictionary['days_length'].sel(time=[date])
        night_length = np.ones_like(days_length) - days_length

        #L1 ###################################################################
        epi     = (self.data_structure.variables_dictionary["forage_epi"][position]
                   * forage_preference_coefficients[0])
        mumeso   = (self.data_structure.variables_dictionary["forage_mumeso"][position]
                   * forage_preference_coefficients[2])
        hmlmeso = (self.data_structure.variables_dictionary["forage_hmlmeso"][position]
                   * forage_preference_coefficients[5])
        
        day_l1  = days_length * epi
        night_l1  = night_length * (epi + mumeso + hmlmeso)
        
        tmp_L1 = np.add(day_l1, night_l1, out=np.zeros_like(days_length),
                        where=self.data_structure.global_mask['mask_L1'][0])
        del epi, day_l1, night_l1

        #L2 ###################################################################
        umeso    = (self.data_structure.variables_dictionary["forage_umeso"][position]
                   * forage_preference_coefficients[1])
        mlmeso  = (self.data_structure.variables_dictionary["forage_mlmeso"][position]
                   * forage_preference_coefficients[4])
        
        day_l2  = days_length * (umeso + mumeso)
        night_l2  = night_length * (umeso + mlmeso)
        
        tmp_L2 = np.add(day_l2, night_l2, out=np.zeros_like(days_length),
                        where=self.data_structure.global_mask['mask_L2'][0])
        del umeso, mumeso, day_l2, night_l2

        #L3 ###################################################################
        lmeso   = (self.data_structure.variables_dictionary["forage_lmeso"][position]
                   * forage_preference_coefficients[3])
        
        day_l3  = days_length * (lmeso + mlmeso + hmlmeso)
        night_l3  = night_length * (lmeso)
        
        tmp_L3 = np.add(day_l3, night_l3, out=np.zeros_like(days_length),
                        where=self.data_structure.global_mask['mask_L3'][0])
        del lmeso, mlmeso, hmlmeso, day_l3, night_l3, night_length

        # Result ##############################################################
        result = np.array([tmp_L1, tmp_L2, tmp_L3])
        del tmp_L1, tmp_L2, tmp_L3
    
        return result

# New Version : 

    def _selSubDataArray(self,
                         data_array: xr.DataArray,
                         time_start: int = None, time_end: int = None,
                         lat_min: int = None, lat_max: int = None,
                         lon_min: int = None, lon_max: int = None) -> xr.DataArray :
        
        return data_array.sel(
                time=data_array.time.data[time_start:time_end if time_end is None else time_end+1 ],
                lat=data_array.lat.data[lat_min:lat_max if lat_max is None else lat_max+1],
                lon=data_array.lon.data[lon_min:lon_max if lon_max is None else lon_max+1])

    def _selSubMask(self,
                    mask: str,
                    lat_min: int = None, lat_max: int = None,
                    lon_min: int = None, lon_max: int = None) -> np.ndarray :

            tmp = self.data_structure.global_mask[mask]

            return tmp[:,
                       lat_min:lat_max if lat_max is None else lat_max+1,
                       lon_min:lon_max if lon_max is None else lon_max+1]

    def _temperature(self,
                     cohort: int,
                     time_start: int = None, time_end: int = None,
                     lat_min: int = None, lat_max: int = None,
                     lon_min: int = None, lon_max: int = None) -> np.ndarray :
        
        sigma_star = self.__sigmaStar__(
            self.data_structure.parameters_dictionary['sigma_0'],
            self.data_structure.parameters_dictionary['sigma_K'])
        T_star = self.__tStar__(
            self.data_structure.parameters_dictionary['T_star_1'],
            self.data_structure.parameters_dictionary['T_star_K'],
            self.data_structure.parameters_dictionary['bT'])
        
        sigma_star = sigma_star[cohort]
        T_star = T_star[cohort]

        layer_buffer = []
        iterate = zip(['temperature_L1', 'temperature_L2', 'temperature_L3'],
                      ['mask_L1', 'mask_L2', 'mask_L3'])

        for layer_name, mask_name in iterate :
            variable = self._selSubDataArray(self.data_structure.variables_dictionary[layer_name],time_start,time_end,lat_min,lat_max,lon_min,lon_max)
            
            mask = self._selSubMask(mask_name, lat_min, lat_max, lon_min, lon_max)

            layer_buffer.append(
                np.exp(
                    ((- np.power((variable - T_star), 2))
                     / (2.0 * (sigma_star**2))),
                    out=np.zeros_like(variable),
                    where=mask)
                )

        return np.array(layer_buffer)

    def _oxygen(self,
                time_start: int = None, time_end: int = None,
                lat_min: int = None, lat_max: int = None,
                lon_min: int = None, lon_max: int = None) -> np.ndarray :

        gamma = self.data_structure.parameters_dictionary['gamma']
        o_star = self.data_structure.parameters_dictionary['o_star']

        layer_buffer = []
        iterate = zip(["oxygen_L1", "oxygen_L2", "oxygen_L3"],
                      ['mask_L1', 'mask_L2', 'mask_L3'])

        for layer_name, mask_name in iterate :

            variable = self.data_structure.variables_dictionary[layer_name]

            if self.data_structure.partial_oxygen_time_axis :
                partial_size = variable.time.data.size
                coords_size = self.data_structure.coords['time'].data.size
                quotient = coords_size // partial_size
                variable = np.tile(variable, (quotient, 1, 1))

                variable = variable[time_start:time_end if time_end is None else time_end+1,
                                    lat_min:lat_max if lat_max is None else lat_max+1,
                                    lon_min:lon_max if lon_max is None else lon_max+1]

            else :
                variable = self._selSubDataArray(
                    variable,time_start,time_end,lat_min,lat_max,lon_min,lon_max)
            
            mask = self._selSubMask(mask_name, lat_min, lat_max, lon_min, lon_max)
            
            layer_buffer.append(
                np.where(
                    mask,
                    1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
                    0.0)
                )
        
        return np.array(layer_buffer) 

    def _forage(self,
                time_start: int = None, time_end: int = None,
                lat_min: int = None, lat_max: int = None,
                lon_min: int = None, lon_max: int = None) -> np.ndarray :

        # Initialize ##########################################################
        days_length = self._selSubDataArray(
            self.data_structure.variables_dictionary['days_length'],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max)

        night_length = np.ones_like(days_length) - days_length

        epi = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_epi"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        epi = epi * self.data_structure.parameters_dictionary['eF_list'][0]

        umeso = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_umeso"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        umeso = umeso * self.data_structure.parameters_dictionary['eF_list'][1]

        mumeso = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_mumeso"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        mumeso = mumeso * self.data_structure.parameters_dictionary['eF_list'][2]

        lmeso = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_lmeso"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        lmeso = lmeso * self.data_structure.parameters_dictionary['eF_list'][3]

        mlmeso = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_mlmeso"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        mlmeso = mlmeso * self.data_structure.parameters_dictionary['eF_list'][4]

        hmlmeso = self._selSubDataArray(
            self.data_structure.variables_dictionary["forage_hmlmeso"],
            time_start,time_end,lat_min,lat_max,lon_min,lon_max).data
        hmlmeso = hmlmeso * self.data_structure.parameters_dictionary['eF_list'][5]

        mask_L1 = self._selSubMask('mask_L1', lat_min, lat_max, lon_min, lon_max)
        mask_L2 = self._selSubMask('mask_L2', lat_min, lat_max, lon_min, lon_max)
        mask_L3 = self._selSubMask('mask_L3', lat_min, lat_max, lon_min, lon_max)  

        # Compute Layers ######################################################
        layer_1 = np.add(
            days_length * epi,
            night_length * (epi + mumeso + hmlmeso),
            out=np.zeros_like(days_length),
            where=mask_L1)

        layer_2 = np.add(
            days_length * (umeso + mumeso),
            night_length * (umeso + mlmeso),
            out=np.zeros_like(days_length),
            where=mask_L2)

        layer_3 = np.add(
            days_length * (lmeso + mlmeso + hmlmeso),
            night_length * (lmeso),
            out=np.zeros_like(days_length),
            where=mask_L3)

        return np.array([layer_1, layer_2, layer_3])

###############################################################################
# ---------------------------------  MAIN  ---------------------------------- #
###############################################################################

# TODO :    Terminer cette fonction :
#           - Ajouter des attributs a la description de chaque DataArray comme l'age, le poid, la taille etc..
#           - Ajouter des attributs a la description du DataSet comme les parametres utilisées, lmin et max de lat lon et time ...

    def computeFeedingHabitat(self,
                              cohorts: Union[int, List[int]],
                              time_start: int = None, time_end: int = None,
                              lat_min: int = None, lat_max: int = None,
                              lon_min: int = None, lon_max: int = None) -> xr.DataArray :
        """
        Description
        """

        if isinstance(cohorts, int) :
            cohorts = [cohorts]

        def controlArguments(data_structure: hds.HabitatDataStructure,
                             cohorts,time_start,time_end,lat_min,lat_max,lon_min,lon_max) :
            for elmt in cohorts :
                if elmt < 0 or elmt >= data_structure.cohorts_number :
                    raise ValueError("cohort out of bounds. Min is 0 and Max is %d"%(self.data_structure.cohorts_number-1))

            coords = data_structure.coords
            
            if (lat_min is not None) :
                if ((lat_min < 0) or (lat_min >= coords['lat'].data.size)) :
                    raise ValueError("lat_min out of bounds. Min is %d and Max is %d"%(
                        0, coords['lat'].data.size - 1))
            if (lat_max is not None) :
                if ((lat_max < 0) or (lat_max >= coords['lat'].data.size)) :
                    raise ValueError("lat_max out of bounds. Min is %d and Max is %d"%(
                        0, coords['lat'].data.size - 1))
            if (lat_min is not None) and (lat_max is not None) and (lat_min > lat_max) :
                raise ValueError("lat_min must be <= to lat_max.")

            if (lon_min is not None) :
                if ((lon_min < 0) or (lon_min >= coords['lon'].data.size)) :
                    raise ValueError("lon_min out of bounds. Min is %d and Max is %d"%(
                        0, coords['lon'].data.size - 1))
            if (lon_max is not None) :
                if ((lon_max < 0) or (lon_max >= coords['lon'].data.size)) :
                    raise ValueError("lon_max out of bounds. Min is %d and Max is %d"%(
                        0, coords['lon'].data.size - 1))
            if (lon_min is not None) and (lon_max is not None) and (lon_min > lon_max) :
                raise ValueError("lon_min must be <= to lon_max.")

            if (time_start is not None) :
                if ((time_start < 0) or (time_start >= coords['time'].data.size)) :
                    raise ValueError("time_start out of bounds. Min is %d and Max is %d"%(
                        0, coords['time'].data.size - 1))
            if (time_end is not None) :
                if ((time_end < 0) or (time_end >= coords['time'].data.size)) :
                    raise ValueError("time_end out of bounds. Min is %d and Max is %d"%(
                        0, coords['time'].data.size - 1))
            if (time_start is not None) and (time_end is not None) and (time_start > time_end) :
                raise ValueError("time_start must be <= to time_end.")     

        controlArguments(self.data_structure,
                         cohorts,time_start,time_end,lat_min,lat_max,lon_min,lon_max)
        
        print("OXYGEN : ",self._oxygen(time_start, time_end, lat_min, lat_max, lon_min, lon_max).shape)
        fh_oxygen = self._oxygen(time_start, time_end, lat_min, lat_max, lon_min, lon_max)

        print("FORAGE : ",self._forage(time_start, time_end, lat_min, lat_max, lon_min, lon_max).shape)
        fh_forage = self._forage(time_start, time_end, lat_min, lat_max, lon_min, lon_max)

        result = {}
        mask_L1 = self._selSubMask('mask_L1', lat_min, lat_max, lon_min, lon_max)

        for elmt in cohorts :
            print("TEMPERATURE : ",self._temperature(elmt, time_start, time_end, lat_min, lat_max, lon_min, lon_max).shape)
            fh_temperature = self._temperature(elmt, time_start, time_end, lat_min, lat_max, lon_min, lon_max)

            name = 'Feeding_Habitat_Cohort_%d'%(elmt)

            result_np_array = np.where(
                mask_L1,
                self.__scaling__(
                    np.sum(
                        fh_forage * (fh_temperature * fh_oxygen + 1e-4),
                        axis=0)),
                np.NaN)

            result_xr_data_array = xr.DataArray(
                result_np_array,
                coords=dict(
                    lon=self.data_structure.coords['lon'].data[
                        lon_min:lon_max if lon_max is None else lon_max+1],
                    lat=self.data_structure.coords['lat'].data[
                        lat_min:lat_max if lat_max is None else lat_max+1],
                    time=self.data_structure.coords['time'].data[
                        time_start:time_end if time_end is None else time_end+1]),
                dims=["time", "lat", "lon"],
                attrs={'Cohort number':elmt}
            )
            result[name] = result_xr_data_array

        dataset_attributs = dict(
            time_start=time_start, time_end=time_end,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            )
        dataset_attributs.update(self.data_structure.parameters_dictionary)

        return xr.Dataset(
            result,
            attrs=dataset_attributs
        )


# TODO : Supprimer si plus utile
    # def OBSOLETEcomputeFeedingHabitat(self, filepath=None, verbose=True) :
    #     """
    #     The main function of the FeedingHabitat class. It will compute the
    #     feeding habitat of each cohort specify in the cohorts_to_compute
    #     attribut of the instance.

    #     Parameters
    #     ----------
    #     filepath : string, optional
    #         The filepath where to save all the netCDF we will produce during
    #         this function execution. If None, the output filepath in the XML 
    #         configuration file will be used.
    #         The default is None.
    #     verbose : boolean, optional
    #         If True, print some informations about the running state.
    #         The default is True.

    #     Returns
    #     -------
    #     None.

    #     """
    #     if filepath is None :
    #         path_save = self.data_structure.output_directory
    #     else :
    #         path_save = filepath
        
    #     if verbose : print('Files will be saved at : %s' % path_save)
        
    #     # FORAGE ##############################################################
    #     if verbose : print('Computing  forage')
    #     ha_forage = self.__forage__(self.data_structure.parameters_dictionary['eF_list'])
        
    #     # OXYGEN ##############################################################
    #     if verbose : print('Computing oxygen')
    #     ha_oxygen = self.__oxygen__(self.data_structure.parameters_dictionary['gamma'],
    #                                 self.data_structure.parameters_dictionary['o_star'])
    #     # When oxygen is from climat model (only 1 year)
    #     if self.data_structure.partial_oxygen_time_axis :
    #         buffer = []
    #         for time_unit in range(ha_forage.shape[2]) :
    #             buffer.append(ha_oxygen[:,time_unit % ha_oxygen.shape[1],:,:])
    #         ha_oxygen = np.transpose(np.array(buffer),(1,0,2,3))
        
    #     # Select the cohorts to compute the feeding habitat
    #     if self.data_structure.cohorts_to_compute is None :
    #         cohorts_range = range(self.data_structure.cohorts_number)
    #     else :
    #         cohorts_range = self.data_structure.cohorts_to_compute
        
    #     # For each cohort compute habitat then save in netCDF file
    #     for cohort in cohorts_range:
            
    #         # TEMPERATURE #####################################################
    #         if verbose : print('Cohort %d : computing temperature' % cohort)
    #         ha_temperature = self.__temperatureNthCohort__(
    #             self.data_structure.parameters_dictionary['sigma_0'],
    #             self.data_structure.parameters_dictionary['sigma_K'], 
    #             self.data_structure.parameters_dictionary['T_star_1'],
    #             self.data_structure.parameters_dictionary['T_star_K'], 
    #             self.data_structure.parameters_dictionary['bT'],
    #             Nth_cohort=cohort)
            
    #         # SAVE ############################################################
    #         if verbose : print('Cohort %d : saving Ha in NetCDF file' % cohort)
            
    #         # WARNING :  1e-4 is added on layer access to copy SEAPODYM behavior
    #         # TODO : 1e-4 is added on layer access to copy SEAPODYM behavior
    #         ha_result = np.where(
    #             self.data_structure.global_mask['mask_L1'],
    #             self.__scaling__(
    #                 np.sum(
    #                     ha_forage * (ha_temperature * ha_oxygen + 1e-4),
    #                     axis=0)),
    #             np.NaN)
            
    #         da_to_save = xr.DataArray(
    #             data=ha_result,
    #             dims=["time", "lat", "lon"],
    #             coords=dict(
    #                 lon=self.data_structure.coords['lon'],
    #                 lat=self.data_structure.coords['lat'],
    #                 time=self.data_structure.coords['time']),
    #             attrs=dict(description=("Ha_cohort_%d" %cohort)))
            
    #         if not os.path.exists(path_save):
    #             os.makedirs(path_save)
                
    #         da_to_save.to_netcdf(path_save +("ha_cohort_%d.nc" % cohort))
    

    # def OBSOLETEcomputeFeedingHabitatForSpecificAgeAndDate(
    #         self, age, date, verbose=False) :
    #     """
    #     Compute feeding habitat of a cohort at a specific date.

    #     Parameters
    #     ----------
    #     age : int
    #         Position of the cohort in the length and weight tabs.
    #         The first one (larvae) is at position 0.
    #     date : str, int, numpy.datetime64
    #         3 formats:
    #             - string as "2010-07-17T12:00:00.000000000"
    #             - numpy.datetime64 (see also. FeedingHabitat.coords['time'])
    #             - integer, starting with 0.
    #     verbose : bool, optional
    #         Print some informations about the selected cohort.
    #         The default is False.
            
    #     Notes
    #     -----
    #     If oxygen is climatologic (1 year only), it must begin in January.

    #     Raises
    #     ------
    #     TypeError
    #         If date type is not str, int or numpy.datetime64 an error will be raised.

    #     Returns
    #     -------
    #     ha_result_da : TYPE
    #         The feeding habitat of the Nth cohort (according to age argument)
    #         at a specific date (according to the date argument).

    #     """
        
    #     if isinstance(date, str) :
    #         date = np.datetime64(date)
    #         position = np.where(self.data_structure.coords['time'].data == date)
    #     elif isinstance(date, int) :
    #         position = date
    #         date = self.data_structure.coords['time'].data[date]
    #     elif isinstance(date, np.datetime64) :
    #         position = np.where(self.data_structure.coords['time'].data == date)
    #     else :
    #         raise TypeError("Date must be : String, Integer or Numpy.datetime64")
        
    #     if verbose :
    #         print("At age %d, length is %f and weight is %f."%(age,
    #             self.data_structure.species_dictionary['cohorts_mean_length'][age],
    #             self.data_structure.species_dictionary['cohorts_mean_weight'][age]))
        
    #     # FORAGE ##############################################################
    #     ha_forage = self.__forageAtDate__(
    #         date, position,
    #         self.data_structure.parameters_dictionary['eF_list'])
        
    #     # OXYGEN ##############################################################
    #     ha_oxygen = self.__oxygenAtDate__(
    #         position, self.data_structure.parameters_dictionary['gamma'],
    #         self.data_structure.parameters_dictionary['o_star'])
        
    #     # TEMPERATURE #########################################################
    #     ha_temperature = self.__temperatureNthCohortAtDate__(
    #         position,
    #         self.data_structure.parameters_dictionary['sigma_0'],
    #         self.data_structure.parameters_dictionary['sigma_K'], 
    #         self.data_structure.parameters_dictionary['T_star_1'],
    #         self.data_structure.parameters_dictionary['T_star_K'], 
    #         self.data_structure.parameters_dictionary['bT'],
    #         Nth_cohort=age)
        
    #     # SAVE ################################################################
       
    #     # WARNING : 1e-4 is added on layer access to copy SEAPODYM behavior
    #     # TODO : 1e-4 is added on layer access to copy SEAPODYM behavior.
    #     #        Should it be removed ?
    #     ha_result = np.where(
    #         self.data_structure.global_mask['mask_L1'],
    #         self.__scaling__(
    #             np.sum(
    #                 ha_forage * (ha_temperature * ha_oxygen + 1e-4),
    #                 axis=0)),
    #         np.NaN)
        
    #     ha_result_da = xr.DataArray(
    #             data=ha_result[0],
    #             dims=["lat", "lon"],
    #             coords=dict(lon=self.data_structure.coords['lon'],
    #                         lat=self.data_structure.coords['lat']),
    #             attrs=dict(description=(("Ha_cohort_%d_at_date_"%age)+str(date)),
    #                        date=str(date)))

    #     return ha_result_da 
            
###############################################################################
# ---------------------------------  TOOLS  --------------------------------- #
###############################################################################        
        
    def __scaling__(self, data) :
        """
        The normalization function used by SEAPODYM. Set all values in
        range [0,1].
        Very similar to : If data > 1 then 1 Else data.

        Parameters
        ----------
        data : Numpy.array
            Contains the habitat to normalize.

        Returns
        -------
        Numpy.array
            Return the normalized data.
        """

        # parameters of hyperbola
        phi = 22.5 * pi/180.0
        a = 0.07
        e = 1.0 / cos(phi)
        b = a * sqrt(e*e - 1.0)
    
        # coordinate center
        # shift is to have all y>=0
        x0 = 1.0-0.00101482322788
        y0 = 1.0
    
        # equation for hyperbola
        sinsq = sin(phi) * sin(phi)
        cossq = 1.0-sinsq
        rasq  = 1.0 / (a*a)
        rbsq  = 1.0 / (b*b)
        A = sinsq*rasq - cossq*rbsq
        B = -2.0 * (data-x0) * cos(phi) * sin(phi) * (rasq+rbsq)
        C = 1.0 - (data-x0) * (data-x0) * (sinsq*rbsq - cossq*rasq)
    
        return (y0+(B+np.sqrt(B*B-4.0*A*C))/(2*A))

    def correctEpiTempWithZeu(self) :
        """
        Correct the T_epi temperature by the vertical gradieng magnitude.
        Improves fit in EPO and shallow-thermocline zones.
        Was tested only for SKJ.
        
        Notes
        -----
        Since the estimate of sigma with SST is always lower due to larger
        extension of warm watermasses in the surface, we will add 1.0 to sigma_0.
        
        Reference
        ---------
        Original is from SEAPODYM, Senina et al. (2020)
            Adapted to python : J. Lehodey (2021)
        
        Returns
        -------
        None.
        """
        
        print("Warning : This function (correctEpiTempWithZeu) was only tested"
              + " for Skipjack.\n It will also add +1 to sigma_min. Cf. function"
              + " documentation for more details.")
        
        dTdz = np.divide(2.0 * (self.data_structure.variables_dictionary['sst']
                         - self.data_structure.variables_dictionary['temperature_L1']),
                         #(1000.0 * self.variables_dictionary['zeu']),
                         self.data_structure.variables_dictionary['zeu'],
                         out=np.zeros_like(self.data_structure.variables_dictionary['zeu']),
                         where=self.data_structure.variables_dictionary['zeu']!=0.0)
        
        dTdz = np.where(dTdz < 0.0, 0.0, dTdz)
        dTdz = np.where(dTdz > 0.2, 0.2, dTdz)
        
        self.data_structure.variables_dictionary['temperature_L1'] = (
            self.data_structure.variables_dictionary['temperature_L1']
            + 4.0 * dTdz * (self.data_structure.variables_dictionary['sst']
                            - self.data_structure.variables_dictionary['temperature_L1'])
            )
        
        # Since the estimate of sigma with sst is always lower 
		# due to larger extension of warm watermasses in the surface
		# will add 1.0 here while passing to integrated T
        self.data_structure.parameters_dictionary['sigma_0'] += 1.0