# -*- coding: utf-8 -*-

"""
Summary
-------
This module is implementing the FeedingHabitat class which can simulate the
feeding habitat the same way as the SEAPODYM model (2020-08).
This class start by initalizing the HabitatDataStructure class using the
FeedingHabitatConfigReader module. Then, it performes computation of the
Feeding Habitat, for each cohort specified in argument, returned as a DataSet.

"""

from types import LambdaType
from typing import List, Union, Tuple
from . import feedinghabitatconfigreader as fhcr
from . import habitatdatastructure as hds
from math import sin, cos, pi, sqrt
import xarray as xr
import numpy as np

def posClosestCoord(coords: Union[list,np.ndarray,xr.DataArray], value: Union[str,np.datetime64,int,float]) -> int :
    """
    Return the position of the closest value in a specific coordinate.

    Value :
    -------
        Time :
            String -> 'YYYY-MM-DD'
            (or)
            Datetime64
    """
    coords = np.array(coords)
    if isinstance(value, str):
        return np.argmin(np.abs(np.Datetime64(value, 'ns') - coords))
    else :
        return np.argmin(np.abs(value - coords))

def closestCoord(coords: Union[list,np.ndarray,xr.DataArray], value: Union[str,np.datetime64,int,float]) -> Union[np.datetime64, float] :
    """
    Return the closest value in a specific coordinate.

    Value :
    -------
        Time :
            String -> 'YYYY-MM-DD'
            (or)
            Datetime64
    """
    return coords[posClosestCoord(coords, value)].data

def coordsAccess(coords: xr.Coordinate) -> Tuple[LambdaType,LambdaType,LambdaType]:
    """Return accessor to closest value in time, lat and lon coordinates."""
    return (lambda time : posClosestCoord(coords['time'], time),
            lambda lat : posClosestCoord(coords['lat'], lat),
            lambda lon : posClosestCoord(coords['lon'], lon))

class FeedingHabitat :

    def __init__(self,
                 xml_filepath: str,
                 float_32: bool = True) :
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
        FeedingHabitat
        """

        self.data_structure = hds.HabitatDataStructure(
            fhcr.loadFromXml(xml_filepath,float_32)
        )

    def __str__(self) -> str:
        return self.data_structure.__str__()

    def __repr__(self) -> str:
        return self.data_structure.__repr__()

###############################################################################

    def _scaling(self, data) :
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

    def _selSubDataArray(self,
                         data_array: xr.DataArray,
                         time_start: int = None, time_end: int = None,
                         lat_min: int = None, lat_max: int = None,
                         lon_min: int = None, lon_max: int = None) -> xr.DataArray :
        """Select a part of the DataArray passed in argument according to time,
        latitude and longitude"""

        return data_array.sel(
            time=data_array.time.data[
                time_start:time_end if time_end is None else time_end+1 ],
            lat=data_array.lat.data[
                lat_min:lat_max if lat_max is None else lat_max+1],
            lon=data_array.lon.data[
                lon_min:lon_max if lon_max is None else lon_max+1])

    def _selSubMask(self,
                    mask: str,
                    lat_min: int = None, lat_max: int = None,
                    lon_min: int = None, lon_max: int = None) -> np.ndarray :
        """Select a part of the Mask (Numpy array) passed in argument
        according to time, latitude and longitude"""

        tmp = self.data_structure.global_mask[mask]

        return tmp[:,
                    lat_min:lat_max if lat_max is None else lat_max+1,
                    lon_min:lon_max if lon_max is None else lon_max+1]

    def _sigmaStar(self, sigma_0, sigma_K) :
        """Return sigmaStar (the termal tolerance intervals, i.e. standard
        deviation) for each cohorts."""

        cohorts_mean_weight = self.data_structure.species_dictionary[
            'cohorts_mean_weight']
        max_weight = np.max(cohorts_mean_weight)

        return sigma_0 + ((sigma_K - sigma_0)
                          * (cohorts_mean_weight / max_weight))

    def _tStar(self, T_star_1, T_star_K, bT) :
        """Return T_star (optimal temperature, i.e. mean) for each cohorts"""

        cohorts_mean_length = self.data_structure.species_dictionary[
            'cohorts_mean_length']
        max_length = np.max(cohorts_mean_length)

        return T_star_1 - ((T_star_1 - T_star_K)
                           * ((cohorts_mean_length / max_length)**bT))

    def _temperature(self,
                     cohort: int,
                     time_start: int = None, time_end: int = None,
                     lat_min: int = None, lat_max: int = None,
                     lon_min: int = None, lon_max: int = None) -> np.ndarray :

        sigma_star = self._sigmaStar(
            self.data_structure.parameters_dictionary['sigma_0'],
            self.data_structure.parameters_dictionary['sigma_K'])
        T_star = self._tStar(
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

    def computeFeedingHabitat(self,
                              cohorts: Union[int, List[int]],
                              time_start: int = None, time_end: int = None,
                              lat_min: int = None, lat_max: int = None,
                              lon_min: int = None, lon_max: int = None,
                              verbose: bool = False) -> xr.Dataset :
        """
        The main function of the FeedingHabitat class. It will compute the
        feeding habitat of each cohort specify in the `cohorts` argument.

        Parameters
        ----------
        cohorts : Union[int, List[int]]
            Specify the cohorts we want to compute the feeding habitat.
        time_start : int = None, optional
            Starting position in the coords['time'] coordinate.
            The default is None, converted into 0.
        time_end : int = None, optional
            Ending position in the coords['time'] coordinate.
            The default is None, converted into last position in `coords['time']`.
        lat_min : int = None, optional
            Starting position in the coords['lat'] coordinate.
            The default is None, converted into 0.
        lat_max : int = None, optional
            Ending position in the coords['lat'] coordinate.
            The default is None, converted into last position in `coords['lat']`.
        lon_min : int = None, optional
            Starting position in the coords['lon'] coordinate.
            The default is None, converted into 0.
        lon_max : int = None, optional
            Ending position in the coords['lon'] coordinate.
            The default is None, converted into last position in `coords['lon']`.
        verbose : boolean, optional
            If True, print some informations about the running state.
            The default is False.

        Returns
        -------
        xarray.DataSet
            A DataSet containing Feeding Habitat of each cohort specified
            in the `cohorts` argument.

        """

        if isinstance(cohorts, (int, np.integer)) :
            cohorts = [cohorts]

        def controlArguments(data_structure: hds.HabitatDataStructure,
                             cohorts, time_start, time_end, lat_min,
                             lat_max, lon_min, lon_max) :

            for elmt in cohorts :
                if elmt < 0 or elmt >= data_structure.cohorts_number :
                    raise ValueError("cohort out of bounds. Min is 0 and Max is %d"%(
                        self.data_structure.cohorts_number-1))

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

        controlArguments(
            self.data_structure, cohorts, time_start, time_end, lat_min,
            lat_max, lon_min, lon_max)

        fh_oxygen = self._oxygen(
            time_start, time_end, lat_min, lat_max, lon_min, lon_max)

        fh_forage = self._forage(
            time_start, time_end, lat_min, lat_max, lon_min, lon_max)

        result = {}
        mask_L1 = self._selSubMask(
            'mask_L1', lat_min, lat_max, lon_min, lon_max)

        for cohort_number in cohorts :
            if verbose :
                print("Computing Feeding Habitat for cohort %d."%(cohort_number))

            fh_temperature = self._temperature(
                cohort_number, time_start, time_end, lat_min, lat_max,
                lon_min, lon_max)

            name = 'Feeding_Habitat_Cohort_%d'%(cohort_number)

            result_np_array = np.where(
                mask_L1,
                self._scaling(
                    np.sum(
                        fh_forage * (fh_temperature * fh_oxygen + 1e-4),
                        axis=0)),
                np.NaN)

            result_xr_data_array = xr.DataArray(
                result_np_array,
                name=name,
                coords=dict(
                    lon=self.data_structure.coords['lon'].data[
                        lon_min:lon_max if lon_max is None else lon_max+1],
                    lat=self.data_structure.coords['lat'].data[
                        lat_min:lat_max if lat_max is None else lat_max+1],
                    time=self.data_structure.coords['time'].data[
                        time_start:time_end if time_end is None else time_end+1]),
                dims=["time", "lat", "lon"],
                attrs={
                    'Cohort number':cohort_number,
                    'Age start (days)' : self.data_structure.species_dictionary[
                        'cohorts_starting_age'][cohort_number],
                    'Age end (days)' : self.data_structure.species_dictionary[
                        'cohorts_final_age'][cohort_number],
                    'Length (cm)' : self.data_structure.species_dictionary[
                        'cohorts_mean_length'][cohort_number],
                    'Weight (kg)' : self.data_structure.species_dictionary[
                        'cohorts_mean_weight'][cohort_number]}
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

    def computeEvolvingFeedingHabitat(
            self,
            cohort_start: int = None, cohort_end: int = None,
            time_start: int = None, time_end: int = None,
            lat_min: int = None, lat_max: int = None,
            lon_min: int = None, lon_max: int = None,
            verbose: bool = False) -> xr.DataArray :
        """
        The feeding habitat evolves over time. Each time step corresponds to
        a cohort at a specific age.

        Parameters
        ----------
        cohort_start : int, optional
            Specify the first cohorts to start with.
            Default is None (correspond to 0).
        cohort_end : int, optional
            Specify the latest cohort. If it is the oldest, the `time_end`
            argument will specify when to stop the calculation.
            The default value is None (corresponds to `cohort_number - 1`).
        time_start : int = None, optional
            Starting position in the coords['time'] coordinate.
            The default is None, converted into 0.
        time_end : int = None, optional
            Ending position in the coords['time'] coordinate.
            The default is None, converted into last position in `coords['time']`.
        lat_min : int = None, optional
            Starting position in the coords['lat'] coordinate.
            The default is None, converted into 0.
        lat_max : int = None, optional
            Ending position in the coords['lat'] coordinate.
            The default is None, converted into last position in `coords['lat']`.
        lon_min : int = None, optional
            Starting position in the coords['lon'] coordinate.
            The default is None, converted into 0.
        lon_max : int = None, optional
            Ending position in the coords['lon'] coordinate.
            The default is None, converted into last position in `coords['lon']`.
        verbose : boolean, optional
            If True, print some informations about the running state.
            The default is False.

        Returns
        -------
        xarray.DataArray
            A DataArray containing Feeding Habitat of a cohort that will evolve
            over time.
        """

        def controlArguments(
            data_structure: hds.HabitatDataStructure,
            cohort_start: int = None, cohort_end: int = None,
            time_start: int = None, time_end: int = None) :

            cohorts_number = data_structure.cohorts_number

            if cohort_start is None :
                cohort_start = 0
            if cohort_end is None :
                if time_end is not None:
                    cohort_end = cohort_start + time_end - time_start
                else:
                    cohort_end = cohorts_number - 1
            if (cohort_start < 0) or (cohort_start >= cohorts_number) :
                raise ValueError("cohort_start out of bounds. Min is %d and Max is %d"%(
                    0, cohorts_number - 1))
            if (cohort_end < 0) or (cohort_end >= cohorts_number) :
                raise ValueError("cohort_end out of bounds. Min is %d and Max is %d"%(
                    0, cohorts_number - 1))
            if cohort_start > cohort_end :
                raise ValueError("cohort_start must be <= to cohort_end.")
            cohort_array = np.arange(cohort_start, cohort_end+1)

            coords = data_structure.coords

            if time_start is None :
                time_start = 0
            if time_end is None :
                time_end = coords['time'].size - 1
            if ((time_start < 0) or (time_start >= coords['time'].data.size)) :
                raise ValueError("time_start out of bounds. Min is %d and Max is %d"%(
                    0, coords['time'].data.size - 1))
            if ((time_end < 0) or (time_end >= coords['time'].data.size)) :
                raise ValueError("time_end out of bounds. Min is %d and Max is %d"%(
                    0, coords['time'].data.size - 1))
            if time_start > time_end :
                raise ValueError("time_start must be <= to time_end.")

            time_array = np.arange(time_start, time_end+1)

            if cohort_array[-1] < cohorts_number-1 :
                max_size = min(cohort_array.size, time_array.size)
                cohort_array = cohort_array[:max_size]
                time_array = time_array[:max_size]


            return cohort_array, time_array

        cohort_array, time_array = controlArguments(
            self.data_structure,
            cohort_start, cohort_end, time_start, time_end)

        cohort_axis = []
        final_array = []
        for i in range(cohort_array.size) :
            # Oldest cohort with many time steps
            if (cohort_array[i:].size == 1) and (time_array[i:].size > 1) :
                cohort_axis.extend([cohort_array[i]] * time_array[i:].size)
                final_array.append(self.computeFeedingHabitat(
                    cohorts=cohort_array[i],
                    time_start=time_array[i], time_end=time_array[-1],
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max).to_array().data[0,:,:,:])

            # Others
            else :
                cohort_axis.append(cohort_array[i])
                final_array.append(self.computeFeedingHabitat(
                    cohorts=cohort_array[i],
                    time_start=time_array[i], time_end=time_array[i],
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max).to_array().data[0,:,:,:])

        return xr.DataArray(
            data=np.concatenate(final_array),
            name='Feeding_Habitat_Cohort_%d_to_%d'%(cohort_array[0], cohort_array[-1]),
            dims=('time','lat','lon'),
            coords=dict(
                time=self.data_structure.coords['time'].data[time_array[0]:time_array[-1]+1],
                lat=self.data_structure.coords['lat'].data[lat_min:lat_max+1],
                lon=self.data_structure.coords['lon'].data[lon_min:lon_max+1],
                cohorts=("time", cohort_axis)),
            attrs=dict(
                description="Feeding habitat of a species that evolves over time.",
                cohort_start=cohort_axis[0],cohort_end=cohort_axis[-1],
                time_start=time_array[0], time_end=time_array[-1],
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                **self.data_structure.parameters_dictionary) # Simple way to merge dictionary
            )

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
