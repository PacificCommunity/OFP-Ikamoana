# -*- coding: utf-8 -*-
"""
@Author : Jules Lehodey
@Date   : 02/08/2021

Summary
-------
This module is implementing the FeedingHabitat class which can simulate the
feeding habitat the same way as the SEAPODYM model (2020-08). This class start
with a initialization using the readFiles module, then performe computation
for each cohort.

"""

from math import sin, cos, pi, sqrt
import xarray as xr
import numpy as np
import readFiles


class FeedingHabitat :
    
###############################################################################    
# -----------------------------  INITIALIZATION  -----------------------------#
###############################################################################

# This is an example of the data you can find in a class instance
    def __init__(self, xml_filepath, partial_cohorts_computation=None, float_32=True) :
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
        
        Notes
        -----
        variables_dictionary contains :
            { "forage_epi", "forage_meso", "forage_mmeso",
             "forage_bathy", "forage_mbathy", "forage_hmbathy" ,
             "temperature_L1", "temperature_L2", "temperature_L3",
             "oxygen_L1", "oxygen_L2", "oxygen_L3",
             "days_length",
             "cohorts_mean_length", "cohorts_mean_weight" ,
             "Zeu" }
        
        parameters_dictionary contains :
            { "eF_list",
             "sigma_0", "sigma_K", "T_star_1", "T_star_K", "bT",
             "gamma", "o_star" }
        
        Returns
        -------
        None.

        """
        
        (self.root_directory,
         self.output_directory, 
         self.layers_number,
         self.cohorts_number,
         self.cohorts_to_compute,
         self.partial_oxygen_time_axis,
         self.global_mask,
         self.coords,
         self.variables_dictionary,
         self.parameters_dictionary) = readFiles.loadFromXml(xml_filepath,
                                                             partial_cohorts_computation,
                                                             float_32)

    def setCohorts_to_compute(self, cohorts_to_compute) :
        """
        You can change the list of cohorts you want to compute the habitat at
        any moment by using this setter.

        Parameters
        ----------
        cohorts_to_compute : list of int
            If you want to perform a partial feeding habitat computation, you
            can  specify a group of cohort using a number corresponding to the
            position in the cohort list.
            Warning : The first cohort is number 0.
            For example, if you want to compute the feeding habitat of the
            second and third cohort : partial_cohorts_computation = [1,2].

        Returns
        -------
        None.

        """
        self.cohorts_to_compute = cohorts_to_compute

###############################################################################
# ------------------  FUNCTIONS USED TO COMPUTE HABITAT  -------------------- #
###############################################################################
    
    def __sigmaStar__(self, sigma_0, sigma_K) :
        # Return sigmaStar (the termal tolerance intervals, i.e. standard
        # deviation) for each cohorts.
        
        cohorts_mean_weight = self.variables_dictionary['cohorts_mean_weight']
        max_weight = np.max(cohorts_mean_weight)

        return sigma_0 + ( (sigma_K - sigma_0) * (cohorts_mean_weight / max_weight) )
    
    def __tStar__(self, T_star_1, T_star_K, bT) :
        # Return T_star (optimal temperature, i.e. mean) for each cohorts
        
        cohorts_mean_length = self.variables_dictionary['cohorts_mean_length']
        max_length = np.max(cohorts_mean_length)
        
        return T_star_1 - ( (T_star_1 - T_star_K) * ((cohorts_mean_length / max_length)**bT) )

    def __temperatureNthCohort__(self, sigma_0, sigma_K, T_star_1, T_star_K, bT, Nth_cohort) :
        # Return accessibility for the Nth cohort to each layer according to temperature
        
        layers = ["temperature_L1", "temperature_L2","temperature_L3"]

        sigma_star = self.__sigmaStar__(sigma_0, sigma_K)
        T_star = self.__tStar__(T_star_1, T_star_K, bT)
        sigma_star_a = sigma_star[Nth_cohort]
        T_star_a = T_star[Nth_cohort]

        layer_buffer = []
        for layer, mask in zip(layers, self.global_mask.keys()) :
            
            variable = self.variables_dictionary[layer]
            layer_buffer.append(
                np.exp(
                      (- np.power((variable - T_star_a), 2) )
                      / (2.0 * (sigma_star_a**2)),
                      out=np.zeros_like(variable),
                      where=self.global_mask[mask])
                )
                
        return np.array(layer_buffer) 
    
    def __temperatureNthCohortAtDate__(self, position, sigma_0, sigma_K, T_star_1,
                                       T_star_K, bT, Nth_cohort) :
        # Return accessibility for the Nth cohort at date to each layer
        # according to temperature
        
        layers = ["temperature_L1", "temperature_L2","temperature_L3"]

        sigma_star = self.__sigmaStar__(sigma_0, sigma_K)
        T_star = self.__tStar__(T_star_1, T_star_K, bT)
        sigma_star_a = sigma_star[Nth_cohort]
        T_star_a = T_star[Nth_cohort]

        layer_buffer = []
        for layer, mask in zip(layers, self.global_mask.keys()) :
            
            variable = self.variables_dictionary[layer][position]
            layer_buffer.append(
                np.exp(
                      (- np.power((variable - T_star_a), 2) )
                      / (2.0 * (sigma_star_a**2)),
                      out=np.zeros_like(variable),
                      where=self.global_mask[mask][0])
                )
                
        return np.array(layer_buffer)
    
    def __oxygen__(self, gamma, o_star) :
        # Return accessibility for the Nth cohort to each layer according to oxygen
        
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

    def __oxygenAtDate__(self, position, gamma, o_star) :
        # Return accessibility for the Nth cohort at a date to each layer
        # according to oxygen
        
        layers = ["oxygen_L1", "oxygen_L2", "oxygen_L3"]

        if self.partial_oxygen_time_axis :
            position %= 12

        layer_buffer = []
        for layer, mask in zip(layers, self.global_mask.keys()) :
            variable = self.variables_dictionary[layer][position] 
            layer_buffer.append(
                np.where(
                    self.global_mask[mask][0],
                    1.0 / (1.0 + (np.power(gamma,(variable - o_star)))),
                    0.0)
                )
        
        return np.array(layer_buffer)

    def __forage__(self, forage_preference_coefficients) :
        # Return forage quantity and weight accessibility in each layer 
        
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
    
    def __forageAtDate__(self, date, position, forage_preference_coefficients) :
        
        # Return forage quantity and weight accessibility in each layer 
        
        # Initialization
        days_length = self.variables_dictionary['days_length'].sel(time=[date])
        night_length = np.ones_like(days_length) - days_length

        #L1 ###################################################################
        epi     = self.variables_dictionary["forage_epi"][position]     * forage_preference_coefficients[0]
        mmeso   = self.variables_dictionary["forage_mmeso"][position]   * forage_preference_coefficients[2]
        hmbathy = self.variables_dictionary["forage_hmbathy"][position] * forage_preference_coefficients[5]
        
        day_l1  = days_length * epi
        night_l1  = night_length * (epi + mmeso + hmbathy)
        
        tmp_L1 = np.add(day_l1, night_l1, out=np.zeros_like(days_length), where=self.global_mask['mask_L1'][0])
        del epi, day_l1, night_l1

        #L2 ###################################################################
        meso    = self.variables_dictionary["forage_meso"][position]    * forage_preference_coefficients[1]
        mbathy  = self.variables_dictionary["forage_mbathy"][position]  * forage_preference_coefficients[4]
        
        day_l2  = days_length * (meso + mmeso)
        night_l2  = night_length * (meso + mbathy)
        
        tmp_L2 = np.add(day_l2, night_l2, out=np.zeros_like(days_length), where=self.global_mask['mask_L2'][0])
        del meso, mmeso, day_l2, night_l2

        #L3 ###################################################################
        bathy   = self.variables_dictionary["forage_bathy"][position]   * forage_preference_coefficients[3]
        
        day_l3  = days_length * (bathy + mbathy + hmbathy)
        night_l3  = night_length * (bathy)
        
        tmp_L3 = np.add(day_l3, night_l3, out=np.zeros_like(days_length), where=self.global_mask['mask_L3'][0])
        del bathy, mbathy, hmbathy, day_l3, night_l3, night_length

        # Result ##############################################################
        result = np.array([tmp_L1, tmp_L2, tmp_L3])
        del tmp_L1, tmp_L2, tmp_L3
    
        return result

###############################################################################
# ---------------------------------  MAIN  --------------------------------- #
###############################################################################

    def computeFeedingHabitat(self, filepath=None, verbose=True) :
        """
        The main function of the FeedingHabitat class. It will compute the
        feeding habitat of each cohort specify in the cohorts_to_compute
        attribut of the instance.

        Parameters
        ----------
        filepath : string, optional
            The filepath where to save all the netCDF we will produce during
            this function execution. If None, the output filepath in the XML 
            configuration file will be used.
            The default is None.
        verbose : boolean, optional
            If True, print some informations about the running state.
            The default is True.

        Returns
        -------
        None.

        """
        path_save = (self.root_directory  + self.output_directory) if filepath is None else filepath
        if verbose : print('Files will be saved at : %s' % path_save)
        
        # FORAGE ##############################################################
        if verbose : print('Computing  forage')
        ha_forage = self.__forage__(self.parameters_dictionary['eF_list'])
        
        # OXYGEN ##############################################################
        if verbose : print('Computing oxygen')
        ha_oxygen = self.__oxygen__(self.parameters_dictionary['gamma'],
                                    self.parameters_dictionary['o_star'])
        # When oxygen is from climat model (only 1 year)
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
        
        # For each cohort compute habitat then save in netCDF file
        for cohort in cohorts_range:
            
            # TEMPERATURE #####################################################
            if verbose : print('Cohort %d : computing temperature' % cohort)
            ha_temperature = self.__temperatureNthCohort__(
                self.parameters_dictionary['sigma_0'],
                self.parameters_dictionary['sigma_K'], 
                self.parameters_dictionary['T_star_1'],
                self.parameters_dictionary['T_star_K'], 
                self.parameters_dictionary['bT'],
                Nth_cohort=cohort)
            
            # SAVE ############################################################
            if verbose : print('Cohort %d : saving Ha in NetCDF file' % cohort)
            
            # WARNING :  1e-4 is added on layer access to copy SEAPODYM behavior
            # TODO : 1e-4 is added on layer access to copy SEAPODYM behavior
            ha_result = np.where(
                self.global_mask['mask_L1'],
                self.__scaling__(
                    np.sum(
                        ha_forage * (ha_temperature * ha_oxygen + 1e-4),
                        axis=0)),
                np.NaN)
            
            da_to_save = xr.DataArray(
                data=ha_result,
                dims=["time", "lat", "lon"],
                coords=dict(
                    lon=self.coords['lon'],
                    lat=self.coords['lat'],
                    time=self.coords['time']),
                attrs=dict(description=("Ha_cohort_%d" %cohort)))
            da_to_save.to_netcdf(path_save +("ha_cohort_%d.nc" % cohort))
            
    def computeFeedingHabitatForSpecificAgeAndDate(self, age, date, verbose=False) :
        """
        Compute feeding habitat of a cohort at a specific date.

        Parameters
        ----------
        age : int
            Position of the cohort in the length and weight tabs. The first one
            (larvae) is at position 0.
        date : str, int, numpy.datetime64
            3 formats:
                - string as "2010-07-17T12:00:00.000000000"
                - numpy.datetime64 (see also. FeedingHabitat.coords['time'])
                - integer, starting with 0.
        verbose : bool, optional
            Print some informations about the selected cohort.
            The default is False.
            
        Notes
        -----
        If oxygen is climatologic (1 year only), it must begin in January.

        Raises
        ------
        TypeError
            If date type is not str, int or numpy.datetime64 an error will be raised.

        Returns
        -------
        ha_result_da : TYPE
            The feeding habitat of the Nth cohort (according to age argument)
            at a specific date (according to the date argument).

        """
        
        if isinstance(date, str) :
            date = np.datetime64(date)
            position = np.where(self.coords['time'].data == date)
        elif isinstance(date, int) :
            position = date
            date = self.coords['time'].data[date]
        elif isinstance(date, np.datetime64) :
            position = np.where(self.coords['time'].data == date)
        else :
            raise TypeError("Date must be : String, Integer or Numpy.datetime64")
        
        if verbose :
            print("At age %d, length is %f and weight is %f."%(age,
                self.variables_dictionary['cohorts_mean_length'][age],
                self.variables_dictionary['cohorts_mean_weight'][age]))
        
        # FORAGE ##############################################################
        ha_forage = self.__forageAtDate__(date, position,
                                          self.parameters_dictionary['eF_list'])
        
        # OXYGEN ##############################################################
        ha_oxygen = self.__oxygenAtDate__(position,
                                          self.parameters_dictionary['gamma'],
                                          self.parameters_dictionary['o_star'])
        
        # TEMPERATURE #####################################################
        ha_temperature = self.__temperatureNthCohortAtDate__(
            position,
            self.parameters_dictionary['sigma_0'],
            self.parameters_dictionary['sigma_K'], 
            self.parameters_dictionary['T_star_1'],
            self.parameters_dictionary['T_star_K'], 
            self.parameters_dictionary['bT'],
            Nth_cohort=age)
        
        # SAVE ############################################################
       
        # WARNING :  1e-4 is added on layer access to copy SEAPODYM behavior
        # TODO : 1e-4 is added on layer access to copy SEAPODYM behavior
        ha_result = np.where(
            self.global_mask['mask_L1'],
            self.__scaling__(
                np.sum(
                    ha_forage * (ha_temperature * ha_oxygen + 1e-4),
                    axis=0)),
            np.NaN)
        
        ha_result_da = xr.DataArray(
                data=ha_result[0],
                dims=["lat", "lon"],
                coords=dict(lon=self.coords['lon'],
                            lat=self.coords['lat']),
                attrs=dict(description=(("Ha_cohort_%d_at_date_"%age)+str(date)),
                           date=str(date)))

        return ha_result_da 
            
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
    
        return (y0+(B+np.sqrt(B*B-4.0*A*C))/(2*A));

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
        
        print("Warning : This function (correctEpiTempWithZeu) was only tested"+
              " for Skipjack.\n It will also add +1 to sigma_min. Cf. function"+
              " documentation for more details.")
        
        dTdz = np.divide(2.0 * (self.variables_dictionary['sst']
                              - self.variables_dictionary['temperature_L1']),
                         #(1000.0 * self.variables_dictionary['zeu']),
                         self.variables_dictionary['zeu'],
                         out=np.zeros_like(self.variables_dictionary['zeu']),
                         where=self.variables_dictionary['zeu']!=0.0)
        
        dTdz = np.where(dTdz < 0.0, 0.0, dTdz)
        dTdz = np.where(dTdz > 0.2, 0.2, dTdz)
        
        self.variables_dictionary['temperature_L1'] = (
            self.variables_dictionary['temperature_L1'] + 
            4.0 * dTdz * (self.variables_dictionary['sst']
                          - self.variables_dictionary['temperature_L1'])
            )
        
        # Since the estimate of sigma with sst is always lower 
		# due to larger extension of warm watermasses in the surface
		# will add 1.0 here while passing to integrated T
        self.parameters_dictionary['sigma_0'] += 1.0
        























