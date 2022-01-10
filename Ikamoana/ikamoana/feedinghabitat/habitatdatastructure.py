"""
This module is used to declare the data structure that will store all
the data that the feedinghabitat module will use to calculate the
feeding habitat.
"""

from typing import List, Union
import xarray as xr
import numpy as np
import warnings

def equalCoords(data_array_1: xr.DataArray,
                data_array_2: xr.DataArray,
                verbose: bool = False) -> bool :
    """
    Compare two DataArray coordinates (`xarray.DataArray.coords`).

    Parameters
    ----------
    data_array_1 : xarray.DataArray
        First DataArray to compare to.
    data_array_2 : xarray.DataArray
        Second DataArray to compare to.
    verbose : bool, optional
        Show the number of differences between the coordinates,
        by default False.

    Returns
    -------
    bool
        True if the coordinates are the same, False otherwise.
    """

    if data_array_1.shape != data_array_2.shape :
        message_equalCoords = (
            ("Array_1(%s) and Array_2(%s) have different shapes :\n\t- Array_1 -> "%(
                data_array_1.name, data_array_2.name))
            + str(data_array_1.shape) + "\n\t- Array_2 -> " + str(data_array_2.shape)
        )
        warnings.warn(message_equalCoords)
        return False

    lat_comparison = data_array_1.lat.data == data_array_2.lat.data
    lat_condition = not (False in lat_comparison)

    lon_comparison = data_array_1.lon.data == data_array_2.lon.data
    lon_condition  = not (False in lon_comparison)

    time_comparison = data_array_1.time.data == data_array_2.time.data
    time_condition = not (False in time_comparison)

    if verbose :
        if not lat_condition :
            print("Latitude coordinates are different : %d errors for %d values."%(
                lat_comparison.size - np.sum(lat_comparison), lat_comparison.size))
        if not lon_condition :
            print("Longitude coordinates are different : %d errors for %d values."%(
                lon_comparison.size - np.sum(lon_comparison), lon_comparison.size))
        if not time_condition :
            print("Time coordinates are different : %d errors for %d values."%(
                time_comparison.size - np.sum(time_comparison), time_comparison.size))

    return lat_condition and lon_condition and time_condition

def groupArrayByCoords(variables_dictionary: dict) -> List[List[str]] :
    """
    Groupe variables with the same coordinates in a list using the 
    equalCoords function.

    Parameters
    ----------
    variables_dictionary : dict
        Dictionnary containing list of `xarray.DataArray`.

    Returns
    -------
    List[List[str]]
        List containing lists of variable names. Each list of names
        corresponds to a group of `xarray.DataArray` with the same
        coordinates.
    """
    
    list_compare = []
    for _, item in variables_dictionary.items() :
        list_tmp = []
        for name_i, item_i in variables_dictionary.items() :
            if equalCoords(item, item_i) :
                list_tmp.append(name_i)
        list_compare.append(tuple(list_tmp))
    return list(set(list_compare))

def compareDims(data_array_1: xr.DataArray, data_array_2: xr.DataArray,
                dim: str, head: int = None) -> None :
    """
    Compare two DataArray coordinates along the 'dim' specified by user.

    Parameters
    ----------
    data_array_1 : xarray.DataArray
        [description]
    data_array_2 : xarray.DataArray
        [description]
    dim : str
        The dimension in `xarray.DataArray.coords` to compare to.
    head : int, optional
        [description], by default None
    """
    dim1 = data_array_1.coords[dim].data
    dim2 = data_array_2.coords[dim].data

    dim_comparison = dim1 == dim2

    if head is None :
        head = dim_comparison.size
    ite = 0
    for i in range(dim_comparison.size) :
        if (not dim_comparison[i]) and (ite < head) :
            print(dim1[i],'\t|||\t',dim2[i])
            ite = ite + 1
    if ite < dim_comparison.size :
        print('[...]')

class HabitatDataStructure :

    def __init__(self, kargs: dict) -> None :
        """
        Initialize the data structure according to the XML file used in
        the FeedingHabitat Class.

        Parameters
        ----------
        kargs : dict
            Contains all the data needed to compute the FeedingHabitat.
            - root_directory
            - output_directory
            - layers_number
            - cohorts_number
            - partial_oxygen_time_axis
            - global_mask
            - coords
            - variables_dictionary
            - parameters_dictionary
            - species_dictionary

        Notes
        -----
        variables_dictionary contains :
            - 'oxygen_L1', 'oxygen_L2', 'oxygen_L3',
            - 'temperature_L1','temperature_L2', 'temperature_L3', 
            - 'forage_epi', 'forage_umeso', 'forage_mumeso', 
            - 'forage_lmeso', 'forage_mlmeso', 'forage_hmlmeso', 
            - 'sst', 'zeu', 'days_length'}
        
        parameters_dictionary contains :
            - 'eF_list', 'sigma_0', 'sigma_K', 'T_star_1', 'T_star_K',
            - 'bT', 'gamma', 'o_star', 'deltaT', 'space_reso'}

        species_dictionary contains :
            - 'sp_name', 'nb_life_stages', 'life_stage',
            - 'nb_cohort_life_stage', 'cohorts_mean_length',
            - 'cohorts_mean_weight', 'cohorts_sp_unit',
            - 'cohorts_starting_age', 'cohorts_final_age'}
        """

        self.root_directory           = kargs['root_directory']
        self.output_directory         = kargs['output_directory']
        self.layers_number            = kargs['layers_number']
        self.cohorts_number           = kargs['cohorts_number']
        self.partial_oxygen_time_axis = kargs['partial_oxygen_time_axis']
        self.global_mask              = kargs['global_mask']
        self.coords                   = kargs['coords']
        self.variables_dictionary     = kargs['variables_dictionary']
        self.parameters_dictionary    = kargs['parameters_dictionary']
        self.species_dictionary       = kargs['species_dictionary']

        pred = 0
        start_age = []
        end_age = []
        for unit in self.species_dictionary['cohorts_sp_unit'] :
            start_age.append(pred)
            pred = pred + unit
            end_age.append(pred)
        self.species_dictionary['cohorts_starting_age'] = np.array(start_age)
        self.species_dictionary['cohorts_final_age'] = np.array(end_age)

    def _summaryToString(self) -> str :
        """
        Summerize the data in this structure.
        
        See Also :
        ----------
            - __str__()
            - __repr__()
        """
        summary_str = (
            "# ------------------------------ #\n"
            "# Summary of this data structure #\n"
            "# ------------------------------ #\n\n"
            "Root directory is :\n"
            "\t{}\n"
            "Output directory is :\n"
            "\t{}\n"
            "Time resolution is (in days) : {}\n"
            "Space resolution is (in degrees) : {}\n"
            "\n\n# -------------------------------SPECIES----------------------------- #\n\n"
            "The short name of the species is {}.\n"
            "There is(are) {} life stage(s) considered in the model which is(are) : {}\n"
            ).format(
                self.root_directory,
                self.output_directory,
                self.parameters_dictionary["deltaT"],
                self.parameters_dictionary["space_reso"],
                self.species_dictionary["sp_name"],
                self.species_dictionary['nb_life_stages'],
                self.species_dictionary['life_stage'])

        for name, number in zip(self.species_dictionary['life_stage'],
                                self.species_dictionary['nb_cohort_life_stage']) :
            summary_str += ("\t- There is(are) {}\tcohort(s) in life stage {}.\n"
                            ).format(number, name)
        
        np.set_printoptions(suppress=True)

        summary_str += (
            "\nFinal age (in day) of each cohort is :\n{}\n"
            "\nMean length for each cohort is :\n{}\n"
            "\nMean weight for each cohort is :\n{}\n"
            "\n\n# -----------------------------PARAMETERS---------------------------- #\n\n"
            "The parameters used are the following :\n"
        ).format(self.species_dictionary['cohorts_final_age'],
                 self.species_dictionary['cohorts_mean_length'],
                 self.species_dictionary['cohorts_mean_weight'])
        
        for name, value in self.parameters_dictionary.items() :
            summary_str += "\t- {0:<12}:  {1}\n".format(name,str(value))
        
        summary_str += (
            "\nReminder : \n\t- Forage \t-> eF\n\t- Temperature\t-> sigma, T* "
            "and bT\n\t- Oxygen \t-> gamma and O*"
            "\n\n# ------------------------------FIELDS------------------------------- #\n\n"
            "WARNING : Fields must start on the same date !\n\n"
            "Fields are grouped by coordinates :\n"
        ).expandtabs(8)
        
        for group in groupArrayByCoords(self.variables_dictionary) :
            summary_str += (
                "\n#\t#\t#\t#\t#\n"
                "\nGroup : {}\n"
                "Their coordinates are :\n{}\n"
            ).format(group,self.variables_dictionary[group[0]].coords)
        
        summary_str += (
            "\n#\t#\t#\t#\t#\n\n"
            "Day Length is calculated using the main coordinates which are"
            "based on temperature (L1) field.\n"
        )

        if self.partial_oxygen_time_axis :
            summary_str +=(
                "\n#\t#\t#\t#\t#\n\n"
                "Oxygen is a climatologic field. It meen that only 1 year is"
                "modelized in the DataArray.\n"
            )

        summary_str += (
            "\n#\t#\t#\t#\t#\n\n"
            "TIPS : The user can use equalCoords() or compareDims() functions"
            "to compare Coordinates."
            "\n\n# ------------------------------------------------------------------- #"
        )

        return summary_str

    def findIndexByDatetime(self, date: Union[np.datetime64, List[np.datetime64]], verbose: bool = False) -> List[int] :
        """Find the time dimension index with the closest date."""

        date_list = self.coords['time']
        date = np.ravel(date)
        date_index_list = []

        for d in date:
            index = np.absolute(date_list-d).argmin()
            if verbose :
                print("The time coordinate closest to %s is index %d, which is %s"%(d, index, date_list[index].data))
            date_index_list.append(index)

        return date_index_list

    def findCoordIndexByValue(self, val: Union[float, List[float]], coord='lon', verbose: bool = False) -> List[int] :
        """Find the time dimension index with the closest date."""

        val_list = self.coords[coord]
        val = np.ravel(val)
        val_index_list = []

        for v in val:
            index = np.absolute(val_list-v).argmin()
            if verbose :
                print("The %s coordinate closest to %s is index %d, which is %s"%(coord, v, index, val_list[index].data))
            val_index_list.append(index)

        return val_index_list

    def findCohortByLength(self, length: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """
        Find the cohort number with the closest length.

        Parameters
        ----------
        length : Union[float, List[float]]
            One or more lengths of which you wish to recover the age.
        verbose : bool, optional
            Print the age of each length, by default False

        Returns
        -------
        List[int]
            List of age.
        """

        length_list = self.species_dictionary['cohorts_mean_length']
        length = np.ravel(length)
        cohort_number_list = []

        for l in length :
            index = np.absolute(length_list-l).argmin()
            if verbose :
                print("The cohort with the length closest to %f is cohort number %d whose length is %f"%(l, index, length_list[index]))
            cohort_number_list.append(index)

        return  cohort_number_list

    def findCohortByWeight(self, weight: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """
        Find the cohort number with the closest weight.
        
        Parameters
        ----------
        length : Union[float, List[float]]
            One or more weights of which you wish to recover the age.
        verbose : bool, optional
            Print the age of each weight, by default False

        Returns
        -------
        List[int]
            List of age.
        """
        
        weight_list = self.species_dictionary['cohorts_mean_weight']
        weight = np.ravel(weight)
        cohort_number_list = []

        for w in weight :
            index = np.absolute(weight_list-w).argmin()
            if verbose :
                print("The cohort with the weight closest to %f is cohort number %d whose weight is %f"%(w, index, weight_list[index]))
            cohort_number_list.append(index)

        return cohort_number_list

    def findLengthByCohort(self, cohort: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """
        Find cohort length according to cohort number

        Parameters
        ----------
        cohort : Union[float, List[float]]
            One or more cohort age of which you wish to recover the length.
        verbose : bool, optional
            Print the length of each age, by default False

        Returns
        -------
        List[int]
            List of length
        """
        
        cohort_length_list = self.species_dictionary['cohorts_mean_length']
        cohort = np.ravel(cohort)

        return  cohort_length_list[cohort]

    def findWeightByCohort(self, cohort: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """Find cohort weight according to cohort number.
        
        
        Parameters
        ----------
        cohort : Union[float, List[float]]
            One or more cohort age of which you wish to recover the weight.
        verbose : bool, optional
            Print the length of each age, by default False

        Returns
        -------
        List[int]
            List of weight
        """

        cohort_weight_list = self.species_dictionary['cohorts_mean_weight']
        cohort = np.ravel(cohort)

        return  cohort_weight_list[cohort]

    def __str__(self) -> str:
        return self._summaryToString()

    def __repr__(self) -> str:
        return self._summaryToString()
    
    def normalizeCoords(self):
        """
        Normalize the time axis. Remove the days if the resolution is in
        months,remove the seconds if the resolution is in days.
        """
        time_reso = self.parameters_dictionary["deltaT"]
        time_coords = self.coords['time'].data
        
        if time_reso == 30 : # SEAPODYM MONTHLY correspond to 30 days
            normalized_time_coords = np.array(time_coords,
                                              dtype='datetime64[M]')
        else :
            normalized_time_coords = np.array(time_coords,
                                              dtype='datetime64[D]')
        variable_update = {}
        for name, da in self.variables_dictionary.items() :
            da = da.to_dataset(name=name)
            da['normalized_time'] = ('time', normalized_time_coords)
            da = da.reset_index('time',drop=True)
            da = da.set_index({'time':'normalized_time'})
            da = da[name]
            variable_update[name] = da
        
        self.variables_dictionary.update(variable_update)
        self.coords = self.variables_dictionary['temperature_L1'].coords
