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
    """Compare two DataArray coordinates"""

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
    list_compare = []
    for _, item in variables_dictionary.items() :
        list_tmp = []
        for name_i, item_i in variables_dictionary.items() :
            if equalCoords(item, item_i) :
                list_tmp.append(name_i)
        list_compare.append(tuple(list_tmp))
    return list(set(list_compare))

def compareDims(data_array_1: xr.DataArray,
                data_array_2: xr.DataArray,
                dim: str,
                head: int = None):
    """Compare to DataArray coordinates along the 'dim' specified
    by user"""

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

        Notes
        -----
        variables_dictionary contains :
            {"forage_epi", "forage_meso", "forage_mumeso",
             "forage_lmeso", "forage_mlmeso", "forage_hmlmeso" ,
             "temperature_L1", "temperature_L2", "temperature_L3",
             "oxygen_L1", "oxygen_L2", "oxygen_L3",
             "days_length",
             "zeu", "sst" }
        
        parameters_dictionary contains :
            {"eF_list",
             "sigma_0", "sigma_K", "T_star_1", "T_star_K", "bT",
             "gamma", "o_star" }
        
        species_dictionary contains :
            {'sp_name', 'nb_life_stages', 'life_stage',
             'nb_cohort_life_stage',
             'cohorts_mean_length', 'cohorts_mean_weight'}
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

    ## NOTE : This is not needed since __str__ and __rep__ have been defined
    #
    # def summary(self) :

    #     print("# ------------------------------ #")
    #     print("# Summary of this data structure #")
    #     print("# ------------------------------ #", end='\n\n')

    #     print('Root directory is :\n\t'+self.root_directory, end='\n')
    #     print('Output directory is :\n\t'+self.output_directory, end='')

    #     print("\n\n\n# -------------------------------SPECIES----------------------------- #\n\n\n",end='')

    #     print('The short name of the species is %s.'%(self.species_dictionary["sp_name"]),end='\n')
    #     print('There is(are) %d\tlife stages considered in the model which are : '%(
    #         self.species_dictionary['nb_life_stages']),end='')
    #     print(self.species_dictionary['life_stage'],end='\n')
    #     for name, number in zip(self.species_dictionary['life_stage'], self.species_dictionary['nb_cohort_life_stage']) :
    #         print("\t- There is(are) %d\tcohort(s) in life stage %s."%(number, name))
    #     np.set_printoptions(suppress=True)
    #     print('\nFinal age (in day) of each cohort is :\n', self.species_dictionary['cohorts_final_age'])
    #     print('\nMean length for each cohort is :\n', self.species_dictionary['cohorts_mean_length'])
    #     print('\nMean weight for each cohort is :\n', self.species_dictionary['cohorts_mean_weight'], end='')

    #     print("\n\n\n# -----------------------------PARAMETERS---------------------------- #\n\n\n",end='')

    #     print('The parameters used are the following :')
    #     for name, value in self.parameters_dictionary.items() :
    #         print("\t- ",name,"   \t:", value)
    #     print("Reminder : \n\t- Forage \t-> eF\n\t- Temperature\t-> sigma, T* and bT\n\t- Oxygen \t-> gamma and O*",end='')

    #     print("\n\n# ------------------------------FIELDS------------------------------- #\n\n")

    #     print('WARNING : Fields must start on the same date !\n')

    #     print('Fields are grouped by coordinates :')

    #     for group in groupArrayByCoords(self.variables_dictionary) :
    #         print('\n#\t#\t#\t#\t#')
    #         print('Group :', group)
    #         print('Their coordinates are :\n',self.variables_dictionary[group[0]].coords)
        
    #     print('\n#\t#\t#\t#\t#\n')
    #     print('Day Length is calculated using the main coordinates which are based on temperature (L1) field.')

    #     if self.partial_oxygen_time_axis :
    #         print('\n#\t#\t#\t#\t#\n')
    #         print('Oxygen is a climatologic field. It meen that only 1 year is modelized in the DataArray.')

    #     print('\n#\t#\t#\t#\t#\n')
    #     print('TIPS : The user can use equalCoords() or compareDims() functions to compare Coordinates.',end='')
        
    #     print("\n\n\n# ------------------------------------------------------------------- #\n\n")
    
    def _summaryToString(self) -> str :
        summary_str = (
            "# ------------------------------ #\n"
            + "# Summary of this data structure #\n"
            + "# ------------------------------ #\n\n"
            + "Root directory is :\n\t"
            + self.root_directory + "\n"
            + 'Output directory is :\n\t'
            + self.output_directory
            + "\n\n# -------------------------------SPECIES----------------------------- #\n\n"
            + "The short name of the species is %s."%(self.species_dictionary["sp_name"]) + "\n"
            + "There is(are) %d\tlife stages considered in the model which are : "%(
                self.species_dictionary['nb_life_stages'])
            + self.species_dictionary['life_stage'].__str__() + "\n"
        )

        for name, number in zip(self.species_dictionary['life_stage'], self.species_dictionary['nb_cohort_life_stage']) :
            summary_str += "\t- There is(are) %d\tcohort(s) in life stage %s."%(number, name) + "\n"
        
        np.set_printoptions(suppress=True)

        summary_str += (
            "\nFinal age (in day) of each cohort is :\n"
            + self.species_dictionary['cohorts_final_age'].__str__() + "\n"
            + "\nMean length for each cohort is :\n"
            + self.species_dictionary['cohorts_mean_length'].__str__() + "\n"
            + '\nMean weight for each cohort is :\n'
            + self.species_dictionary['cohorts_mean_weight'].__str__() + "\n"
            + "\n\n# -----------------------------PARAMETERS---------------------------- #\n\n"
            + "The parameters used are the following :\n"
        )
        
        for name, value in self.parameters_dictionary.items() :
            summary_str += "\t- " + name.__str__() + "   \t:" + value.__str__() + "\n"
        
        summary_str += (
            "\nReminder : \n\t- Forage \t-> eF\n\t- Temperature\t-> sigma, T* and bT\n\t- Oxygen \t-> gamma and O*"
            + "\n\n# ------------------------------FIELDS------------------------------- #\n\n"
            + 'WARNING : Fields must start on the same date !\n\n'
            + 'Fields are grouped by coordinates :\n'
        )
        
        for group in groupArrayByCoords(self.variables_dictionary) :
            summary_str += (
                '\n#\t#\t#\t#\t#\n'
                + '\nGroup :' + group.__str__() + "\n"
                + 'Their coordinates are :\n'
                + self.variables_dictionary[group[0]].coords.__str__() + "\n"
            )
        
        summary_str += (
            '\n#\t#\t#\t#\t#\n\n'
            + 'Day Length is calculated using the main coordinates which are based on temperature (L1) field.\n'
        )

        if self.partial_oxygen_time_axis :
            summary_str +=(
                '\n#\t#\t#\t#\t#\n\n'
                + 'Oxygen is a climatologic field. It meen that only 1 year is modelized in the DataArray.\n'
            )
        
        summary_str += (
            '\n#\t#\t#\t#\t#\n\n'
            + 'TIPS : The user can use equalCoords() or compareDims() functions to compare Coordinates.'
            + "\n\n# ------------------------------------------------------------------- #"
        )

        return summary_str

    def findCohortByLength(self, length: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """Find the cohort number with the closest length."""
        
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
        """Find the cohort number with the closest weight."""
        
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
        """Find cohort length according to cohort number."""
        
        cohort_length_list = self.species_dictionary['cohorts_mean_length']
        cohort = np.ravel(cohort)

        return  cohort_length_list[cohort]
    
    def findWeightByCohort(self, cohort: Union[float, List[float]], verbose: bool = False) -> List[int] :
        """Find cohort weight according to cohort number."""

        cohort_weight_list = self.species_dictionary['cohorts_mean_weight']
        cohort = np.ravel(cohort)

        return  cohort_weight_list[cohort]

    def __str__(self) -> str:
        return self._summaryToString()
    
    def __repr__(self) -> str:
        return self._summaryToString()