"""
This module is used to declare the data structure that will store all
the data that the feedinghabitat module will use to calculate the
feeding habitat.
"""

from typing import List


class HabitatDataStructure :

    def __init__(self, kargs: dict) -> None :
        """
        Initialize the data structure according to the XML file used in
        the FeedingHabitat Class.

        Notes
        -----
        variables_dictionary contains :
            { "forage_epi", "forage_meso", "forage_mumeso",
             "forage_lmeso", "forage_mlmeso", "forage_hmlmeso" ,
             "temperature_L1", "temperature_L2", "temperature_L3",
             "oxygen_L1", "oxygen_L2", "oxygen_L3",
             "days_length",
             "cohorts_mean_length", "cohorts_mean_weight" ,
             "Zeu" }
        
        parameters_dictionary contains :
            { "eF_list",
             "sigma_0", "sigma_K", "T_star_1", "T_star_K", "bT",
             "gamma", "o_star" }
        """
        self.root_directory           = kargs['root_directory']
        self.output_directory         = kargs['output_directory']
        self.layers_number            = kargs['layers_number']
        self.cohorts_number           = kargs['cohorts_number']
        self.cohorts_to_compute       = kargs['cohorts_to_compute']
        self.partial_oxygen_time_axis = kargs['partial_oxygen_time_axis']
        self.global_mask              = kargs['global_mask']
        self.coords                   = kargs['coords']
        self.variables_dictionary     = kargs['variables_dictionary']
        self.parameters_dictionary    = kargs['parameters_dictionary']
        self.species_dictionary       = kargs['species_dictionary']


    def setCohorts_to_compute(self, cohorts_to_compute: List[int]) -> None :
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

        """
        self.cohorts_to_compute = cohorts_to_compute

    def summary(self) :

        separator = "\n\n# ------------------------------------------------------------------- #\n\n"

        print("# ------------------------------ #")
        print("# Summary of this data structure #")
        print("# ------------------------------ #", end='\n\n')

        print('Root directory is :\t'+self.root_directory, end='\n')
        print('Output directory is :\t'+self.output_directory, end=separator)

        print('The short name of the species is %s.'%(self.species_dictionary["sp_name"]),end='\n')
        print('There is(are) %d\tlife stages considered in the model which are : '%(
            self.species_dictionary['nb_life_stages']),end='')
        print(self.species_dictionary['life_stage'],end='\n')
        for name, number in zip(self.species_dictionary['life_stage'], self.species_dictionary['nb_cohort_life_stage']) :
            print("\t- There is(are) %d\tcohort(s) in life stage %s."%(number, name))
# TODO : Add length and weight visualization
        print("\n# ------------------------------------------------------------------- #\n\n",end='')

        print('The parameters used are the following :')
        for name, value in self.parameters_dictionary.items() :
            print("\t- ",name,"   \t:", value)
        print("Reminder : \n\t- Forage \t-> eF\n\t- Temperature\t-> sigma, T* and bT\n\t- Oxygen \t-> gamma and O*",end=separator)

        print('',end='')
        print('',end='')
        print('',end='')
        print('',end='')
        print('',end='')
        print('',end='')