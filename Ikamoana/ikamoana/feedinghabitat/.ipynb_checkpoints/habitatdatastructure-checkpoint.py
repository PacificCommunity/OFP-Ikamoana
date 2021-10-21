"""
This module is used to declare the data structure that will store all
the data that the feedinghabitat module will use to calculate the
feeding habitat.
"""


__author__ = 'Jules Lehodey'
__last_modification__ = '18/10/2021'


from typing import List


class HabitatDataStructure :

    def __init__(self, kargs: dict) -> None :
        """
        Initialize the data structure according to the XML file used in
        the FeedingHabitat Class.

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