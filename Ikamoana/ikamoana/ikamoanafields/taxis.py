from ..feedinghabitat import FeedingHabitat
from .forcingfield import ForcingField
from functools import singledispatchmethod

## NOTE : 2 way to compute Taxis :
# - Using a Feeding Habitat field and calculate gradient then taxis
# - Compute Feeding habitat on the fly then calculate gradient and taxis
#
# Then, Taxis class can be initialized with or without ForcingField.
# But if there is no forcingfield, FeedingHabitat need to be computed
# in Taxis computation functions.

class Taxis :

    # + feeding_habitat : ForcingField
    # + taxis_data_structure : dict

    @singledispatchmethod
    def __init__(self, feeding_habitat : ForcingField) :
## NOTE : Do we have the possibility to init directly from a ForcingField ?
        self.feeding_habitat = feeding_habitat
## TODO : define what to add in this dictionary. Parameters from XML ?
        self.taxis_data_structure = None
    
    @__init__.register
    def _(self, feeding_habitat : str) :
        self.feeding_habitat = FeedingHabitat(xml_filepath=feeding_habitat)
    
    def computeTaxisField(self):
        """Wrapper from FeedingHabitat.computeFeedingHabitat if
        self.feeding_habitat is None. Then compute Gradient and Taxis."""
        pass

    def computeEvolvingTaxisField(self):
        """Wrapper from FeedingHabitat.computeEvolvingFeedingHabitat if
        self.feeding_habitat is None. Then compute Gradient and Taxis."""
        pass


    