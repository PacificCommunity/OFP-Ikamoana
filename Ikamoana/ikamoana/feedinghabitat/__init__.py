"""
Summary
-------
This class can simulate the feeding habitat in the same manner as
the SEAPODYM model (2020-08). The FeedingHabitat class first
initializes the HabitatDataStructure class using the
FeedingHabitatConfigReader module. Then, it performs the feeding
habitat calculation for each cohort specified as an argument,
returned as a DataSet. It can also calculate the habitat for a
cohort that changes over time (moving from one cohort to another).

Examples
--------

First example : Simply initialize with Xml file. Then print some informations
    about the data used by the module. Finaly, compute Feeding Habitat for the
    first cohort on complet time, latitude and longitude series.

>>> fh = ikamoana.feedinghabitat.FeedingHabitat(
...     xml_filepath="./path/to/file.xml")
>>> fh.data_structure.summary()
# ------------------------------ #
# Summary of this data structure #
# ------------------------------ #
[...Many informations to print...]
>>> result = fh.computeFeedingHabitat(cohorts=0)


Second example : Compute Feeding Habitat with sub-part of time, latitude and
    longitude series for every cohorts.

>>> import numpy as np
>>> fh = ikamoana.feedinghabitat.FeedingHabitat(
...     xml_filepath="./path/to/file.xml")
>>> result = fh.computeFeedingHabitat(
...     cohorts=np.arange(0, fh.data_structure.cohorts_number),
...     time_start=12, time_end=36,
...     lat_min=None, lat_max=40,
...     lon_min=10, lon_max=None,
...     verbose=True)
Computing Feeding Habitat for cohort 0.
Computing Feeding Habitat for cohort 1.
Computing Feeding Habitat for cohort 2.
[...]
Computing Feeding Habitat for cohort [cohorts_number-1].

Third example : Calculate the habitat for a cohort that evolves over time.

>>> fh = ikamoana.feedinghabitat.FeedingHabitat(
...     xml_filepath="./path/to/file.xml")
>>> result = fh.computeEvolvingFeedingHabitat(
...     cohort_start=0, cohort_end=25)
<xarray.DataArray 'Feeding_Habitat_Cohort_0_to_25' (time: 26,
    lat: ..., lon: ...)>
array(...)
Coordinates:
    (...)
[...Many informations to print...]

"""

from .feedinghabitat import FeedingHabitat
from .feedinghabitat import closestCoord, indexClosestCoord, coordsAccess