import sys
import numpy as np
from parcels import Field
sys.path.append('/nethome/3830241/ikamoana/OFP-Ikamoana/')
import ikamoana.ikaseapodym as ikadym
from ikamoana.utils import seapodymFieldConstructor
from ikamoana.ikafish.ikafish import IkaFADFish
root = '/nethome/3830241/ikamoana/OFP-Ikamoana/tests/FishingIkatest/'


def add_mortality_field(fieldset, interp='nearest'):
    print(fieldset.U.data.shape)
    mort = np.ones(fieldset.U.data[0].shape)*0
    lats, lons = np.meshgrid(fieldset.U.lat, fieldset.U.lon)

    print(lats.shape, mort.shape)
    idx = np.where(np.logical_and(lons>175, lats>0))
    mort[idx] = 0.03
    fieldF = Field('F', mort, lon=fieldset.U.lon, lat=fieldset.U.lat,
                   mesh='spherical', interp_method=interp,
                   #time_origin=TimeConverter(datetime(2000, 1, 15)),
                   to_write=False)
    fieldset.add_field(fieldF)

def add_FADordersField(fieldset):
    # array used to keep track of FAD sorting according to their
    # numbers of associated tuna
    fieldF = Field('FADorders', np.arange(nfad), lon=np.arange(nfad), lat=np.array([0]), time=np.array([0]),
                   interp_method='nearest', mesh='spherical', allow_time_extrapolation=True)
    fieldset.add_field(fieldF) # prey field added to the velocity FieldSet
    fieldset.FADorders.to_write = False # enabling the writing of Field prey during execution


configuration_filepath = root+"Test_Params.xml"
IkaSim = ikadym.IkaSeapodym(filepath=configuration_filepath)
IkaSim.loadFields()
#IkaSim.oceanToNetCDF()
#'MPexp', 'MPmax', 'MSmax', 'MSslope', 'Mrange'

IkaSim.ocean.add_constant('RtF', 0.3)
IkaSim.ocean.add_constant('kappaF', 1.2)
# Parameters of the Logistic curve, which determines
# the dependence of FAD attraction strength on the number
# of associated tuna
IkaSim.ocean.add_constant("lL", 1.) # maximum of logistic curve
IkaSim.ocean.add_constant("lk", 0.35) # steepness of logistic curve
IkaSim.ocean.add_constant("lx0", 12) # value of the sigmoid midpoint
# And for the vp
IkaSim.ocean.add_constant("pL", 2.5) # maximum of logistic curve

# add beta distribution parameters
IkaSim.ocean.add_constant("betaa", 1.5)
IkaSim.ocean.add_constant("betab", 7)
assert IkaSim.ocean.betab > 1
nfad = 3
IkaSim.ocean.add_constant("nfad", nfad)
# determines the probability to catch at dominant FAD
# from the geometric distribution:
IkaSim.ocean.add_constant("p", 0.5)
# Fraction of FAD fishing vs. free school fishing:
IkaSim.ocean.add_constant("FADfishingP", 0)

add_mortality_field(IkaSim.ocean)
add_FADordersField(IkaSim.ocean)
IkaSim.initializeParticleSet(particles_class=IkaFADFish)

## Interaction kernel hacks
#First, put all particles near the middle of the domain to force some interaction
lons = np.random.uniform(174.5, 175.5,len(IkaSim.fish))
lats = np.random.uniform(-0.5, 0.5,len(IkaSim.fish))
for f in range(len(IkaSim.fish)):
    IkaSim.fish[f].lon = lons[f]
    IkaSim.fish[f].lat = lats[f]
# First nfad fish distributed in the domain are FADs and start near other particles
for f in range(nfad):
    IkaSim.fish[f].ptype = 1
    IkaSim.fish[f].lon = IkaSim.fish[f+nfad].lon + 0.0001
    IkaSim.fish[f].lat = IkaSim.fish[f+nfad].lat + 0.0001

for f in range(nfad,len(IkaSim.fish)):
    IkaSim.fish[f].ptype = 0
print('longitude of 1 tuna particle: ', IkaSim.fish[0].lon)

IkaSim.runKernels(save=True, verbose=False)



