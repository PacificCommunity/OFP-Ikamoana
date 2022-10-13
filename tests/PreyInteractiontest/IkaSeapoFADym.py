import sys
from glob import glob
from parcels import version, Field
from parcels.tools.converters import TimeConverter
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import numpy as np
sys.path.append('/nethome/3830241/ikamoana/OFP-Ikamoana/')
import ikamoana.ikaseapodym as ikadym
from ikamoana.utils import seapodymFieldConstructor
from ikamoana.ikafish.ikafish import IkaFADFish
root = '/nethome/3830241/ikamoana/OFP-Ikamoana/tests/PreyInteractiontest/'
print(version)
from dateutil.relativedelta import relativedelta

def remove_land(fieldset, glon, glat, vals):
    ocean = np.logical_and(fieldset.U.data[0].flatten()!=0,
                           fieldset.V.data[0].flatten()!=0)
    glon = glon[ocean]
    glat = glat[ocean]
    vals = vals[ocean]
    return vals, glon, glat

def add_PreySourceField(fieldset):
    dirr = '/nethome/3830241/ikamoana/Sdata/SourcePrey_1deg/'
    fn = np.sort(glob(dirr+'*MTLPP*.nc'))
    fnames = {'lon': fn[0], 'lat': fn[0], 'data': fn}
    varb =  'epi_mnk_pp'
#    ts = np.array([datetime(2006, 1, 15) + relativedelta(months=+i) for i in range(120)])
#    print(ts)
    dims = {'lat': 'latitude', 'lon': 'longitude', 'time': 'Time'}
    PS = Field.from_netcdf(fnames, varb, dims)#, timestamps=ts)
    fieldset.add_field(PS)  # P field added to the velocity FieldSet

#    print(PS.time_index)
#    print(fieldset.U.time_index)
#    print(PS.computeTimeChunk('varb', 3))
#    assert False

def create_IField(fieldset, field, res=0.1, interp = 'linear'):
    # create the grid of the interactive field
    lons = np.arange(field.lon[:].min(), field.lon[:].max()+res, res)
    lats = np.arange(field.lat[:].min(), field.lat[:].max()+res, res)
    # interpolate Field data on the interactive field
    gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
    values = field.data[0].flatten()
    # remove land points before interpolation
    values, gridlon, gridlat = remove_land(fieldset,
                                           gridlon.flatten(),
                                           gridlat.flatten(),
                                           values)
    points = np.swapaxes(np.vstack((gridlat,
                                    gridlon)), 0, 1)
    grid_x, grid_y = np.meshgrid(lons, lats)
    dataP = np.expand_dims(griddata(points, values,
                                    (grid_y, grid_x),
                                    method=interp),
                           axis=0)

    if(False):  # compare original and interpolated fields
        import matplotlib.pylab as plt
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(dataP[0])
        ax[1].imshow(field.data[0])
        plt.show()
        assert False

    # Add interactive field to fieldset
    fieldP = Field('P', dataP, lon=lons, lat=lats,
                   mesh='spherical', interp_method=interp,
                   #time_origin=TimeConverter(datetime(2000, 1, 15)),
                   to_write=True)
    fieldset.add_field(fieldP)  # P field added to the velocity FieldSet
    # Add interactive field that keeps track between the difference between P and H fields
#    fieldP = Field('Pdiff', np.zeros(dataP.shape), lon=lons, lat=lats,
#                   mesh='spherical', interp_method='nearest',
#                   #time_origin=TimeConverter(datetime(2000, 1, 15)),
#                   to_write=False)
#    fieldset.add_field(fieldP)  # P field added to the velocity FieldSet

def create_LMaskField(fieldset, field, res=0.1, interp = 'nearest'):
    # create the grid of the interactive field
    lons = np.arange(field.lon[:].min(), field.lon[:].max()+res, res)
    lats = np.arange(field.lat[:].min(), field.lat[:].max()+res, res)
    # interpolate Field data on the interactive field
    gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
    values = field.data[0].flatten()
    points = np.swapaxes(np.vstack((gridlat.flatten(),
                                    gridlon.flatten())), 0, 1)
    grid_x, grid_y = np.meshgrid(lons, lats)
    dataU = np.expand_dims(griddata(points, values,
                                    (grid_y, grid_x),
                                    method=interp),
                           axis=0)
    land = (dataU==0)

    # Add interactive field to fieldset
    fieldLand = Field('Land', land, lon=lons, lat=lats,
                   mesh='spherical', interp_method='nearest',
                   to_write=False)
    fieldset.add_field(fieldLand)  # P field added to the velocity FieldSet
    fieldset.P.data[land] = 0


configuration_filepath = root+"Test_Params.xml"
IkaSim = ikadym.IkaSeapodym(filepath=configuration_filepath)
IkaSim.loadFields()

# Create prey field and add to fieldset 
create_IField(IkaSim.ocean, IkaSim.ocean.H)
create_LMaskField(IkaSim.ocean, IkaSim.ocean.U)
add_PreySourceField(IkaSim.ocean)
#IkaSim.oceanToNetCDF()
IkaSim.initializeParticleSet(particles_class=IkaFADFish)

## Interaction kernel hacks
#First, put all particles near the middle of the domain to force some interaction
lons = np.random.uniform(174.5, 175.5,len(IkaSim.fish))
lats = np.random.uniform(-0.5, 0.5,len(IkaSim.fish))
for f in range(len(IkaSim.fish)):
    IkaSim.fish[f].lon = lons[f]
    IkaSim.fish[f].lat = lats[f]
# First 50 fish distributed in the domain are FADs and start near other particles
nfad = 2
for f in range(nfad):
    IkaSim.fish[f].ptype = 1
    IkaSim.fish[f].lon = IkaSim.fish[f+nfad].lon + 0.0001
    IkaSim.fish[f].lat = IkaSim.fish[f+nfad].lat + 0.0001

for f in range(nfad,len(IkaSim.fish)):
    IkaSim.fish[f].ptype = 0

IkaSim.ocean.add_constant('RtF', 2.)
IkaSim.ocean.add_constant('kappaF', 1.2)
# Parameters of the Logistic curve, which determines
# the dependence of FAD attraction strength on the number
# of associated tuna
IkaSim.ocean.add_constant("lL", 1.) # maximum of logistic curve
IkaSim.ocean.add_constant("lk", 0.35) # steepness of logistic curve
IkaSim.ocean.add_constant("lx0", 12) # value of the sigmoid midpoint
# And for the vp
IkaSim.ocean.add_constant("pL", 2.5) # maximum of logistic curve

# Rate of field depletion (per particle per day)
IkaSim.ocean.add_constant('deplete', 0.05)
# Prey field restoring time scale (days)
#IkaSim.ocean.add_constant('restore', 15)
IkaSim.ocean.add_constant('restore', 30)

print('longitude of 1 tuna particle: ', IkaSim.fish[0].lon)

IkaSim.runKernels(save=True, verbose=False)



