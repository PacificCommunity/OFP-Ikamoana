import sys
sys.path.append('./../')
import ikamoana.ikaseapodym as ikadym
from scipy.interpolate import griddata
from parcels import version, Field
from ikamoana.ikafish.ikafish import IkaFishDebug
from datetime import datetime
from parcels.tools.converters import TimeConverter
import numpy as np
import matplotlib.pylab as plt
print(version)

def create_IField(fieldset, field, res=0.1, interp = 'linear'):
    # create the grid of the interactive field
    lons = np.arange(field.lon[:].min(), field.lon[:].max()+res, res)
    lats = np.arange(field.lat[:].min(), field.lat[:].max()+res, res)
    # interpolate Field data on the interactive field
    gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
    points = np.swapaxes(np.vstack((gridlat.flatten(),
                                    gridlon.flatten())), 0, 1)
    values = field.data[0].flatten()
    grid_x, grid_y = np.meshgrid(lons, lats)
    dataP = np.expand_dims(griddata(points, values,
                                    (grid_y, grid_x),
                                    method=interp),
                           axis=0)
    if(False):  # compare original and interpolated fields
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(dataP[0])
        ax[1].imshow(field.data[0])
        plt.show()
    # Add interactive field to fieldset
    fieldP = Field('P', dataP, lon=lons, lat=lats,
                   mesh='spherical', interp_method='nearest',
                   time_origin=TimeConverter(datetime(2000, 1, 15)),
                   to_write=True)
    fieldset.add_field(fieldP)  # P field added to the velocity FieldSet

if(__name__=='__main__'):
    configuration_filepath = "./../data/ikamoana_config/IkaSim_Example_FishInteraction.xml"
    my_sim = ikadym.IkaSeapodym(filepath=configuration_filepath)

    my_sim.loadFields()

    # Create prey field and add to fieldset 
    create_IField(my_sim.ocean, my_sim.ocean.H)
    # Rate of field depletion (per particle per day)
    my_sim.ocean.add_constant('deplete', 0.1)
    # Prey field restoring time scale (days)
    my_sim.ocean.add_constant('restore', 15)

    my_sim.initializeParticleSet(particles_class=IkaFishDebug, method='start_cell')

    my_sim.runKernels(save=True,
                      output_name='/nethome/3830241/ikamoana/OFP-Ikamoana/notebook/outputfile.nc',
                      verbose=True)

print('executed')
