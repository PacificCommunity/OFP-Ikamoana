import matplotlib.pylab as plt
from netCDF4 import Dataset

dirr = '/nethome/3830241/ikamoana/Sdata/SourcePrey_1deg/'

ncf = Dataset(dirr+'glo_freeglorys_2006-2015_MTLPP_20151015.nc')

lon = ncf['longitude'][:]
lat = ncf['latitude'][:]
PS = ncf['epi_mnk_pp']
ncf.close()

im = plt.pcolormesh(lon, lat, PS)
plt.colorbar(PS)
plt.show()


