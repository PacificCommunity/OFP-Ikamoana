__author__ = "J. Lehodey"
__date__ = "2021-10-15"

from .core import utilities, dym2
import xarray as xr
import numpy as np

def dym2ToDataArray(infile, varname, attributs=None):

    inDym  = dym2.DymFile(infile)

    veclon = inDym.header_.xLon_[:,0]
    veclat = inDym.header_.yLat_[0,:]
    n_time   = inDym.header_.nLev_

    dict_time = dict(standard_name='time')
    dict_lat = dict(standard_name ='latitude',units = 'degrees_north')
    dict_lon = dict(standard_name ='longitude',units = 'degrees_east')

    outdata_list = []
    datestr_list = []
    
    # Extract all values from Dym structure and transform date
    for k in range(n_time):
        
        datestr = utilities.date_dym2tostr(inDym.header_.zLev_[k])
        datestr = np.datetime64(datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8])
        datestr_list.append(datestr)
        filval = np.NaN
        outdata = inDym.readData(k+1)
        outdata[outdata==0] = filval
        outdata_list.append(outdata)
        
    # Xarray DataArray creation
    returned_da = xr.DataArray(data=np.array(outdata_list),
                               name=varname,
                               dims=('time','lat','lon'),
                               coords=(('time',np.array(datestr_list,dtype='datetime64[D]'),dict_time),
                                       ('lat',veclat,dict_lat),
                                       ('lon',veclon,dict_lon)),
                               attrs=attributs
                              )
    return returned_da
