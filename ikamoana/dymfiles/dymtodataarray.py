"""This module generate xarray.DataArray from DYM files."""

from .core import utilities, dym2, dym
import xarray as xr
import numpy as np

def dym2ToDataArray(
        infilepath: str, varname: str, attributs: dict = None, dymformat = 'dym2') -> xr.DataArray:
    """
    Generate a `xarray.DataArray` from the DYM file which the filepath
    is `infile`.

    Parameters
    ----------
    infilepath : str
        The path to the DYM file.
    varname : str
        The name of the generated DataArray.
    attributs : dict, optional
        Attributs to add to the generated DataArray, by default None.

    Returns
    -------
    xarray.DataArray
        The DataArray generated from the DYM file.
    """
    if dymformat == 'dym2':
        inDym  = dym2.DymFile(infilepath)
    else:
        inDym = dym.DymFile(infilepath)
    #print(inDym.header_.firstDate_)

    veclon = inDym.header_.xLon_[:,0]
    veclat = inDym.header_.yLat_[0,:]
    n_time   = inDym.header_.nLev_

    #is the first coordinate time, or age?
    if utilities.date_dym2tostr(inDym.header_.zLev_[0])[0] == 'x':
        age_structured = True
        dict_time = dict(standard_name='age')
    else:
        age_structured = False
        dict_time = dict(standard_name='time')

    dict_lat = dict(standard_name ='latitude',units = 'degrees_north')
    dict_lon = dict(standard_name ='longitude',units = 'degrees_east')

    outdata_list = []

    if age_structured:
        age_list = []
        for k in range(n_time):
            age_list.append(inDym.header_.zLev_[k])
            filval = np.NaN
            outdata = inDym.readData(k + 1)
            outdata[outdata == -999] = filval
            outdata_list.append(outdata)

        # Xarray DataArray creation
        returned_da = xr.DataArray(data=np.array(outdata_list),
                                   name=varname,
                                   dims=('age', 'lat', 'lon'),
                                   coords=(('age', np.array(age_list, dtype='float32'), dict_time),
                                           ('lat', veclat, dict_lat),
                                           ('lon', veclon, dict_lon)),
                                   attrs=attributs
                                   )
    else:
        datestr_list = []

        # Extract all values from Dym structure and transform date
        for k in range(n_time):
            datestr = utilities.date_dym2tostr(inDym.header_.zLev_[k])
            datestr = np.datetime64(datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8])
            datestr_list.append(datestr)
            filval = np.NaN
            outdata = inDym.readData(k+1)
            outdata[outdata==-999] = filval
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
