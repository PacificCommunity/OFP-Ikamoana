from math import acos, asin, pi, sin, tan
from os.path import exists
from types import LambdaType
from typing import Tuple, Union

import numpy as np
import xarray as xr

from .. import dymfiles as df

def indexClosestCoord(
        coords: Union[list,np.ndarray,xr.DataArray], 
        value: Union[str,np.datetime64,int,float]
        ) -> int :
    """
    Return the position of the closest value in a specific coordinate.

    Parameters
    ----------
    coords : Union[list,np.ndarray,xr.DataArray]
        Coordinates in which we want to find the closest value to the
        `value` argument.
    value : Union[str,np.datetime64,int,float]
        The value we want to find the closest element in the `coords`
        argument.

    Returns
    -------
    int
        Index of the closest element to `value` in `coords`.
    """
    coords = np.array(coords)
    if isinstance(value, str):
        return np.argmin(np.abs(np.Datetime64(value, 'ns') - coords))
    else :
        return np.argmin(np.abs(value - coords))

def closestCoord(
        coords: Union[list,np.ndarray,xr.DataArray],
        value: Union[str,np.datetime64,int,float]
        ) -> Union[np.datetime64, float] :
    """
    Return the closest value in a specific coordinate.

    Parameters
    ----------
    coords : Union[list,np.ndarray,xr.DataArray]
        Coordinates in which we want to find the closest value to the
        `value` argument.
    value : Union[str,np.datetime64,int,float]
        The value we want to find the closest element in the `coords`
        argument.

    Returns
    -------
    Union[np.datetime64, float]
        The closest element to `value` in `coords`.
    """

    return coords[indexClosestCoord(coords, value)].data

def coordsAccess(coords: xr.Coordinate) -> Tuple[LambdaType,LambdaType,LambdaType]:
    """
    Return accessor to closest value for time, latitude and longitude
    coordinates.

    Parameters
    ----------
    coords : xr.Coordinate
        [description]

    See Also
    --------
    DataArray.loc
    DataArray.sel

    Returns
    -------
    Tuple[LambdaType,LambdaType,LambdaType]
        Tuple of lambda functions (time_access, lat_access, lon_access)
        which return the index of the closest value in `coords` argument.
    """
    return (lambda time : indexClosestCoord(coords['time'], time),
            lambda lat : indexClosestCoord(coords['lat'], lat),
            lambda lon : indexClosestCoord(coords['lon'], lon))

def seapodymFieldConstructor(
        filepath: str, dym_varname : str = None, dym_attributs : dict = None
        ) -> xr.DataArray :
    """
    Return a Seapodym field as a DataArray using NetCDF or Dym method
    according to the file extension : 'nc', 'cdf' or 'dym'.

    Parameters
    ----------
    filepath : str
        The path to the NetCDF or DYM.
    dym_varname : str, optional
        If the file is a DYM, dym_varname is the name of the variable
        represented inside the file. By default None which is replaced
        by the filepath.
    dym_attributs : str, optional
        If the file is a DYM, dym_attributs is , by default None

    Returns
    -------
    xr.DataArray
        [description]
    """
    if exists(filepath) :
        #NetCDF
        if filepath.lower().endswith(('.nc', '.cdf')) :
            return xr.open_dataarray(filepath)
        #DymFile
        if filepath.lower().endswith('.dym') :
            if dym_varname is None :
                dym_varname = filepath
            return df.dym2ToDataArray(infilepath = filepath,
                                      varname = dym_varname,
                                      attributs = dym_attributs)
    else :
        raise ValueError("No such file : {}".format(filepath))

def dayLengthPISCES(jday: int, lat: float) -> float:
    """
    Compute the day length depending on latitude and the day. New
    function provided by Laurent Bopp as used in the PISCES model and
    used by SEAPODYM in 2020.

    Parameters
    ----------
    jday : int
        Day of the year.
    lat : float
        Latitude.

    Modification
    ------------
    original       : E. Maier-Reimer (GBC 1993)
	additions      : C. Le Quere (1999)
	modifications  : O. Aumont (2004)
    	Adapted to C      : P. Lehodey (2005)
        Adapted to Python : J. Lehodey (2021)

    Returns
    -------
    float
        The duration of the day (i.e. while the sun is shining) as a ratio in
        range [0,1].

    """

    rum = (jday - 80.0) / 365.25
    delta = sin(rum * pi * 2.0) * sin(pi * 23.5 / 180.0)
    codel = asin(delta)
    phi = lat * pi / 180.0

    argu = tan(codel) * tan(phi)
    argu = min(1.,argu)
    argu = max(-1.,argu)

    day_length = 24.0 - (2.0 * acos(argu) * 180.0 / pi / 15 )
    day_length = max(day_length,0.0)

    return day_length / 24.0
