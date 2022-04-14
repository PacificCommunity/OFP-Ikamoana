from typing import Union

import numpy as np
import parcels
import xarray as xr


def convertToField(
        field : Union[xr.DataArray, xr.Dataset], name=None
        ) -> parcels.Field:
    """Converts a DataSet/DataArray to a `parcels.FieldSet`."""

    if isinstance(field, xr.DataArray) :
        field = field.to_dataset(name=name if name is not None else field.name)

    return parcels.FieldSet.from_xarray_dataset(
        ((field.reindex(lat=field.lat[::-1]))
         if field.lat[0] > field.lat[-1] else field),
        variables=dict([(i,i) for i in field.keys()]),
        dimensions=dict([(i,i) for i in field.dims.keys()]))

def convertToDataArray(field: parcels.Field) -> xr.DataArray:
    """Convert a parcels field into a xarray DataArray."""
    
    origin = np.datetime64(str(field.grid.time_origin))
    time_list = field.grid.time
    convert = lambda origin, time : origin + np.timedelta64(int(time), "s")
    time_list = [convert(origin, time) for time in time_list]
    
    return xr.DataArray(
        data=field.data,
        coords={"time":time_list,
                "lat":field.lat,
                "lon":field.lon}
    )

def convertToNauticMiles(
        field:Union[xr.DataArray, xr.Dataset], timestep: float = 1.,
        square: bool = False
        ) -> Union[xr.DataArray, xr.Dataset] :
    """Converts the unit of a field from meters per second to nautical
    miles per timestep."""
    divider = 1852 if  not square else 1852**2
    factor = timestep / divider
    convertion = lambda x : x * factor
    return xr.apply_ufunc(convertion, field)

def convertToMeters(
        field:Union[xr.DataArray, xr.Dataset], timestep: float = 1.,
        square: bool = False
        ) -> Union[xr.DataArray, xr.Dataset] :
    """Converts the unit of a field from nautic miles per timestep to
    meters per second."""
    multiply = 1852 if not square else 1852**2
    factor = multiply / timestep
    convertion = lambda x : x * factor
    return xr.apply_ufunc(convertion, field)
