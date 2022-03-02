import xml.etree.ElementTree as ET
from typing import List, Union

import numpy as np
import parcels
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar


def convertToField(field : Union[xr.DataArray, xr.Dataset], name=None) :
    """Converts a DataSet/DataArray to a `parcels.FieldSet`."""

    if isinstance(field, xr.DataArray) :
        field = field.to_dataset(name=name if name is not None else field.name)

    return parcels.FieldSet.from_xarray_dataset(
        ((field.reindex(lat=field.lat[::-1]))
         if field.lat[0] > field.lat[-1] else field),
        variables=dict([(i,i) for i in field.keys()]),
        dimensions=dict([(i,i) for i in field.dims.keys()]))

def convertToNauticMiles(
        field:Union[xr.DataArray, xr.Dataset], timestep:float
        ) -> Union[xr.DataArray, xr.Dataset] :
    """Converts the unit of a field from meters per second to nautical
    miles per timestep."""
    
    convertion = lambda x : x * timestep / 1852
    return xr.apply_ufunc(convertion, field)

def sliceField(
        field : Union[xr.DataArray, xr.Dataset], time_start: int = None,
        time_end: int = None, lat_min: int = None, lat_max: int = None,
        lon_min: int = None, lon_max: int = None
        ) -> Union[xr.DataArray, xr.Dataset] :
    """
    This function is equivalent to `xarray.DataArray.isel()`. Moreover,
    sliceField will not automaticaly find the nearest value.
    
    See Also
    --------
    xr.DataArray.loc
    xr.DataArray.sel
    xr.DataArray.isel
    slice
    """

    coords = field.coords

    if (lat_min is not None) :
        if ((lat_min < 0) or (lat_min >= coords['lat'].data.size)) :
            raise ValueError("lat_min out of bounds. Min is %d and Max is %d"%(
                0, coords['lat'].data.size - 1))
    if (lat_max is not None) :
        if ((lat_max < 0) or (lat_max >= coords['lat'].data.size)) :
            raise ValueError("lat_max out of bounds. Min is %d and Max is %d"%(
                0, coords['lat'].data.size - 1))
    if (lat_min is not None) and (lat_max is not None) and (lat_min > lat_max) :
        raise ValueError("lat_min must be <= to lat_max.")

    if (lon_min is not None) :
        if ((lon_min < 0) or (lon_min >= coords['lon'].data.size)) :
            raise ValueError("lon_min out of bounds. Min is %d and Max is %d"%(
                0, coords['lon'].data.size - 1))
    if (lon_max is not None) :
        if ((lon_max < 0) or (lon_max >= coords['lon'].data.size)) :
            raise ValueError("lon_max out of bounds. Min is %d and Max is %d"%(
                0, coords['lon'].data.size - 1))
    if (lon_min is not None) and (lon_max is not None) and (lon_min > lon_max) :
        raise ValueError("lon_min must be <= to lon_max.")

    if (time_start is not None) :
        if ((time_start < 0) or (time_start >= coords['time'].data.size)) :
            raise ValueError("time_start out of bounds. Min is %d and Max is %d"%(
                0, coords['time'].data.size - 1))
    if (time_end is not None) :
        if ((time_end < 0) or (time_end >= coords['time'].data.size)) :
            raise ValueError("time_end out of bounds. Min is %d and Max is %d"%(
                0, coords['time'].data.size - 1))
    if (time_start is not None) and (time_end is not None) and (time_start > time_end) :
        raise ValueError("time_start must be <= to time_end.")

    coord_lat = coords["lat"][lat_min:lat_max+1 if lat_max is not None else None]
    coord_lon = coords["lon"][lon_min:lon_max+1 if lon_max is not None else None]
    coord_time = coords["time"][time_start:time_end+1 if time_end is not None else None]

    return field.sel(time=coord_time, lat=coord_lat, lon=coord_lon)

def getCellEdgeSizes(field) :
    """Calculate the size (in kilometers) of each cells of a grid
    defined by latitude and longitudes coordinates. Copy of the
    `Parcels.Field.calc_cell_edge_sizes` function in Parcels. Avoid the
    convertion of DataArray into `Parcels.Field`.
    
    It already take into account the latitude correction for narrower
    grid cells closer to the poles.

    Returns
    -------
    Tuple
        (x : longitude edge size, y : latitude edge size)
    """

    field_grid = parcels.grid.RectilinearZGrid(
        field.lon.data, field.lat.data,
        depth=None, time=None, time_origin=None,
        mesh='spherical') # In degrees

    field_grid.cell_edge_sizes['x'] = np.zeros((field_grid.ydim, field_grid.xdim), dtype=np.float32)
    field_grid.cell_edge_sizes['y'] = np.zeros((field_grid.ydim, field_grid.xdim), dtype=np.float32)

    x_conv = GeographicPolar()
    y_conv = Geographic()

    for y, (lat, dlat) in enumerate(zip(field_grid.lat, np.gradient(field_grid.lat))):
        for x, (lon, dlon) in enumerate(zip(field_grid.lon, np.gradient(field_grid.lon))):
            field_grid.cell_edge_sizes['x'][y, x] = x_conv.to_source(dlon, lon, lat, field_grid.depth[0])
            field_grid.cell_edge_sizes['y'][y, x] = y_conv.to_source(dlat, lon, lat, field_grid.depth[0])

    return field_grid.cell_edge_sizes['x'], field_grid.cell_edge_sizes['y']

def latitudeDirection(
        field: Union[xr.DataArray, xr.Dataset], south_to_north: bool = False) :
    """Reindexes the latitude axis of a field in a south-north or
    north-south direction according to the `south_to_north` argument."""
    
    f_is_north_to_south = field.lat.data[0] > field.lat.data[-1]
    # logical XAND 
    if ((f_is_north_to_south and south_to_north)
        or (not f_is_north_to_south and not south_to_north)) :
        return field.reindex(lat=field.lat[::-1])
    else :
        return field

def tagReading(
        root: ET.Element, tags: Union[str,List[str]],
        default: Union[str,int,float] = None, attribute: str = None
        ) -> Union[int,float,str]:
    """Move through a chain of XML `tags` to read a parameter. Return
    `default` value if this parameter `text` (or a specific `attribute`)
    is empty."""
    
    tags = np.ravel(tags)
    elmt = root.find(tags[0])
    for tag_name in tags[1:]:
        elmt = elmt.find(tag_name)
        
    elmt = elmt.text if attribute is None else elmt.attrib[attribute]
    return default if (elmt == '') or (elmt is None) else elmt
