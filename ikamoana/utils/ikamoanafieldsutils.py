import numpy as np
import parcels
from parcels.tools.converters import Geographic, GeographicPolar


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
