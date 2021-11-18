from typing import Tuple, Union, List

import numpy as np
import parcels
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar

from ..feedinghabitat import FeedingHabitat
from .ikamoanafieldsconfigreader import readIkamoanaFieldsXML


def convertToField(field : Union[xr.DataArray, xr.Dataset], name="Name_to_convert") :

    if isinstance(field, xr.DataArray) :
        field = field.to_dataset(name=name if field.name is None else None)

    return parcels.FieldSet.from_xarray_dataset(
        (field.reindex(lat=list(reversed(field.lat)))
         if field.lat[0] > field.lat[-1] else field),
        variables=dict([(i,i) for i in field.keys()]),
        dimensions=dict([(i,i) for i in field.dims.keys()]))

class IkamoanaFields :

    def __init__(self,
                 xml_fields : str,
                 xml_feeding_habitat : str,
                 feeding_habitat : xr.DataArray = None):
        """Create a IkamoanaFields class. Can compute Taxis, Current and Diffusion
        fields."""

        self.ikamoana_fields_structure = readIkamoanaFieldsXML(xml_fields)
        self.feeding_habitat_structure = FeedingHabitat(xml_feeding_habitat)
        self.feeding_habitat = feeding_habitat

    def vMax(self, length : float) -> float :
        """Return the maximum velocity of a fish with a given length."""

        return (self.ikamoana_fields_structure.vmax_a
             * np.power(length, self.ikamoana_fields_structure.vmax_b))

    def landmask(self, habitat_field : xr.DataArray = None,
                 shallow_sea_to_ocean=False, lim=1e-45) -> xr.DataArray :
        """Return the landmask of a given habitat or FeedingHabitat.global_mask.
        Mask values :
            2 -> is Shallow
            1 -> is Land or No_Data
            0 -> deep ocean with habitat data

        Note:
        -----
            Landmask in Original (with Parcels Fields) is flipped on latitude axis.
        """

        if habitat_field is None :
            mask_L1 = np.invert(
                self.feeding_habitat_structure.data_structure.global_mask['mask_L1'])[0,:,:]
            mask_L3 = np.invert(
                self.feeding_habitat_structure.data_structure.global_mask['mask_L3'])[0,:,:]
            
            landmask = np.zeros(mask_L1.shape, dtype=np.int8)
            if not shallow_sea_to_ocean : landmask[mask_L3] = 2
            landmask[mask_L1] = 1

            coords = self.feeding_habitat_structure.data_structure.coords

        else :
            habitat_f = habitat_field[0,:,:]
            ## TODO : Should I use temperature_L3 rather than forage_lmeso ?
            lmeso_f = self.feeding_habitat_structure.data_structure.variables_dictionary['forage_lmeso'][0,:,:]

            if habitat_f.shape != lmeso_f.shape :
                raise ValueError("Habitat and forage_lmeso must have the same dimension.")

            landmask = np.zeros_like(habitat_f)
            if not shallow_sea_to_ocean :
                landmask[(np.abs(lmeso_f) <= lim) | np.isnan(lmeso_f)] = 2
            landmask[(np.abs(habitat_f) <= lim) | np.isnan(habitat_f)] = 1

            coords = habitat_field.coords

        ## TODO : Ask why lon is between 1 and ny-1
        # Answer -> in-coming
        landmask[-1,:] = landmask[0,:] = 0

        return xr.DataArray(
                data=landmask,
                name='landmask',
                coords={'lat':coords['lat'],
                        'lon':coords['lon']},
                dims=('lat', 'lon')
            )

    def gradient(self,
                 field: xr.DataArray,
                 landmask: xr.DataArray) -> Tuple[xr.DataArray]:

        """
        Gradient calculation for a Xarray DataArray seapodym-equivalent calculation
        requires LandMask forward and backward differencing for domain edges
        and land/shallow sea cells.
        """

        if ((field.lat.size != landmask.lat.size)
                or (field.lon.size != landmask.lon.size)) :
            raise ValueError("Field and landmask must have the same dimension.")

        ## WARNING : To have the same behavior as original gradient function,
        # latitude must be south-north rather than north-south.

        flip_lat = field.lat[0] > field.lat[-1]
        if flip_lat : field = field.reindex(lat=np.flip(field.lat))
        if landmask.lat[0] > landmask.lat[-1] :
            landmask = landmask.reindex(lat=np.flip(landmask.lat))

        def getCellEdgeSizes(field) :
            """Copy of the Field.calc_cell_edge_sizes() function in Parcels.
            Avoid the convertion of DataArray into Field."""
        
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

        dlon, dlat = getCellEdgeSizes(field)
        
        nlat = field.lat.size
        nlon = field.lon.size

        data = np.nan_to_num(field.data)
        landmask = landmask.data
        dVdlon = np.zeros(data.shape, dtype=np.float32)
        dVdlat = np.zeros(data.shape, dtype=np.float32)

        ## NOTE : Parallelised execution may help to do it faster.
        # I think it can also be vectorized.
        for t in range(field.time.size):
            for lon in range(1, nlon-1):
                for lat in range(1, nlat-1):
                    if landmask[lat, lon] < 1:
                        if landmask[lat, lon+1] == 1:
                            dVdlon[t,lat,lon] = (data[t,lat,lon] - data[t,lat,lon-1]) / dlon[lat, lon]
                        elif landmask[lat, lon-1] == 1:
                            dVdlon[t,lat,lon] = (data[t,lat,lon+1] - data[t,lat,lon]) / dlon[lat, lon]
                        else:
                            dVdlon[t,lat,lon] = (data[t,lat,lon+1] - data[t,lat,lon-1]) / (2*dlon[lat, lon])

                        if landmask[lat+1, lon] == 1:
                            dVdlat[t,lat,lon] = (data[t,lat,lon] - data[t,lat-1,lon]) / dlat[lat, lon]
                        elif landmask[lat-1, lon] == 1:
                            dVdlat[t,lat,lon] = (data[t,lat+1,lon] - data[t,lat,lon]) / dlat[lat, lon]
                        else:
                            dVdlat[t,lat,lon] = (data[t,lat+1,lon] - data[t,lat-1,lon]) / (2*dlat[lat, lon])

            for lon in range(nlon):
                dVdlat[t,0,lon] = (data[t,1,lon] - data[t,0,lon]) / dlat[0,lon]
                dVdlat[t,-1,lon] = (data[t,-1,lon] - data[t,-2,lon]) / dlat[-2,lon]
            for lat in range(nlat):
                dVdlon[t,lat,0] = (data[t,lat,1] - data[t,lat,0]) / dlon[lat,-1]
                dVdlon[t,lat,-1] = (data[t,lat,-1] - data[t,lat,-2]) / dlon[lat,-1]

        return (xr.DataArray(
                    name = 'd' + field.name + '_dlon',
                    data = dVdlon,
                    coords = field.coords,
                    dims=('time','lat','lon'),
                    attrs=field.attrs),
                xr.DataArray(
                    name = 'd' + field.name + '_dlat',
                    data = np.flip(dVdlat, axis=1) if flip_lat else dVdlat,
                    coords = field.coords,
                    dims=('time','lat','lon'),
                    attrs=field.attrs))

    def taxis(self, dHdlon: xr.DataArray, dHdlat: xr.DataArray) -> Tuple[xr.DataArray,xr.DataArray] : 
        """
        Calculation of the Taxis field from the gradient.
        """

        def argumentCheck(array) :
            if array.attrs.get('cohort_start') is not None :
                is_evolving = True
                age = array.cohorts
            elif array.attrs.get('Cohort number') is not None :
                is_evolving = False
                age = array.attrs.get('Cohort number')
            else :
                raise ValueError("Fields must contain either 'cohort_start' or 'Cohort number'")
            return is_evolving, age

        is_evolving, age = argumentCheck(dHdlon)
        Tlon = np.zeros(dHdlon.data.shape, dtype=np.float32)
        Tlat = np.zeros(dHdlat.data.shape, dtype=np.float32)
        lat_tile_transpose_cos = np.cos(
            np.tile(dHdlon.lat.data, (dHdlon.lon.size, 1)).T
            * np.pi/180)
        factor = self.ikamoana_fields_structure.taxis_scale * 250 * 1.852 * 15
        f_length = self.feeding_habitat_structure.data_structure.findLengthByCohort

        for t in range(dHdlon.time.size):
            t_age = age[t] if is_evolving else age
            # Convert cm to meter (/100) : See original function
            t_length = f_length(t_age) / 100

            Tlon[t,:,:] = (self.vMax(t_length)
                           * dHdlon.data[t,:,:]
                           * factor * lat_tile_transpose_cos)
            Tlat[t,:,:] = (self.vMax(t_length)
                           * dHdlat.data[t,:,:]
                           * factor)

        if self.ikamoana_fields_structure.units == 'nm_per_timestep':
            Tlon *= (16/1852)
            Tlat *= (16/1852)
        ## NOTE :       (timestep/1852) * (1000*1.852*60) * 1/timestep
        #           <=> (250*1.852*15) * (16/1852)

        return (xr.DataArray(name = 'Tlon',
                             data = Tlon,
                             coords = dHdlon.coords,
                             dims=('time','lat','lon'),
                             attrs=dHdlon.attrs),
                xr.DataArray(name = 'Tlat',
                             data = Tlat,
                             coords = dHdlat.coords,
                             dims=('time','lat','lon'),
                             attrs=dHdlat.attrs))
