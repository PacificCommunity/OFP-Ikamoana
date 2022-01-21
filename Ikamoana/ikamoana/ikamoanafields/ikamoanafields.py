import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union

import numpy as np
import parcels
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar

from ..feedinghabitat import FeedingHabitat, coordsAccess
from ..feedinghabitat import feedinghabitatconfigreader as fhcf
from ..fisherieseffort import fisherieseffort
from .fieldsdatastructure import FieldsDataStructure


def convertToField(field : Union[xr.DataArray, xr.Dataset], name=None) :
    """Converts a DataSet/DataArray to a `parcels.FieldSet`."""

    if isinstance(field, xr.DataArray) :
        field = field.to_dataset(name=name if name is not None else field.name)

    return parcels.FieldSet.from_xarray_dataset(
        ((field.reindex(lat=field.lat[::-1]))
         if field.lat[0] > field.lat[-1] else field),
        variables=dict([(i,i) for i in field.keys()]),
        dimensions=dict([(i,i) for i in field.dims.keys()]))

def sliceField(field : Union[xr.DataArray, xr.Dataset],
               time_start: int = None, time_end: int = None,
               lat_min: int = None, lat_max: int = None,
               lon_min: int = None, lon_max: int = None
               ) -> Union[xr.DataArray, xr.Dataset] :
    """
    This function is equivalent to `xarray.DataArray.isel()`. Moreover,
    sliceField will not automaticaly find the nearest value.
    
    See Also
    --------
    xr.DataArray.loc[]
    xr.DataArray.sel()
    xr.DataArray.isel()
    python slice()
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

# NOTE : maybe a generalized function would be usefull. Depending on the
# user needs.
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

class IkamoanaFields :

# ------------------------- CORE FUNCTIONS ------------------------- #

    def __init__(
            self, IKAMOANA_config_filepath : str,
            root_directory: str = None, SEAPODYM_config_filepath : str = None,
            feeding_habitat : xr.DataArray = None):
        """Create a IkamoanaFields class. Can compute Taxis, Current, Diffusion,
        and mortality fields.

        Parameters
        ----------
        IKAMOANA_config_filepath : str
            Path to the IKAMOANA configuration XML file.
        root_directory : str, optional
            If the SEAPODYM configuration file is not in the root of the
            working directory, this directory path must be specified.
        SEAPODYM_config_filepath : str, optional
            SEAPODYM configuration filepath can also be specified by
            user rather than in the IKAMOANA configuration file.
        feeding_habitat : xr.DataArray, optional
            If the feeding habitat has already been calculated, it can
            be passed directly to the constructor.

        Raises
        ------
        TypeError
            feeding_habitat must be a Xarray.DataArray or None.
        """

        self.ikamoana_fields_structure = FieldsDataStructure(
            IKAMOANA_config_filepath, root_directory, SEAPODYM_config_filepath)
        self.feeding_habitat_structure = FeedingHabitat(
            self.ikamoana_fields_structure.SEAPODYM_config_filepath)
        
        if (feeding_habitat is None) or isinstance(feeding_habitat,xr.DataArray) :
            self.feeding_habitat = feeding_habitat
        else :
            raise TypeError((
                "feeding_habitat must be a Xarray.DataArray or None."
                "Current type is : {}").format(type(feeding_habitat)))

    def readFisheriesXML(
            self, xml_config_file: str, species_name: str = None
            ) -> dict :
        """Read a XML file to get all parameters needed for mortality
        field production."""

        tree = ET.parse(xml_config_file)
        root = tree.getroot()
        if species_name == None :
            species_name = root.find("sp_name").text

        nb_fishery = int(root.find('nb_fishery').attrib['value'])
        list_fishery_name = root.find('list_fishery_name').text.split()

        # fisheries name
        if len(list_fishery_name) != nb_fishery :
            raise ValueError((
                "nb_fishery is {} but list_fishery_name contains {} elements."
                ).format(nb_fishery, len(list_fishery_name)))

        f_param = {}
        for f in list_fishery_name :
            tmp_dict = {
                "function_type":int(root.find("s_sp_fishery").find(f).find(
                    "function_type").attrib["value"]),
                "q":float(root.find("q_sp_fishery").find(f).attrib[species_name]),
                "variable":float(root.find(
                    "s_sp_fishery").find(f).attrib[species_name]),
                "length_threshold":float(root.find('s_sp_fishery').find(f).find(
                    "length_threshold").attrib[species_name])}

            if tmp_dict['function_type'] == 3 :
                tmp_dict['right_asymptote'] = float(
                    root.find("s_sp_fishery").find(f).find(
                        "right_asymptote").attrib[species_name])
            f_param[f] = tmp_dict

        return f_param

    def readMortalityXML(
            self, xml_config_file: str, species_name: str = None
            ) -> dict :
        """Read a XML file to get all parameters needed for mortality
        field production."""

        tree = ET.parse(xml_config_file)
        root = tree.getroot()
        if species_name == None :
            species_name = root.find("sp_name").text

        n_param = {'MPmax': float(root.find(
                                  'Mp_mean_max').attrib[species_name]),
                   'MPexp': float(root.find(
                                  'Mp_mean_exp').attrib[species_name]),
                   'MSmax': float(root.find(
                                  'Ms_mean_max').attrib[species_name]),
                   'MSslope': float(root.find(
                                    'Ms_mean_max').attrib[species_name]),
                   'Mrange': float(root.find(
                                    'Ms_mean_max').attrib[species_name])}

        return n_param

    def vMax(self, length : float) -> float :
        """Return the maximum velocity of a fish with a given length."""

        return (self.ikamoana_fields_structure.vmax_a
             * np.power(length, self.ikamoana_fields_structure.vmax_b))

    def landmask(
            self, habitat_field : xr.DataArray = None,
            use_SEAPODYM_global_mask: bool = False, shallow_sea_to_ocean=False,
            lim=1e-45, lat_min: int = None, lat_max: int = None,
            lon_min: int = None, lon_max: int = None, field_output: bool = False
            ) -> xr.DataArray :
        """Return the landmask of a given habitat (`habitat_field`) or
        generated from the FeedingHabitat.global_mask which is used by
        SEAPODYM (`use_SEAPODYM_global_mask: bool = True`).

        Mask values :
        - 2 -> is Shallow
        - 1 -> is Land or No_Data
        - 0 -> deep ocean with habitat data

        If field_output is True, time coordinate is added to landmask.

        Note
        ----
        Landmask in Original (with Parcels Fields) is flipped on latitude axis.
        """

        def controlArguments(habitat_field, lat_min, lat_max, lon_min, lon_max) :
            if habitat_field is not None:
                if lat_min is None :
                    lat_min = habitat_field.attrs['lat_min']
                if lat_max is None :
                    lat_max = habitat_field.attrs['lat_max']
                if lon_min is None :
                    lon_min = habitat_field.attrs['lon_min']
                if lon_max is None :
                    lon_max = habitat_field.attrs['lon_max']
            if lat_max is not None :
                lat_max += 1
            if lon_max is not None :
                lon_max += 1

            return lat_min, lat_max, lon_min, lon_max

        lat_min, lat_max, lon_min, lon_max = controlArguments(
            habitat_field, lat_min, lat_max, lon_min, lon_max)

        if use_SEAPODYM_global_mask :
            mask_L1 = np.invert(
                self.feeding_habitat_structure.data_structure.global_mask[
                    'mask_L1'])[0, lat_min:lat_max, lon_min:lon_max]
            mask_L3 = np.invert(
                self.feeding_habitat_structure.data_structure.global_mask[
                    'mask_L3'])[0, lat_min:lat_max, lon_min:lon_max]

            landmask = np.zeros(mask_L1.shape, dtype=np.int8)
            if not shallow_sea_to_ocean :
                landmask[mask_L3] = 2
            landmask[mask_L1] = 1

            coords = self.feeding_habitat_structure.data_structure.coords
            coords = {'lat':coords['lat'][lat_min:lat_max],
                      'lon':coords['lon'][lon_min:lon_max]}

        else :
            if habitat_field is None :
                raise ValueError("You must specify a habitat_field argument if"
                                 " use_SEAPODYM_global_mask is False.")
            habitat_f = habitat_field[0,:,:]
            lmeso_f = self.feeding_habitat_structure.data_structure.variables_dictionary[
                'forage_lmeso'][0, lat_min:lat_max, lon_min:lon_max]

            if habitat_f.shape != lmeso_f.shape :
                raise ValueError("Habitat {} and forage_lmeso {} must have the"
                                 " same dimension.".format(habitat_f.shape,
                                                           lmeso_f.shape))

            landmask = np.zeros_like(habitat_f)
            if not shallow_sea_to_ocean :
                landmask[(np.abs(lmeso_f) <= lim) | np.isnan(lmeso_f)] = 2
            landmask[(np.abs(habitat_f) <= lim) | np.isnan(habitat_f)] = 1

            coords = {'lat':habitat_field.coords['lat'],
                      'lon':habitat_field.coords['lon']}

        ## TODO : Why is lon between 1 and ny-1 ?
        landmask[-1,:] = landmask[0,:] = 0

        if field_output :
            if habitat_field is None :
                raise ValueError("If field_output is True you must passe a "
                                 "habitat_field. Otherwise the time coordinate"
                                 " length can't be calculated.")
            landmask = np.tile(landmask[np.newaxis],
                               (habitat_field.time.size, 1, 1))
            coords['time'] = habitat_field.time
            dimensions = ('time', 'lat', 'lon')
        else :
            dimensions = ('lat', 'lon')

        return xr.DataArray(data=landmask, name='landmask', coords=coords,
                            dims=dimensions)

    def gradient(
            self, field: xr.DataArray, landmask: xr.DataArray, name: str = None
            ) -> Tuple[xr.DataArray]:
        """
        Gradient calculation for a Xarray DataArray seapodym-equivalent calculation
        requires LandMask forward and backward differencing for domain edges
        and land/shallow sea cells."""

        if ((field.lat.size != landmask.lat.size)
                or (field.lon.size != landmask.lon.size)) :
            raise ValueError("Field and landmask must have the same dimension.")

        ## WARNING : To have the same behavior as original gradient function,
        # latitude must be south-north rather than north-south.

        flip_lat = field.lat[0] > field.lat[-1]
        if flip_lat :
            field = field.reindex(lat=field.lat[::-1])
        if landmask.lat[0] > landmask.lat[-1] :
            landmask = landmask.reindex(lat=landmask.lat[::-1])

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

        if flip_lat : field = field = field.reindex(lat=field.lat[::-1])

        return (xr.DataArray(
                    name = "Gradient_longitude_"+(field.name if name is None else name),
                    data = dVdlon,
                    coords = field.coords,
                    dims=('time','lat','lon'),
                    attrs=field.attrs),
                xr.DataArray(
                    name = "Gradient_latitude_"+(field.name if name is None else name),
                    data = np.flip(dVdlat, axis=1) if flip_lat else dVdlat,
                    coords = field.coords,
                    dims=('time','lat','lon'),
                    attrs=field.attrs))

## TODO plus tard : Take into account L1 is a simplification.
# Should use accessibility + forage distribution + current L1/L2/L3
    def current_forcing(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Load current forcing for NetCDF or Dym files.

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray]
            U, V
        """
        
        U = fhcf.seapodymFieldConstructor(
            self.feeding_habitat_structure.data_structure.root_directory
            + self.ikamoana_fields_structure.u_file,  dym_varname='u_L1')
        V = fhcf.seapodymFieldConstructor(
            self.feeding_habitat_structure.data_structure.root_directory
            + self.ikamoana_fields_structure.v_file,  dym_varname='v_L1')
        
        U = latitudeDirection(U, south_to_north=True)
        V = latitudeDirection(V, south_to_north=True)

        if self.feeding_habitat is not None:
            # NOTE : We assume that U and V have same coordinates.
            timefun, latfun, lonfun = coordsAccess(U)
            minlon_idx = lonfun(min(self.feeding_habitat.coords['lon'].data))
            maxlon_idx = lonfun(max(self.feeding_habitat.coords['lon'].data))
            minlat_idx = latfun(min(self.feeding_habitat.coords['lat'].data))
            maxlat_idx = latfun(max(self.feeding_habitat.coords['lat'].data))
            mintime_idx = timefun(min(self.feeding_habitat.coords['time'].data))
            maxtime_idx = timefun(max(self.feeding_habitat.coords['time'].data))
            U = U[mintime_idx:maxtime_idx+1,
                  minlat_idx:maxlat_idx+1,
                  minlon_idx:maxlon_idx+1]
            V = V[mintime_idx:maxtime_idx+1,
                  minlat_idx:maxlat_idx+1,
                  minlon_idx:maxlon_idx+1]
        return U, V

    def start_distribution(self, filepath: str) -> xr.DataArray :
        """Description"""
        
        dist = fhcf.seapodymFieldConstructor(filepath, dym_varname='start')
        #dist = xr.apply_ufunc(np.nan_to_num,dist)
        
        # Clip dimensions to the same as the feeding habitats, but only
        dist = latitudeDirection(dist, south_to_north=True)
        if self.feeding_habitat is not None :
            _, latfun, lonfun  = coordsAccess(dist)
            minlon_idx = lonfun(min(self.feeding_habitat.coords['lon'].data))
            maxlon_idx = lonfun(max(self.feeding_habitat.coords['lon'].data))
            minlat_idx = latfun(min(self.feeding_habitat.coords['lat'].data))
            maxlat_idx = latfun(max(self.feeding_habitat.coords['lat'].data))
            return dist[:2, minlat_idx:maxlat_idx+1, minlon_idx:maxlon_idx+1]
        else :
            return dist[:2]

    def taxis(
            self, dHdlon: xr.DataArray, dHdlat: xr.DataArray, name: str = None
            ) -> Tuple[xr.DataArray,xr.DataArray] :
        """Calculation of the Taxis field from the gradient."""

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

        return (xr.DataArray(name = "Taxis_longitude_"+(dHdlon.name if name is None else name),
                             data = Tlon,
                             coords = dHdlon.coords,
                             dims=('time','lat','lon'),
                             attrs=dHdlon.attrs),
                xr.DataArray(name = "Taxis_latitude_"+(dHdlat.name if name is None else name),
                             data = Tlat,
                             coords = dHdlat.coords,
                             dims=('time','lat','lon'),
                             attrs=dHdlat.attrs))

    def diffusion(
            self, habitat: xr.DataArray, name: str = None
            ) -> xr.DataArray :
        """This is simply calculating the required indices of the
        forcing for this simulation.
        
        See Also
        --------
        Seapodym user manual page 32 : Active random movement
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

        is_evolving, age = argumentCheck(habitat)
        timestep = self.ikamoana_fields_structure.timestep

        end = habitat.time.size
        ## TODO : How do we manage NaN values ?
        # Hdata = habitat.data
        Hdata = np.nan_to_num(habitat.data)
        K = np.zeros_like(Hdata)
        f_length = self.feeding_habitat_structure.data_structure.findLengthByCohort

        for t in range(end):

            t_age = age[t] if is_evolving else age
            t_length = f_length(t_age) / 100 # Convert into meter

            if self.ikamoana_fields_structure.units == 'nm_per_timestep':
                Dmax = (t_length*(timestep/1852))**2 / 4
            else:
                Dmax = (t_length**2 / 4) * timestep
            sig_D = self.ikamoana_fields_structure.sigma_K * Dmax

            ## VECTORIZED
            K[t,:,:] = (
                self.ikamoana_fields_structure.sig_scale
                * sig_D
                * (1 - self.ikamoana_fields_structure.c_scale
                    * self.ikamoana_fields_structure.c
                    * np.power(Hdata[t,:,:], self.ikamoana_fields_structure.P))
                * self.ikamoana_fields_structure.diffusion_scale
                + self.ikamoana_fields_structure.diffusion_boost
            )


        return xr.DataArray(data=K,
                            name="K_" + (habitat.name if name is None else name),
                            coords=habitat.coords,
                            dims=("time","lat","lon"),
                            attrs=habitat.attrs)

# TODO : finish DocString
    def fishingMortality(
            self, effort_ds: xr.Dataset, fisheries_parameters: dict,
            start_age: int = 0, evolving: bool = True,
            convertion_tab: Dict[str, Union[str,int,float]] = None,
            ) -> xr.DataArray :
        """Convert effort by fishery to fishing mortality by applying
        a selectivity function which can be :
        
        - Limit one (not supported yet)
        - Sigmoid
        - Asymmetric Gaussian

        Parameters
        ----------
        effort_ds : xr.Dataset
            [description]
        fisheries_parameters : dict
            [description]
        start_age : int, optional
            [description]
        evolving : bool, optional
            [description]
        convertion_tab : Dict[str, Union[str,int,float]], optional
            [description]

        Returns
        -------
        xr.DataArray
            [description]
        """

        # # This is not necessary :
        # if len(effort_ds) != len(fisheries_parameters.keys()) :
        #     raise ValueError((
        #         "effort_ds ({}) and fisheries_parameters ({}) must have "
        #         "same length.").format(
        #             len(effort_ds),len(fisheries_parameters.keys())))

        # TYPE I - Not Supported
        def selectivityLimitOne(length, sigma):
            warnings.warn("Selectivity Function not supported! q set to 0")
            #return length / (sigma+length)
            return 0

        # TYPE II
        def selectivitySigmoid(length, sigma, mu) :
            return (1 + np.exp(-sigma*(length-mu)))**(-1)

        # TYPE III
        def selectivityAsymmetricGaussian(length, mu, r_asymp, sigma_sq) :
            if length > mu:
                return np.exp(-((length-mu)**2/sigma_sq)) * (1-r_asymp) + r_asymp
            else:
                return np.exp(-((length-mu)**2/sigma_sq))

        # Functions Generator
        def selectivity(function_type, sigma, mu, r_asymp=None) :
            if function_type == 3 :
                return lambda length : selectivityAsymmetricGaussian(
                    length, mu=mu, r_asymp=r_asymp, sigma_sq=sigma**2
                )
            elif function_type == 2 :
                return lambda length : selectivitySigmoid(
                    length, sigma=sigma, mu=mu
                )
            else :
                return lambda length : selectivityLimitOne(
                    length, sigma=sigma
                )

        ## NOTE : Original code
        # E_scaler = (1.0/30.0)*7.0
        # F_scaler = 30.0 / 7.0 / 7.0

        length_fun = self.feeding_habitat_structure.data_structure.findLengthByCohort

        fishing_mortality = {}
        for p_name, params in fisheries_parameters.items() :
            f_name = convertion_tab[p_name] if p_name in convertion_tab else p_name
            if f_name in effort_ds :
                data = effort_ds[f_name].data
                f_data = np.empty_like(data)

                q = params['q']
                selectivity_fun = selectivity(
                    function_type=params['function_type'],
                    sigma=params['variable'],
                    mu=params['length_threshold'],
                    r_asymp=(params['right_asymptote']
                             if 'right_asymptote' in params else None))

                if evolving :
                    c_nb = self.feeding_habitat_structure.data_structure.cohorts_number
                    tmp = np.arange(start_age, c_nb)
                    age = np.concatenate(
                        (tmp,np.repeat(c_nb-1, effort_ds.time.data.size - tmp.size)))
                else : age = start_age

                for t in range(effort_ds.time.data.size) :
                    # length in cm
                    length = length_fun(age[t]) if evolving else length_fun(age)
                    f_data[t,:,:] = data[t,:,:] * q * selectivity_fun(length)

                fishing_mortality[f_name] = xr.DataArray(
                    f_data,
                    coords=effort_ds[f_name].coords,
                    attrs=effort_ds[f_name].attrs)

        fishing_mortality_ds = xr.Dataset(fishing_mortality)
        fishing_mortality_ds.attrs.update(effort_ds.attrs)
        fishing_mortality_ds.attrs["Fisheries"] = list(fishing_mortality.keys())

        return fisherieseffort.sumDataSet(fishing_mortality_ds,
                                          name="Mortality")

# ------------------------------ WRAPPER ----------------------------- #

    def _commonWrapperTaxis(
            self, feeding_habitat, name, lat_min, lat_max, lon_min, lon_max):

        hf_cond, ssto_cond = (self.ikamoana_fields_structure.landmask_from_habitat,
                              self.ikamoana_fields_structure.shallow_sea_to_ocean)
        param = dict(
            habitat_field=feeding_habitat,
            use_SEAPODYM_global_mask=(not hf_cond), shallow_sea_to_ocean=ssto_cond,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

        landmask = self.landmask(**param)

        grad_lon, grad_lat = self.gradient(feeding_habitat, landmask)

        return self.taxis(grad_lon, grad_lat,
                          name=feeding_habitat.name if name is None else name)

# TODO : Finish the description
    def computeTaxis(
            self, cohort: int = None,
            time_start: int = None, time_end: int = None,
            lat_min: int = None, lat_max: int = None,
            lon_min: int = None, lon_max: int = None,
            name: str = None, use_already_computed_habitat: bool = False,
            verbose: bool = False
            ) -> Tuple[xr.DataArray,xr.DataArray] :
        """
        Calculates the taxis field of a given habitat. If the feeding
        habitat is not already calculated, it also calculates the
        feeding habitat using the FeedingHabitat class.

        Parameters
        ----------
        cohort : int, optional if `use_already_computed_habitat` is True
            The cohort whose habitat is to be calculated.
        time_start : int, optional
            [description]
        time_end : int, optional
            [description]
        lat_min : int, optional
            [description]
        lat_max : int, optional
            [description]
        lon_min : int, optional
            [description]
        lon_max : int, optional
            [description]
        name : str, optional
            Will name the DataArray with.
        use_already_computed_habitat : bool, optional
            If True, the feeding habitat will be calculated anyway.
            Otherwise, if `self.feeding_habitat` is not None, self
            habitat will be used for taxis calculation.
        verbose : bool, optional

        See Also
        --------
        FeedingHabitat.computeFeedingHabitat : computeTaxis is based on
            the feeding habitat.

        Returns
        -------
        Tuple[xr.DataArray,xr.DataArray]
            The first one is the Taxis_longitude and the second is the
            Taxis_latitude DataArray.

        Raises
        ------
        ValueError
            An error is raised if the feeding_habitat must be calculated
            but the `cohort` argument is None.
        """

        (time_start,time_end,lat_min,lat_max,lon_min,lon_max) = (
            self.feeding_habitat_structure.controlArguments(
                time_start, time_end, lat_min, lat_max, lon_min, lon_max))

        if (self.feeding_habitat is None) or (not use_already_computed_habitat) :
            if cohort is None :
                raise ValueError("Cohort argument must be specified. "
                                 "Actual is %s."%(str(cohort)))
            feeding_habitat = (
                self.feeding_habitat_structure.computeFeedingHabitat(
                    cohort, time_start, time_end, lat_min, lat_max, lon_min,
                    lon_max, False, verbose))
            fh_name = list(feeding_habitat.var()).pop()
            feeding_habitat_da = feeding_habitat[fh_name]
            feeding_habitat_da.attrs.update(feeding_habitat.attrs)
            self.feeding_habitat = feeding_habitat_da
        else :
            feeding_habitat_da = self.feeding_habitat

        return self._commonWrapperTaxis(feeding_habitat_da, name, lat_min,
                                        lat_max, lon_min, lon_max)

# TODO : Finish the description
    def computeEvolvingTaxis(
            self, cohort_start: int = None, cohort_end: int = None,
            time_start: int = None, time_end: int = None,
            lat_min: int = None, lat_max: int = None,
            lon_min: int = None, lon_max: int = None,
            name: str = None, use_already_computed_habitat: bool = False,
            verbose: bool = False
            ) -> Tuple[xr.DataArray,xr.DataArray] :
        """
        Calculates the taxis field of a given evolving habitat. If the
        evolving feeding habitat is not already calculated, it also
        calculates the feeding habitat using the FeedingHabitat class.

        Parameters
        ----------
        cohort_start : int, optional
            The age of the first cohort for which we will calculate the
            habitat. If None, it corresponds to the youngest cohort (0).
        cohort_end : int, optional
            The age of the last cohort for which we will calculate the
            habitat. If None, it corresponds to the oldest cohort.
        time_start : int, optional
            [description]
        time_end : int, optional
            [description]
        lat_min : int, optional
            [description]
        lat_max : int, optional
            [description]
        lon_min : int, optional
            [description]
        lon_max : int, optional
            [description]
        name : str, optional
            Will name the DataArray with.
        use_already_computed_habitat : bool, optional
            If True, the feeding habitat will be calculated anyway.
            Otherwise, if `self.feeding_habitat` is not None, self
            habitat will be used for taxis calculation.
        verbose : bool, optional

        See Also
        --------
        FeedingHabitat.computeFeedingHabitat : computeEvolvingTaxis is
            based on the evolving feeding habitat.

        Returns
        -------
        Tuple[xr.DataArray,xr.DataArray]
            The first one is the Taxis_longitude and the second is the
            Taxis_latitude DataArray.
        """

        (time_start,time_end,lat_min,lat_max,lon_min,lon_max) = (
            self.feeding_habitat_structure.controlArguments(
                time_start, time_end, lat_min, lat_max, lon_min, lon_max))

        if (self.feeding_habitat is None) or (not use_already_computed_habitat):
            feeding_habitat = (
                self.feeding_habitat_structure.computeEvolvingFeedingHabitat(
                    cohort_start, cohort_end, time_start, time_end, lat_min,
                    lat_max, lon_min, lon_max, False, verbose))
            self.feeding_habitat = feeding_habitat
        else :
            feeding_habitat = self.feeding_habitat

        return self._commonWrapperTaxis(feeding_habitat, name, lat_min,
                                        lat_max, lon_min, lon_max)

# TODO : Write the description
    def computeMortality(
            self, effort_filepath: str, fisheries_xml_filepath: str,
            time_reso: int, space_reso: float, skiprows: int = 0,
            removeNoCatch: bool = False, predict_effort: bool = False,
            remove_fisheries: List[Union[float,str,int]] = None,
            convertion_tab: Dict[str, Union[str,int,float]] = None,
            verbose: bool = False
            ) -> xr.DataArray :

        params_fisheries = self.readFisheriesXML(fisheries_xml_filepath)

        to_remove = []
        for f in remove_fisheries :
            to_remove.append(
                convertion_tab[f] if f in convertion_tab.keys() else f)

        effort_ds = fisherieseffort.effortByFishery(
            effort_filepath, time_reso=time_reso, space_reso=space_reso,
            skiprows=skiprows, removeNoCatch=removeNoCatch,
            remove_fisheries=to_remove, predict_effort=predict_effort,
            verbose=verbose)

        return self.fishingMortality(effort_ds, params_fisheries,
                                     convertion_tab=convertion_tab)

# ------------------------------- MAIN ------------------------------- #

# TODO : Mortality isn't calculated for now. Uncomment to do so.
    def computeIkamoanaFields(
            self, effort_filepath: str, fisheries_xml_filepath: str,
            time_reso: int, space_reso: float, skiprows: int = 0,
            removeNoCatch: bool = False, predict_effort: bool = False,
            remove_fisheries: List[Union[float,str,int]] = None,
            convertion_tab: Dict[str, Union[str,int,float]] = None,

            from_habitat: xr.DataArray = None,
            evolve: bool = True, cohort_start: int = None,
            cohort_end: int = None, time_start: int = None, time_end: int = None,
            lat_min: int = None, lat_max: int = None, lon_min: int = None,
            lon_max: int = None, verbose: bool = False,
  
            south_to_north: bool = True
            ) -> Dict[str, xr.DataArray]:
        """
        Feeding Habitat is calculated everytime, see WARNING commentary.
        """

        #self.feeding_habitat_structure.data_structure.normalizeCoords()
        hf_cond, ssto_cond = (
            self.ikamoana_fields_structure.landmask_from_habitat,
            self.ikamoana_fields_structure.shallow_sea_to_ocean)

        # TODO : vérifier qu'il n'y ai pas de manière plus propre de le
        # faire. Par exemple en chargeant le ffeding_habitat lors de
        # l'initialisation.
        if from_habitat is not None:
            use_already_computed_habitat = True
            self.feeding_habitat=from_habitat
        else:
            use_already_computed_habitat = False
            
        if evolve :
            taxis_lon, taxis_lat = self.computeEvolvingTaxis(
                cohort_start=cohort_start, cohort_end=cohort_end, time_start=time_start,
                time_end=time_end, lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max, verbose=verbose,
                use_already_computed_habitat=use_already_computed_habitat)
        else :
            taxis_lon, taxis_lat = self.computeTaxis(
                cohort=cohort_start, time_start=time_start, time_end=time_end,
                lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                use_already_computed_habitat=use_already_computed_habitat,
                verbose=verbose)

        landmask = self.landmask(
            habitat_field=self.feeding_habitat, use_SEAPODYM_global_mask=not(hf_cond),
            shallow_sea_to_ocean=ssto_cond, lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max, field_output=True)

        diffusion = self.diffusion(self.feeding_habitat)
        gradient_diffusion_lon, gradient_diffusion_lat = self.gradient(
            diffusion, landmask.loc[landmask.time.data[0],:,:])

        U, V = self.current_forcing()
        # start = self.start_distribution()

        # mortality = self.computeMortality(
        #     effort_filepath=effort_filepath, fisheries_xml_filepath=fisheries_xml_filepath,
        #     time_reso=time_reso, space_reso=space_reso, skiprows=skiprows,
        #     removeNoCatch=removeNoCatch, predict_effort=predict_effort,
        #     remove_fisheries=remove_fisheries, convertion_tab=convertion_tab,
        #     verbose=verbose)
        
        # TODO : add the feeding habitat
        return {'Tx':latitudeDirection(
            taxis_lon,south_to_north).drop_vars('cohorts'),
                'Ty':latitudeDirection(
                    taxis_lat,south_to_north).drop_vars('cohorts'),
                'K':latitudeDirection(
                    diffusion,south_to_north).drop_vars('cohorts'),
                'dK_dx':latitudeDirection(
                    gradient_diffusion_lon,south_to_north).drop_vars('cohorts'),
                'dK_dy':latitudeDirection(
                    gradient_diffusion_lat,south_to_north).drop_vars('cohorts'),
                'U':latitudeDirection(U,south_to_north),
                'V':latitudeDirection(V,south_to_north),
                'landmask':latitudeDirection(landmask,south_to_north),
                'H':latitudeDirection(self.feeding_habitat,south_to_north)
                #'mortality':latitudeDirection(mortality,south_to_north)
        }
