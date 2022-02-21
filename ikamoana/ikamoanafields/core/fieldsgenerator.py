import warnings
from typing import Dict, Tuple, Union

import numpy as np
import parcels
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar

from ...feedinghabitat.habitatdatastructure import HabitatDataStructure
from ...fisherieseffort import fisherieseffort
from ..core import IkamoanaFieldsDataStructure
from ...utils import latitudeDirection


def landmask(
        data_structure: HabitatDataStructure,
        habitat_field : xr.DataArray = None,
        use_SEAPODYM_global_mask: bool = False, shallow_sea_to_ocean: bool = False,
        lim: float = 1e-45, lat_min: int = None, lat_max: int = None,
        lon_min: int = None, lon_max: int = None, field_output: bool = False
        ) -> xr.DataArray :
    """Return the landmask of a given habitat (`habitat_field`) based on
    the lower mesopelagic forage field or generated from the
    FeedingHabitat.global_mask which is used by SEAPODYM
    (`use_SEAPODYM_global_mask: bool = True`).

    Mask values are :
    - 2 -> Shallow
    - 1 -> Land or no data
    - 0 -> deep ocean with habitat data

    Parameters
    ----------
    data_structure : HabitatDataStructure
        The structure that contains : globale_mask, coords and
        forage_lmeso.
    habitat_field : xr.DataArray, optional
        Habitat used to compute the landmask. Unnecessary if
        `use_SEAPODYM_global_mask` is True.
    use_SEAPODYM_global_mask : bool, optional
        Indicate whether you want to create a mask using the habitat
        (False) or the SEAPODYM mask (True) which is text format.
    shallow_sea_to_ocean : bool, optional
        Consider the shallow sea as ocean (True) or land (False).
    lim : float, optional
        If the `habitat_field` value is under this limit, it is
        considered as Land or No_Data.
    field_output : bool, optional
        If True, landmask time coordinate is expanded to follow the
        habitat time axis.

    Note
    ----
    Landmask in Original (with Parcels Fields) is flipped on latitude axis.

    Returns
    -------
    xr.DataArray
        Landmask as DataArray which coords are [latitude,longitude] or
        [time,latitude,longitude] according to `field_output`.
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
        ## NOTE : SEAPODYM mask contains {0,1,2,3} which are the layers 
        # number and where 0 is corresponding to land. This mask is
        # converted to boolean masks. False (0) is land and True (1) is
        # ocean.
        mask_L1 = data_structure.global_mask[
            'mask_L1'][0, lat_min:lat_max, lon_min:lon_max]
        mask_L3 = data_structure.global_mask[
            'mask_L3'][0, lat_min:lat_max, lon_min:lon_max]

        landmask = np.zeros(mask_L1.shape, dtype=np.int8)
        if not shallow_sea_to_ocean :
            landmask = np.where(mask_L3 == 0, 2, landmask)
        landmask = np.where(mask_L1 == 0, 1, landmask)
        coords = {'lat':data_structure.coords['lat'][lat_min:lat_max],
                  'lon':data_structure.coords['lon'][lon_min:lon_max]}

    else :
        if habitat_field is None :
            raise ValueError("You must specify a habitat_field argument if"
                             " use_SEAPODYM_global_mask is False.")
        habitat_f = habitat_field[0,:,:]
        lmeso_f = data_structure.variables_dictionary[
            'forage_lmeso'][0, lat_min:lat_max, lon_min:lon_max]

        if habitat_f.shape != lmeso_f.shape :
            raise ValueError("Habitat {} and forage_lmeso {} must have the"
                             " same dimension.".format(habitat_f.shape,
                                                       lmeso_f.shape))

        landmask = np.zeros_like(habitat_f)
        if not shallow_sea_to_ocean :
            condition_shallow = (np.abs(lmeso_f) <= lim) | np.isnan(lmeso_f)
            landmask = np.where(condition_shallow, 2, landmask)
        condition_land = (np.abs(habitat_f) <= lim) | np.isnan(habitat_f)
        landmask = np.where(condition_land, 1, landmask)

        coords = {'lat':habitat_field.coords['lat'],
                  'lon':habitat_field.coords['lon']}

    ## TODO : Why is lon between 1 and ny-1 ?
    landmask[-1,:] = landmask[0,:] = 0

    if field_output :
        if habitat_field is None :
            raise ValueError("If field_output is True you must passe a "
                             "habitat_field. Otherwise the time coordinate "
                             "length can't be calculated.")
        landmask = np.tile(landmask[np.newaxis],
                            (habitat_field.time.size, 1, 1))
        coords['time'] = habitat_field.time
        dimensions = ('time', 'lat', 'lon')
    else :
        dimensions = ('lat', 'lon')

    landmask = latitudeDirection(
        xr.DataArray(data=landmask, name='landmask', coords=coords, dims=dimensions),
        south_to_north=True)
    return landmask

def _getCellEdgeSizes(field) :
    """Calculate the size (in kilometers) of each cells of a grid
    defined by latitude and longitudes coordinates. Copy of the
    `Parcels.Field.calc_cell_edge_sizes` function in Parcels. Avoid the
    convertion of DataArray into `Parcels.Field`.

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

def gradient(
        field: xr.DataArray, landmask: xr.DataArray, name: str = None
        ) -> Tuple[xr.DataArray,xr.DataArray]:
    """Compute the gradient of a Xarray.DataArray (equivalent to the
    SEAPODYM method). Requires LandMask forward and backward
    differencing for domain edges and land/shallow sea cells.

    Parameters
    ----------
    field : xr.DataArray
        The field whose gradient you want to calculate.
    landmask : xr.DataArray
        A landmask who contains sea (0), land (1) and shallow sea(2).
        Refer to `landmask`.
    name : str, optional
        Specify the name of the returned DataArrays.
        Syntax is : Gradient_(longitude/latitude)_`name`.

    See Also
    --------
    landmask

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        Longitude gradient first, then latitude gradient.

    Raises
    ------
    ValueError
        Field and landmask must have the same dimension.
    """

    if ((field.lat.size != landmask.lat.size)
            or (field.lon.size != landmask.lon.size)) :
        raise ValueError("Field and landmask must have the same dimension.")

    ## WARNING : To have the same behavior as original gradient function,
    # latitude must be south-north rather than north-south.

    field = latitudeDirection(field, south_to_north=True)
    landmask = latitudeDirection(landmask, south_to_north=True)

    dlon, dlat = _getCellEdgeSizes(field)

    nlat = field.lat.size
    nlon = field.lon.size

    data = np.nan_to_num(field.data)
    landmask = landmask.data
    dVdlon = np.zeros(data.shape, dtype=np.float32)
    dVdlat = np.zeros(data.shape, dtype=np.float32)

    ## NOTE : Parallelised execution may help to do it faster.
    # - I think it can also be vectorized.
    # - Uses of Numba prange() over time axis ?
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

    if field.name is None :
        field.name = 'Unnamed_DataArray'
    
    return (
        xr.DataArray(
            name="Gradient_longitude_"+(field.name if name is None else name),
            data=dVdlon, coords=field.coords, dims=('time','lat','lon'),
            attrs=field.attrs),
        xr.DataArray(
            name="Gradient_latitude_"+(field.name if name is None else name),
            data=dVdlat, coords=field.coords, dims=('time','lat','lon'),
            attrs=field.attrs))
    
def taxis(
        ika_structure: IkamoanaFieldsDataStructure,
        fh_structure: HabitatDataStructure, 
        dHdlon: xr.DataArray, dHdlat: xr.DataArray, name: str = None
        ) -> Tuple[xr.DataArray,xr.DataArray] :
    """Calculation of the Taxis field from the habitat gradient.

    Parameters
    ----------
    ika_structure : IkamoanaFieldsDataStructure
        Ikamoana data structure that contains parameters.
    fh_structure : HabitatDataStructure
        SEAPODYM data structure that contains fields and parameters
    dHdlon : xr.DataArray
        The habitat gradient along the longitude axis.
    dHdlat : xr.DataArray
        The habitat gradient along the latitude axis.
    name : str, optional
        Specify the name of the returned DataArrays.
        Syntax is : Taxis_(longitude/latitude)_`name`.

    Returns
    -------
    Tuple[xr.DataArray,xr.DataArray]
        Longitude taxis first, then latitude taxis.
    """

    def vMax(length : float) -> float :
        """Return the maximum velocity of a fish with a given length."""

        return (ika_structure.vmax_a * np.power(length, ika_structure.vmax_b))

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
    latitude_correction = np.tile(dHdlon.lat.data, (dHdlon.lon.size, 1)).T
    latitude_correction = np.cos(latitude_correction * np.pi/180)
    # factor = ika_structure.taxis_scale * 250 * 1.852 * 15
    factor = ika_structure.taxis_scale * 1852 * 120
    f_length = fh_structure.findLengthByCohort

    for t in range(dHdlon.time.size):
        t_age = age[t] if is_evolving else age
        # Convert cm to meter (/100) : See original function
        t_length = f_length(t_age) / 100

        Tlon[t,:,:] = (vMax(t_length) * dHdlon.data[t,:,:] * factor
                       * latitude_correction)
        Tlat[t,:,:] = (vMax(t_length) * dHdlat.data[t,:,:] * factor)

    if ika_structure.units == 'nm_per_timestep':
        Tlon *= (16/1852)
        Tlat *= (16/1852)
    ## NOTE :       (timestep/1852) * (1000*1.852*60) * 1/timestep
    #           <=> (250*1.852*15) * (16/1852)

    return (
        xr.DataArray(
            name="Taxis_longitude_"+(dHdlon.name if name is None else name),
            data=Tlon, coords=dHdlon.coords, dims=('time','lat','lon'),
            attrs=dHdlon.attrs),
        xr.DataArray(
            name="Taxis_latitude_"+(dHdlat.name if name is None else name),
            data=Tlat, coords=dHdlat.coords, dims=('time','lat','lon'),
            attrs=dHdlat.attrs))
    
def diffusion(
        ika_structure: IkamoanaFieldsDataStructure,
        fh_structure: HabitatDataStructure, habitat: xr.DataArray,
        landmask: xr.DataArray = None, name: str = None,
        ) -> xr.DataArray :
    """Computes the diffusion field (K) based on the feeding `habitat`.
    See the SEAPODYM User's Manual, page 32, Active Random Movement.
    
    See also SEAPODYM code :
    `Calpop_recompute_coefs.cpp` line 272, function `Recomp_DEF_UV_coef`

    Parameters
    ----------
    ika_structure : IkamoanaFieldsDataStructure
        Ikamoana data structure that contains parameters.
    fh_structure : HabitatDataStructure
        SEAPODYM data structure that contains fields and parameters
    habitat : xr.DataArray
        The feeding habitat used to compute the diffusion field.
    name : str, optional
        Specify the name of the returned DataArrays.
        Syntax is : K_`name`.

    Returns
    -------
    xr.DataArray
        The diffusion field.
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

    habitat = latitudeDirection(habitat, south_to_north=True)
    is_evolving, age = argumentCheck(habitat)
    timestep = ika_structure.timestep

    end = habitat.time.size
    ## TODO : How do we manage NaN values ?
    # Hdata = habitat.data
    Hdata = np.nan_to_num(habitat.data)
    K = np.zeros_like(Hdata)
    f_length = fh_structure.findLengthByCohort

    for t in range(end):

        t_age = age[t] if is_evolving else age
        t_length = f_length(t_age) / 100 # Convert into meter

        if ika_structure.units == 'nm_per_timestep':
            Dmax = (t_length*timestep/1852)**2 / 4
        elif ika_structure.units == 'm_per_s':
            Dmax = (t_length**2 * timestep) / 4
        else :
            raise ValueError(("Ikamoana units must be either 'nm_per_timestep' "
                             "or 'm_per_s'. Not {}").format(ika_structure.units))
        sig_D = ika_structure.sigma_K * Dmax

        ## VECTORIZED
        K[t,:,:] = (
            ika_structure.sig_scale
            * sig_D
            * (1 - ika_structure.c_scale * ika_structure.c
               * np.power(Hdata[t,:,:], ika_structure.P))
            * ika_structure.diffusion_scale + ika_structure.diffusion_boost
        )


    if landmask is not None :
        landmask = latitudeDirection(landmask, south_to_north=True)
        K = np.where(landmask != 1, K, np.NaN)

    if habitat.name is None :
        habitat.name = 'Unnamed_DataArray'

    return xr.DataArray(
        data=K, name="K_"+(habitat.name if name is None else name),
        coords=habitat.coords, dims=("time","lat","lon"), attrs=habitat.attrs)
    
def fishingMortality(
        fh_structure: HabitatDataStructure, effort_ds: xr.Dataset,
        fisheries_parameters: dict, start_age: int = 0, evolving: bool = True,
        convertion_tab: Dict[str, Union[str,int,float]] = None,
        ) -> xr.DataArray :
    """Convert effort by fishery to fishing mortality by applying
    a selectivity function which can be :
    
    - Limit one (not supported yet)
    - Sigmoid
    - Asymmetric Gaussian

    Parameters
    ----------
    fh_structure : HabitatDataStructure
        SEAPODYM data structure that contains parameters.
    effort_ds : xr.Dataset
        A Dataset that contains DataArray for each fishery, representing
        fishing effort across time. See the 
    fisheries_parameters : dict
        Selectivity function parameters for each fishery. The keys are
        the names of the fisheries in the SEAPODYM configuration file.
    start_age : int, optional
        The cohort number of the first timestep.
    evolving : bool, optional
        Specify whether the age of the cohort changes over time (True)
        or not (False).
    convertion_tab : Dict[str, Union[str,int,float]], optional
        The keys of this dictionary are the names of the fisheries in
        the SEAPODYM configuration file. The values are the names of the
        fisheries in the effort/catch text file.
        
        Example :
        convertion_tab = {'P1':1, 'P21':2, 'P3':3, 'S4':4, 'S5':5}

    See Also
    --------
    fisherieseffort :
        A module that contains many functions used to create the
        fisheries effort Dataset.
    fieldsdatastructure._readFisheries :
        This function is used to read fisheries parameters and create
        the `fisheries_parameters` dictionary.

    Returns
    -------
    xr.DataArray
        The fishery mortality.
    """

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

    length_fun = fh_structure.findLengthByCohort

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
                c_nb = fh_structure.cohorts_number
                tmp = np.arange(start_age, c_nb)
                age = np.concatenate(
                    (tmp,np.repeat(c_nb-1, effort_ds.time.data.size-tmp.size)))
            else :
                age = start_age

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

    return fisherieseffort.sumDataSet(fishing_mortality_ds, name="Mortality")










