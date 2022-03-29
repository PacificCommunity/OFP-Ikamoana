import warnings
from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr

from ...feedinghabitat.habitatdatastructure import HabitatDataStructure
from ...fisherieseffort import fisherieseffort
from ...utils import getCellEdgeSizes, latitudeDirection
from ..core import IkamoanaFieldsDataStructure


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

    dlon, dlat = getCellEdgeSizes(field)

    nlat = field.lat.size
    nlon = field.lon.size

    data = np.nan_to_num(field.data)
    landmask = landmask.data
    dVdlon = np.zeros(data.shape, dtype=np.float32)
    dVdlat = np.zeros(data.shape, dtype=np.float32)

    ## NOTE : Parallelised execution may help to do it faster.
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
    f_length = fh_structure.findLengthByCohort

    dx, dy = getCellEdgeSizes(dHdlon)
    
    for t in range(dHdlon.time.size):
        t_age = age[t] if is_evolving else age
        t_length = f_length(t_age) / 100 # Convert cm to meter

        ## NOTE : We use _getCellEdgeSizes function to compute dx and dy.
        # This function use GeographicPolar function from Parcels. The
        # latitude correction has already been applied.
        Tlon[t,:,:] = vMax(t_length) * dx * dHdlon.data[t,:,:] 
        Tlat[t,:,:] = vMax(t_length) * dy * dHdlat.data[t,:,:] 

    taxis_attrs = {**dHdlat.attrs, "units":"m².s⁻¹"}
        
    return (
        xr.DataArray(
            name="Taxis_longitude_"+(dHdlon.name if name is None else name),
            data=Tlon, coords=dHdlon.coords, dims=('time','lat','lon'),
            attrs=taxis_attrs),
        xr.DataArray(
            name="Taxis_latitude_"+(dHdlat.name if name is None else name),
            data=Tlat, coords=dHdlat.coords, dims=('time','lat','lon'),
            attrs=taxis_attrs))

def _diffusionCorrection(
        diffusion, dHdx, dHdy, dx, dy, current_u=None, current_v=None,
        vertical_movement=True) :
    """Copy the SEAPODYM behaviour. Correction by rho + vertical_movement
    which include currents magnitude.
    Currents and diffusion units must be m².s⁻¹.
    """
    
    rho = 0.99
    rho_x = 1.0 - rho * np.abs(dHdx) * dx
    rho_y = 1.0 - rho * np.abs(dHdy) * dy
    if (vertical_movement) :
        if current_u is None or current_v is None :
            raise ValueError("If vertical_movement is True, you must specify "
                             "currents fields (current_u, current_v)")
        #correction of rho by passive advection
        currents_magnitude = np.sqrt(current_u**2 + current_v**2)
        # TODO: this was used in SEAPODYM with nmi.dt, use carefully
        # fV = 1.0 - currents_magnitude/(500.0*dt/30.0+currents_magnitude)
        # We transformed 500 nm.dt into m.s
        fV = 1.0 - currents_magnitude/((926/2592)+currents_magnitude)
        rho_x = rho_x * fV
        rho_y = rho_y * fV
        
    
    ## NOTE : Latitude correction isn't necessary ? We use _getCellEdgeSizes
        # to compute dx and dy which use function GeographicPolar.
        # parcels.tools.converters.GeographicPolar :
        # Unit converter from geometric to geographic coordinates (m to degree)
        # with a correction to account for narrower grid cells closer to the poles.
    # latitude_correction = np.tile(dHdx.lat.data, (dHdx.lon.size, 1)).T
    # latitude_correction = np.cos(latitude_correction * np.pi/180)
    diffusion_x = rho_x * diffusion # * latitude_correction
    diffusion_y = rho_y * diffusion 
       
    return diffusion_x, diffusion_y
    
def diffusion(
        ika_structure: IkamoanaFieldsDataStructure,
        fh_structure: HabitatDataStructure, habitat: xr.DataArray,
        current_u: xr.DataArray = None, current_v: xr.DataArray = None,
        landmask: xr.DataArray = None, name: str = None
        ) -> Tuple[xr.DataArray,xr.DataArray] :
    
    """Computes the diffusion field (K) based on the feeding `habitat`.
    See the SEAPODYM User's Manual, page 32, Active Random Movement.
    
    See also SEAPODYM code :
    `dv_caldia.cpp` line 57, function `precaldia_comp`


    Parameters
    ----------
    ika_structure : IkamoanaFieldsDataStructure
        Ikamoana data structure that contains parameters.
    fh_structure : HabitatDataStructure
        SEAPODYM data structure that contains fields and parameters.
    habitat : xr.DataArray
        The feeding habitat used to compute the diffusion field.
    current_u : xr.DataArray, optional
        Current field along longitude.
    current_v : xr.DataArray, optional
        Current field along latitude.
    landmask : xr.DataArray, optional
        Mask describing land and ocean cells.
    name : str, optional
        Specify the name of the returned DataArrays.
        Syntax is : K_`name`.

    Returns
    -------
    Tuple[xr.DataArray,xr.DataArray]
        The diffusion fields : Kx, Ky
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
    habitat_data = np.nan_to_num(habitat.data)
    diffusion_x = np.zeros_like(habitat_data)
    diffusion_y = np.zeros_like(habitat_data)
    f_length = fh_structure.findLengthByCohort
    dHdx, dHdy = gradient(habitat,landmask)
    dx, dy = getCellEdgeSizes(habitat)
    lmax = f_length(-1) / 100
    Vmax_diff = 1.25
    
    # TODO : Cf. TODO below
    timestep = ika_structure.timestep
    
    for t in range(habitat.time.size):

        t_age = age[t] if is_evolving else age
        t_length = f_length(t_age) / 100 # Convert into meter
        d_speed  = Vmax_diff-0.25*t_length/lmax # fixed, given in 'body length' units
        d_inf = (d_speed*t_length)**2 / 4
        
        #######################################################
        # TODO : check if it is correct to multiply by timestep.
        # See also previous version of the ikamoana
        d_inf = ((d_speed*t_length)**2 / 4) * timestep
        #######################################################
        
        d_max = ika_structure.sigma_K * d_inf

        ## VECTORIZED
        diffusion = (
            ika_structure.sig_scale
            * d_max
            * (1 - ika_structure.c_scale * ika_structure.c
               * np.power(habitat_data[t,:,:], ika_structure.P))
            * ika_structure.diffusion_scale + ika_structure.diffusion_boost
        )
        
        diffusion_x[t,:,:], diffusion_y[t,:,:] = _diffusionCorrection(
            diffusion, dHdx[t,:,:], dHdy[t,:,:], dx, dy, current_u[t,:,:],
            current_v[t,:,:], ika_structure.vertical_movement)
        
    if landmask is not None :
        landmask = latitudeDirection(landmask, south_to_north=True)
        diffusion_x = np.where(landmask != 1, diffusion_x, np.NaN)
        diffusion_y = np.where(landmask != 1, diffusion_y, np.NaN)

    if (habitat.name is None) and name is None :
        name = 'Unnamed_DataArray'
    
    diffusion_attrs = {**habitat.attrs, "units":"m².s⁻¹"}
        
    return (
        xr.DataArray(
            data=diffusion_x, name="Kx_"+(habitat.name if name is None else name),
            coords=habitat.coords, dims=("time","lat","lon"), attrs=diffusion_attrs),
        xr.DataArray(
            data=diffusion_y, name="Ky_"+(habitat.name if name is None else name),
            coords=habitat.coords, dims=("time","lat","lon"), attrs=diffusion_attrs))
    
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

    return fisherieseffort.sumDataSet(fishing_mortality_ds, name="F")


