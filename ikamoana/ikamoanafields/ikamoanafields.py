from typing import Dict, List, Tuple, Union

import xarray as xr
from ikamoana.utils import coordsAccess, latitudeDirection

from ..feedinghabitat import FeedingHabitat
from ..feedinghabitat import feedinghabitatconfigreader as fhcf
from ..fisherieseffort import fisherieseffort
from . import core
from ..utils import convertToNauticMiles

class IkamoanaFields :

# -------------------------- CORE FUNCTIONS -------------------------- #

    def __init__(
            self, IKAMOANA_config_filepath : str,
            SEAPODYM_config_filepath : str = None,
            feeding_habitat : xr.DataArray = None, root_directory: str = None):
        """Create a IkamoanaFields class. Can compute Taxis, Current, Diffusion,
        and Mortality fields.

        Parameters
        ----------
        IKAMOANA_config_filepath : str
            Path to the IKAMOANA configuration XML file.
        SEAPODYM_config_filepath : str, optional
            SEAPODYM configuration filepath can also be specified by
            user rather than in the IKAMOANA configuration file.
        root_directory : str, optional
            If the SEAPODYM configuration file is not in the root of the
            working directory, this (working) directory path must be specified.
        feeding_habitat : xr.DataArray, optional
            If the feeding habitat has already been calculated, it can
            be passed directly to the constructor.

        Raises
        ------
        TypeError
            `feeding_habitat` must be a Xarray.DataArray or None.
        """

        if feeding_habitat is None :
            self.feeding_habitat = None
        elif isinstance(feeding_habitat, xr.DataArray) :
            self.feeding_habitat = latitudeDirection(feeding_habitat,
                                                     south_to_north=True)
        else :
            raise TypeError((
                "feeding_habitat must be a Xarray.DataArray or None."
                "Current type is : {}").format(type(feeding_habitat)))

        self.ikamoana_fields_structure = core.IkamoanaFieldsDataStructure(
            IKAMOANA_config_filepath, SEAPODYM_config_filepath,
            root_directory=root_directory)
        self.feeding_habitat_structure = FeedingHabitat(
            self.ikamoana_fields_structure.SEAPODYM_config_filepath,
            root_directory=root_directory)
        
        if self.ikamoana_fields_structure.indonesian_filter :
            self.feeding_habitat_structure.indonesianFilter()
        if self.ikamoana_fields_structure.correct_epi_temp_with_vld :
            self.feeding_habitat_structure.correctEpiTempWithVld()

    def computeFeedingHabitat(
            self, cohort: int, time_start: int = None, time_end: int = None,
            lat_min: int = None, lat_max: int = None,lon_min: int = None,
            lon_max: int = None
            ):
        """This is a wrapper of the `FeedingHabitat.computeFeedingHabitat`."""
    
        # Manually control argument because their are used in the
        # returned section.
        (time_start,time_end,lat_min,lat_max,lon_min,lon_max) = (
            self.feeding_habitat_structure.controlArguments(
                time_start, time_end, lat_min, lat_max, lon_min, lon_max))

        feeding_habitat = (
            self.feeding_habitat_structure.computeFeedingHabitat(
                cohort, time_start, time_end, lat_min, lat_max, lon_min,
                lon_max, False))

        fh_name = list(feeding_habitat.var()).pop()
        feeding_habitat_da = feeding_habitat[fh_name]
        feeding_habitat_da.attrs.update(feeding_habitat.attrs)
        self.feeding_habitat = latitudeDirection(feeding_habitat_da,
                                                 south_to_north=True)
        
    def computeEvolvingFeedingHabitat(
            self, cohort_start: int = None, cohort_end: int = None,
            time_start: int = None, time_end: int = None, lat_min: int = None,
            lat_max: int = None, lon_min: int = None, lon_max: int = None
            ):
        """This is a wrapper of the `FeedingHabitat.computeEvolvingFeedingHabitat`."""
        
        # Manually control argument because their are used in the
        # returned section.
        (time_start,time_end,lat_min,lat_max,lon_min,lon_max) = (
            self.feeding_habitat_structure.controlArguments(
                time_start, time_end, lat_min, lat_max, lon_min, lon_max))

        feeding_habitat = (
            self.feeding_habitat_structure.computeEvolvingFeedingHabitat(
                cohort_start, cohort_end, time_start, time_end, lat_min,
                lat_max, lon_min, lon_max, False))
        self.feeding_habitat = latitudeDirection(feeding_habitat,
                                                 south_to_north=True)

## TODO later : Take into account L1 is a simplification.
# Should use accessibility + forage distribution + current L1/L2/L3
    def current(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Load current forcing from NetCDFs or Dymfiles. No unit
        convertion is applied here.

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray]
            U, V
        """

        u = fhcf.seapodymFieldConstructor(
            self.feeding_habitat_structure.data_structure.root_directory
            + self.ikamoana_fields_structure.u_file,  dym_varname='u_L1')
        v = fhcf.seapodymFieldConstructor(
            self.feeding_habitat_structure.data_structure.root_directory
            + self.ikamoana_fields_structure.v_file,  dym_varname='v_L1')
        
        u = latitudeDirection(u, south_to_north=True)
        v = latitudeDirection(v, south_to_north=True)

        if self.feeding_habitat is not None:
            # NOTE : We assume that U and V have same coordinates.
            timefun, latfun, lonfun = coordsAccess(u)
            minlon_idx = lonfun(min(self.feeding_habitat.coords['lon'].data))
            maxlon_idx = lonfun(max(self.feeding_habitat.coords['lon'].data))
            minlat_idx = latfun(min(self.feeding_habitat.coords['lat'].data))
            maxlat_idx = latfun(max(self.feeding_habitat.coords['lat'].data))
            mintime_idx = timefun(min(self.feeding_habitat.coords['time'].data))
            maxtime_idx = timefun(max(self.feeding_habitat.coords['time'].data))
            u = u[mintime_idx:maxtime_idx+1, minlat_idx:maxlat_idx+1,
                  minlon_idx:maxlon_idx+1]
            v = v[mintime_idx:maxtime_idx+1, minlat_idx:maxlat_idx+1,
                  minlon_idx:maxlon_idx+1]
        return u, v

    def computeTaxis(self) -> xr.DataArray:
        
        hf_cond, ssto_cond = (self.ikamoana_fields_structure.landmask_from_habitat,
                              self.ikamoana_fields_structure.shallow_sea_to_ocean)
        fh = self.feeding_habitat
        param = dict(data_structure=self.feeding_habitat_structure.data_structure,
                     habitat_field=fh, use_SEAPODYM_global_mask=(not hf_cond),
                     shallow_sea_to_ocean=ssto_cond,
                     lat_min=fh.attrs['lat_min'], lat_max=fh.attrs['lat_max'],
                     lon_min=fh.attrs['lon_min'], lon_max=fh.attrs['lon_max'])
        
        landmask = core.landmask(**param)

        grad_lon, grad_lat = core.gradient(self.feeding_habitat, landmask)

        return core.taxis(self.ikamoana_fields_structure,
                          self.feeding_habitat_structure.data_structure,
                          grad_lon, grad_lat, name=self.feeding_habitat.name)

    def computeMortality(self, verbose: bool = False) -> xr.DataArray :
        
        if not hasattr(self.ikamoana_fields_structure, "selected_fisheries") :
            raise ValueError("selected_fisheries can not be find. Make sure that"
                             " you gave mortality and selected_fisheries tags.")
        
        fh_struct = self.feeding_habitat_structure.data_structure
        ika_struct = self.ikamoana_fields_structure
        
        selected_fisheries = []
        for k, v in ika_struct.selected_fisheries.items() :
            selected_fisheries.append(k if (v == '') or (v is None) else v)

        param_dict = dict(
            filepath = ika_struct.fishery_filepaths,
            space_reso = fh_struct.parameters_dictionary['space_reso'],
            time_reso = fh_struct.parameters_dictionary['deltaT'],
            coords=fh_struct.coords,
            skiprows = ika_struct.skiprows,
            selected_fisheries = selected_fisheries,
            predict_effort = ika_struct.predict_effort,
            verbose=verbose         
        )
        effort_ds = fisherieseffort.effortByFishery(**param_dict)

        params_fisheries = self.ikamoana_fields_structure.f_param

        return core.fishingMortality(
            self.feeding_habitat_structure.data_structure, effort_ds,
            params_fisheries, convertion_tab=ika_struct.selected_fisheries)
        
#TODO : Rewrite documentation
    def computeDiffusion(
            self, landmask: xr.DataArray = None, lat_min: int = None,
            lat_max: int = None, lon_min: int = None, lon_max: int = None,
            current_u: xr.DataArray = None, current_v: xr.DataArray = None
            ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray,xr.DataArray]:
        """[summary]

        Parameters
        ----------
        landmask : xr.DataArray, optional
            [description], by default None

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray, xr.DataArray,xr.DataArray]
            [description]
        """
        
        if landmask is None :    
            hf_cond, ssto_cond = (
                self.ikamoana_fields_structure.landmask_from_habitat,
                self.ikamoana_fields_structure.shallow_sea_to_ocean)
            landmask = core.landmask(
                data_structure=self.feeding_habitat_structure.data_structure,
                habitat_field=self.feeding_habitat,
                use_SEAPODYM_global_mask=not(hf_cond), shallow_sea_to_ocean=ssto_cond,
                lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                field_output=False)
        
        diffusion_x, diffusion_y = core.diffusion(
            self.ikamoana_fields_structure,
            self.feeding_habitat_structure.data_structure,
            self.feeding_habitat, current_u, current_v, landmask)
        dKxdx, _ = core.gradient(diffusion_x, landmask)
        _, dKydy = core.gradient(diffusion_y, landmask)
        
        return diffusion_x, diffusion_y, dKxdx, dKydy

# ------------------------------- MAIN ------------------------------- #

    def computeIkamoanaFields(
            self, from_habitat: xr.DataArray = None, evolve: bool = True,
            cohort_start: int = None, cohort_end: int = None,
            time_start: int = None, time_end: int = None, lat_min: int = None,
            lat_max: int = None, lon_min: int = None, lon_max: int = None,
            south_to_north: bool = True, verbose: bool = False
            ) -> Dict[str, xr.DataArray]:
        """This is the main function of this module. It is use to
        provide all the fields needed by the `ikamoana` module.
        Note that the starting distribution is given by the
        `start_distribution()` function."""

        hf_cond, ssto_cond = (self.ikamoana_fields_structure.landmask_from_habitat,
                              self.ikamoana_fields_structure.shallow_sea_to_ocean)

        domain = dict(time_start=time_start, time_end=time_end, lat_min=lat_min,
                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
                
        if from_habitat is None :
            if evolve :
                self.computeEvolvingFeedingHabitat(
                    cohort_start=cohort_start, cohort_end=cohort_end, **domain)
            else :
                self.computeFeedingHabitat(cohort=cohort_start, **domain)
        else :
            self.feeding_habitat=from_habitat
                
        ## TODO : add possibility to invert latitudinal values (V * -1)
        u, v = self.current()
                
        taxis_lon, taxis_lat = self.computeTaxis()
        
        landmask = core.landmask(
            self.feeding_habitat_structure.data_structure,
            habitat_field=self.feeding_habitat, use_SEAPODYM_global_mask=not(hf_cond),
            shallow_sea_to_ocean=ssto_cond, lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max, field_output=True)
                
        diffusion_x, diffusion_y, dKxdx, dKydy = self.computeDiffusion(
            landmask[0], lat_min, lat_max, lon_min, lon_max, u, v)
        
        feeding_habitat = latitudeDirection(self.feeding_habitat,south_to_north)
        diffusion_x = latitudeDirection(diffusion_x,south_to_north)
        diffusion_y = latitudeDirection(diffusion_y,south_to_north)
        taxis_lon = latitudeDirection(taxis_lon,south_to_north)
        taxis_lat = latitudeDirection(taxis_lat,south_to_north)
        dKxdx = latitudeDirection(dKxdx,south_to_north)
        dKydy = latitudeDirection(dKydy,south_to_north)
        landmask = latitudeDirection(landmask,south_to_north)
        u = latitudeDirection(u,south_to_north)
        v = latitudeDirection(v,south_to_north)
        
        if evolve :
            feeding_habitat = feeding_habitat.drop_vars('cohorts')
            diffusion_x = diffusion_x.drop_vars('cohorts')
            diffusion_y = diffusion_y.drop_vars('cohorts')
            taxis_lon = taxis_lon.drop_vars('cohorts')
            taxis_lat = taxis_lat.drop_vars('cohorts')
            dKxdx = dKxdx.drop_vars('cohorts')
            dKydy = dKydy.drop_vars('cohorts')
                
        mortality_dict = {}
        if hasattr(self.ikamoana_fields_structure, "selected_fisheries") :
            mortality = self.computeMortality(verbose)
            mortality = latitudeDirection(mortality,south_to_north
                                          ).reindex_like(feeding_habitat)
            mortality_dict['F'] = mortality
        
        if self.ikamoana_fields_structure.units == 'nm_per_timestep' :
            timestep = self.ikamoana_fields_structure.timestep
            def convertionSimple(field):
                field = convertToNauticMiles(field, timestep)
                field.attrs['units'] = "nmi.dt⁻¹"
                return field
            def convertionSquare(field):
                field = convertToNauticMiles(field, timestep, square=True)
                field.attrs['units'] = "nmi².dt⁻¹"
                return field

            diffusion_x = convertionSquare(diffusion_x)
            diffusion_y = convertionSquare(diffusion_y)
            dKxdx = convertionSquare(dKxdx)
            dKydy = convertionSquare(dKydy)
            taxis_lon = convertionSimple(taxis_lon)
            taxis_lat = convertionSimple(taxis_lat)
            u = convertionSimple(u)
            v = convertionSimple(v)
        
        return {'H':feeding_habitat, 'landmask':landmask,
                'Kx':diffusion_x, 'Ky':diffusion_y,
                'dKx_dx':dKxdx, 'dKy_dy':dKydy,
                'Tx':taxis_lon, 'Ty':taxis_lat,
                'U':u, 'V':v,
                **mortality_dict}
