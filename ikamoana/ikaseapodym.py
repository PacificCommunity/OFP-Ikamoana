import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import numpy as np
import parcels
import xarray as xr
from parcels.particle import JITParticle

from ikamoana.ikafish import behaviours
from ikamoana.ikamoanafields.ikamoanafields import IkamoanaFields
from ikamoana.ikasimulation import IkaSimulation, KernelType
from ikamoana.utils.feedinghabitatutils import (coordsAccess,
                                                seapodymFieldConstructor)
from ikamoana.utils.ikamoanafieldsutils import latitudeDirection


class IkaSeapodym(IkaSimulation) :
    
    def __init__(self, filepath: str):
        """Overrides `IkaSimulation.__init__()` by first reading a
        configuration file and then passing it all parameters."""
        
        parameter_file_dict = self._readConfigFile(filepath)
        
        super().__init__(parameter_file_dict.pop("run_name"),
                         parameter_file_dict.pop("random_seed"))
        
        self.ika_params = parameter_file_dict
        self.ika_params['ikamoana_file'] = filepath
    
    def _readConfigFile(self, filepath:str) -> dict :
        """Reads a configuration file and returns a dictionary with all
        the parameters necessary for the initialization of this class."""
        
        def readBasis(root:ET.Element, params:dict) :
            params['run_name'] = root.find('run_name').text
            params['random_seed'] = root.find('random_seed').text
            if params['run_name'] == '':
                params['run_name'] = None
            if params['random_seed'] == '':
                params['random_seed'] = None
                
        def readDirectories(root:ET.Element, params:dict) :
            
            params['start_distribution'] = root.find('start_distribution').text
            params['seapodym_file'] = root.find('seapodym_file').text
            params['forcing_dir'] = root.find('forcing_dir').text
        
        def readDomain(root:ET.Element, params:dict) :
            time = root.find('time')
            params['start_time'] = np.datetime64(time.find('start').text)
            # All times are in seconds (converted from days)
            params['duration_time'] = int(time.find('sim_time').text)*86400
            params['delta_time'] = int(time.find('dt').text)*86400
            params['output_delta_time'] = int(time.find('output_dt').text)*86400

            params['spatial_limits'] = {
                'lonlim': (np.float32(root.find('lon').find('min').text),
                           np.float32(root.find('lon').find('max').text)),
                'latlim': (np.float32(root.find('lat').find('min').text),
                           np.float32(root.find('lat').find('max').text))
            }
        
        def readCohortInfo(root:ET.Element, params:dict) :
            params['start_length'] = float(root.find('start_length').text)
            tmp = root.find('ageing').text
            params['ageing_cohort'] = (tmp == 'True') or (tmp == 'true')

            if root.find('start_dynamic_file') is not None :
                params['start_dynamic_file'] = (params['start_distribution']
                                                + root.find('start_dynamic_file').text)
                if "file_extension" in root.find('start_dynamic_file').attrib :
                    tmp = root.find(
                        'start_dynamic_file').attrib["file_extension"]
                    if tmp == '' or tmp is None :
                        params['start_dynamic_file_extension'] = "nc"
                    else :
                        params['start_dynamic_file_extension'] = tmp
            
            if root.find('start_static_file') is not None :
                params['start_static_file'] = (params['start_distribution']
                                               + root.find('start_static_file').text)
                        
            if root.find('start_cell') is not None :
                params['start_cell'] = {
                    "lon":float(root.find('start_cell').find('lon').text),
                    "lat":float(root.find('start_cell').find('lat').text)}
            
        def readForcing(root:ET.Element, params:dict) :
            files = root.find("files")
            params['files_only'] = False
            forcing_files = {}
            if files is not None :
                if "file_only" in files.attrib :
                    params['files_only'] = (files.attrib['files_only']
                                            in ["True", "true"])
                else :
                    params['files_only'] = False
                for elmt in files :
                    forcing_files[elmt.tag] = elmt.text
            params["forcing_files"] = forcing_files
        
        def readKernels(root:ET.Element, params:dict) :
            params['kernels'] = [i.text for i in root.findall('kernel')]
        
        def readAge(seapodym_filepath:str, start_length:float, params:dict):
            tree = ET.parse(seapodym_filepath)
            root = tree.getroot()
            
            sp_name = root.find('sp_name').text
            length_list = np.array(
                [float(x) for x in root.find('length').find(sp_name).text.split()])
            index = np.absolute(length_list-start_length).argmin()
            params["start_age"] = index

        tree = ET.parse(filepath)
        root = tree.getroot()
        params = {}

        readBasis(root, params)
        readDirectories(root.find('directories'), params)
        readDomain(root.find('domain'), params)
        readForcing(root.find("forcing"), params)
        readCohortInfo(root.find('cohort_info'), params)
        readKernels(root.find('kernels'), params)
        readAge(params['seapodym_file'], params["start_length"], params)

        return params
    
    def loadFields(
            self, from_habitat: xr.DataArray = None,
            landmask_interp_methode: str = 'nearest',
            allow_time_extrapolation: bool = True):
        
        def loadFieldsFromFiles():
# TODO : verify that we use the right function to load netcdf
# TODO : use reshaping
            forcing = {}
            for k, v in self.ika_params["forcing_files"].items() :
                forcing[k] = xr.open_dataarray(v)
            return forcing
        
        def generateFields():
            generator = IkamoanaFields(self.ika_params['ikamoana_file'])
            
            data_structure = generator.feeding_habitat_structure.data_structure
            ages = data_structure.findCohortByLength(self.ika_params['start_length'])
            start = data_structure.findIndexByDatetime(self.ika_params['start_time'])[0]
            end = data_structure.findIndexByDatetime(
                self.ika_params['start_time']
                + np.timedelta64(self.ika_params['duration_time']+1, 's'))[0]
            lonlims = self.ika_params['spatial_limits']['lonlim']
            latlims = self.ika_params['spatial_limits']['latlim']
            lonlims = np.int32(data_structure.findCoordIndexByValue(lonlims,'lon'))
            latlims = np.int32(data_structure.findCoordIndexByValue(latlims,'lat'))
            evolve = self.ika_params['ageing_cohort']

            return generator.computeIkamoanaFields(
                from_habitat=from_habitat, evolve=evolve,
                cohort_start=ages[0], cohort_end=None, time_start=start, time_end=end,
                lon_min=lonlims[0], lon_max=lonlims[1], lat_min=latlims[1], lat_max=latlims[0],
            )
        
        files_forcing = loadFieldsFromFiles()
        
        generated_forcing = {} if self.ika_params["files_only"] else generateFields()
        
        # If a key is defined in both generated_forcing and files_forcing,
        # only the value in files_forcing is conserved.
        forcing = {**generated_forcing, **files_forcing}
        
        super().loadFields(fields=forcing,inplace=True,
                           landmask_interp_methode=landmask_interp_methode,
                           allow_time_extrapolation=allow_time_extrapolation)
    
    def _fromCellToStartField(self):
        grid_lon = self.ocean.U.grid.lon
        grid_lat = self.ocean.U.grid.lat
        shape = (1,len(grid_lat),len(grid_lon))
        start = np.zeros(shape, dtype=np.float32)
        coords = {'time': [self.ika_params['start_time']],
                    'lat': grid_lat, 'lon': grid_lon}
        dimensions = ('time','lat','lon')
        
        start_dist = xr.DataArray(
            name="start_dist", data=start, coords=coords, dims=dimensions)
        self.start_coords = start_dist.coords #For density calculation later
        start_dist = parcels.Field.from_xarray(
            start_dist, name='start_dist', dimensions={d:d for d in dimensions},
            interp_method='nearest')
        
        lat = self.ika_params["start_cell"]["lat"]
        lon = self.ika_params["start_cell"]["lon"]
        latidx = np.argmin(np.abs(start_dist.lat-lat))
        lonidx = np.argmin(np.abs(start_dist.lon-lon))
        start_dist.data[:,latidx,lonidx] = 1
        return start_dist
    
    def _fromStartFieldToCoordinates(
            self, start_dist, particles_number, area_scale=True):
        """Simple function returning random particle start positions using
        the density distribution saved in `start_dist`. Includes option
        for scaling density by grid cell size (default true)."""
     
        def cell_area(lat,dx,dy):
            """For distributions from a density on a spherical grid, we
            need to rescale to a flat mesh"""
            R = 6378.1
            Phi1 = lat*np.pi/180.0
            Phi2 = (lat+dy)*np.pi/180.0
            dx_radian = (dx)*np.pi/180
            S = R*R*dx_radian*(np.sin(Phi2)-np.sin(Phi1))
            return S
        
        def add_jitter(pos, width, min, max):
            value = pos + np.random.uniform(-width, width)
            while not (min <= value <= max):
                value = pos + np.random.uniform(-width, width)
            return value

        data = start_dist.data[0,:,:]
        grid = start_dist.grid
        #Assuming regular grid
        lonwidth = (grid.lon[1] - grid.lon[0]) / 2
        latwidth = (grid.lat[1] - grid.lat[0]) / 2

        if area_scale:
            for l in range(len(grid.lat)):
                area = cell_area(grid.lat[l],lonwidth,latwidth)
                data[l,:] *= area
                
        p = np.reshape(data, (1, data.size))
        inds = np.random.choice(
            data.size, particles_number, replace=True, p=p[0] / np.sum(p))
        lat, lon = np.unravel_index(inds, data.shape)
        lon = grid.lon[lon]
        lat = grid.lat[lat]

        for i in range(lon.size):
            lon[i] = add_jitter(lon[i], lonwidth, grid.lon[0], grid.lon[-1])
            lat[i] = add_jitter(lat[i], latwidth, grid.lat[0], grid.lat[-1])

        return lon, lat
    
    def _rescaleFieldWithUCoordinates(self, field):
        _, latfun, lonfun = coordsAccess(field)
        minlon_idx = lonfun(min(self.ocean.U.lon.data))
        maxlon_idx = lonfun(max(self.ocean.U.lon.data))
        minlat_idx = latfun(min(self.ocean.U.lat.data))
        maxlat_idx = latfun(max(self.ocean.U.lat.data))
        if "time" in field.indexes :
            return field.isel(time=0, lat=slice(minlat_idx,maxlat_idx+1),
                              lon=slice(minlon_idx,maxlon_idx+1))
        else :
            return field.isel(lat=slice(minlat_idx,maxlat_idx+1),
                              lon=slice(minlon_idx,maxlon_idx+1))
    
    def initializeParticleSet(
            self, particles_longitude:Union[list,np.ndarray] = None,
            particles_latitude:Union[list,np.ndarray] = None,
            particles_class: JITParticle = JITParticle, method: str = None,
            particles_number: int = 10,
            particles_starting_time: Union[np.datetime64,List[np.datetime64]] = None,
            particles_variables: Dict[str,List[Any]] = {}):
        """If both `start_cell`, `start_static_file` and
        `start_dynamic_file` are defined in configuration file, specify
        which one you want to use in `method` attribut.
        
        Warning
        -------
        loadFields() must be called before this function.
        """
        
        # Internal functions
        
        def initializeWithCell():
            start_field = self._fromCellToStartField()
            return self._fromStartFieldToCoordinates(start_field, particles_number)
            
        def initializeWithStaticFile():
            start_field = seapodymFieldConstructor(self.ika_params["start_static_file"])
            start_field = latitudeDirection(start_field, south_to_north=True)
            start_field = self._rescaleFieldWithUCoordinates(start_field)
            
            start_field = parcels.Field.from_xarray(
                start_field, name="start_distribution",
                dimensions={d:d for d in list(start_field.indexes)},
                interp_method='nearest')
            
            if hasattr(self.ocean, "start_distribution") :
                delattr(self.ocean, "start_distribution")
            self.ocean.add_field(start_field, "start_distribution")
            
            return self._fromStartFieldToCoordinates(start_field, particles_number)
        
        def initializeWithDynamicFile():
            file_prefix = self.ika_params["start_dynamic_file"]
            start_age = self.ika_params['start_age'] - 1
            file_extension = self.ika_params["start_dynamic_file_extension"]
            if start_age < 0:
                raise ValueError("start_age must be greater than 0. Initial "
                                 "distribution will use previous age cohort file.")
                
            start_field = seapodymFieldConstructor(
                "{}{}.{}".format(file_prefix, start_age, file_extension),
                dym_varname="{}{}".format(file_prefix, start_age))
            start_field = latitudeDirection(start_field, south_to_north=True)
            
            timefun, _, _  = coordsAccess(start_field)
            mintime_idx = timefun(self.ika_params['start_time']) - 1
            start_field = start_field.isel(time=mintime_idx)
            start_field = self._rescaleFieldWithUCoordinates(start_field)
            start_field = parcels.Field.from_xarray(
                start_field, name="start_distribution",
                dimensions={d:d for d in list(start_field.indexes)},
                interp_method='nearest')
            
            if hasattr(self.ocean, "start_distribution") :
                delattr(self.ocean, "start_distribution")
            self.ocean.add_field(start_field, "start_distribution")
            
            return self._fromStartFieldToCoordinates(start_field, particles_number)
        
        # Verification : Is ParticleSet is already initialized ?
        
        if hasattr(self, "fish") :
            delattr(self, "fish")
        if self.ocean.completed :
            self.ocean.completed = False
        
        if particles_longitude is None and particles_latitude is None :

            condition_cell = 'start_cell' in self.ika_params
            condition_static_file = 'start_static_file' in self.ika_params
            condition_dynamic_file = 'start_dynamic_file' in self.ika_params
            condition_multiple = (condition_cell+condition_static_file+condition_dynamic_file)>1
            
            if condition_multiple and method not in ['start_cell','start_static_file',
                                                     'start_dynamic_file'] :
                raise ValueError(
                    "Both start_cell, start_filestem and start_dynamic_file are "
                    "defined in configuration file. Specify which one you want "
                    "to use in method attribut.")
            
            if ((condition_multiple and method == 'start_cell')
                    or (condition_cell and not condition_multiple)) :
                particles_longitude, particles_latitude = initializeWithCell()
            elif ((condition_multiple and method == 'start_static_file')
                    or (condition_static_file and not condition_multiple)) :
                particles_longitude, particles_latitude = initializeWithStaticFile()
            elif ((condition_multiple and method == 'start_dynamic_file')
                    or (condition_dynamic_file and not condition_multiple)) :
                particles_longitude, particles_latitude = initializeWithDynamicFile()
            else :
                raise ValueError(
                    "If you don't use start_cell, start_static_file or "
                    "start_dynamic_file, you must specify longitude and latitude "
                    "positions for each particle using particles_longitude and "
                    "particles_latitude attributs.")

        if particles_starting_time is None :
            particles_starting_time = self.ika_params['start_time']
        nb_particles = len(particles_latitude)
        variables = {"age_class":[self.ika_params["start_age"]]*nb_particles,
                     "age":[self.ika_params["start_age"]
                            * self.ika_params["delta_time"]]*nb_particles,
                     **particles_variables}

        super().initializeParticleSet(
            particles_longitude, particles_latitude, particles_class,
            particles_starting_time, variables)
        
    def runKernels(
            self, kernels: Union[KernelType, Dict[str, KernelType]] = None,
            recovery: Dict[int, KernelType] = None, delta_time: int = None,
            duration_time: int = None, save: bool = False, output_name: str = None,
            output_delta_time: int = None, verbose: bool = False, **kargs):
        # kargs is passed to ParticleSet.execute()
        #
        # All kernels here are already writed in an other file
        # (behaviours). Configuration file will specify the ones we want
        # to use in this simulation.

        if delta_time is None :
            delta_time = self.ika_params['delta_time']
        if duration_time is None :
            duration_time = self.ika_params['duration_time']
        if output_name is None :
            output_name = "{}.nc".format(self.run_name)
        if output_delta_time is None :
            output_delta_time = self.ika_params['output_delta_time']

        if kernels is None :
            behaviours_dict = {}
            for k in self.ika_params['kernels'] :
                if k in behaviours.AllKernels :
                    behaviours_dict[k] = behaviours.AllKernels[k]
                else :
                    raise ValueError(("{} kernel is not defined by "
                                      "behaviours.AllKernels.").format(k))
        
        if recovery is None :
            recovery = {parcels.ErrorCode.ErrorOutOfBounds:behaviours.KillFish}

        super().runKernels(
            kernels=behaviours_dict, delta_time=delta_time, duration_time=duration_time,
            recovery=recovery, save=save, output_name=output_name,
            output_delta_time=output_delta_time, verbose=verbose, *kargs)
