import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import parcels
import xarray as xr
from parcels.particle import JITParticle
import os

from ikamoana.ikafish import behaviours
from ikamoana.ikamoanafields.ikamoanafields import IkamoanaFields
from ikamoana.ikasimulation import IkaSimulation, KernelType
from ikamoana.utils import (coordsAccess,
                                                seapodymFieldConstructor)
from ikamoana.utils import latitudeDirection


class IkaSeapodym(IkaSimulation) :
    """
    Attributes
    ----------
    run_name : str
        The name of the simulation.
    random_seed : float
        Seed to generate random values.
    ika_params : dict
        Contains all the parameters and the kernels necessary to run the
        simulation. Also contains the filepaths to fields and to
        SEAPODYM configuration file.
    ocean : parcels.FieldSet
        Contains all the fields necessary for the simulation.
    fish : parcels.ParticleSet
        Contains the state of all particles.
    start_coords : DataArrayCoordinates
        Dictionary-like container of coordinate arrays.

    Examples
    --------
    First example : Simple simulation using a configuration file. Both
    fields and particle set are saved to NetCDF.
    See also the documentation on configuration files in `doc` directory.

    >>> my_sim = ikadym.IkaSeapodym(filepath="~/configuration_filepath.xml")
    >>> my_sim.loadFields()
    >>> my_sim.oceanToNetCDF(dir_path="~/ocean_fieldset", to_dataset=True)
    >>> my_sim.initializeParticleSet(
    ...     particles_class=IkaFish,
    ...     particles_number=10,
    ...     method="start_cell")
    >>> my_sim.fish.show(field=my_sim.ocean.U)
    >>> my_sim.runKernels(save=True)
    """

    def __init__(self, filepath: str):
        """Overrides `IkaSimulation.__init__()` by first reading a
        configuration file and then passing it all parameters.

        Parameters
        ----------
        filepath : str
            Path to the IKAMOANA configuration file. Refere to the
            documentation if you don't know what does this file must
            contains.
        """

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
            params['duration_time'] = int(float(time.find('sim_time').text)*86400)
            params['delta_time'] = int(float(time.find('dt').text)) #*86400)
            params['output_delta_time'] = int(float(time.find('output_dt').text)*86400)

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
            tmp = root.find('number_of_cohorts').text
            params['number_of_cohorts'] = int(tmp) if tmp not in [None,''] else 1

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
            forcing_dataarray = {}
            forcing_dataset = {}
            if files is not None :
                if "files_only" in files.attrib :
                    params['files_only'] = (files.attrib['files_only']
                                            in ["True", "true"])
                else :
                    params['files_only'] = False

                home=""
                if "home_directory"  in files.attrib :
                    home = files.attrib['home_directory']
                    params['files_home_directory'] = home

                for elmt in files :
                    if ("dataset" in elmt.attrib
                            and elmt.attrib["dataset"] in ["True","true"]):
                            forcing_dataset[elmt.tag] = os.path.join(home,elmt.text)
                    else :
                        forcing_dataarray[elmt.tag] = os.path.join(home,elmt.text)
            params["forcing_files"] = {"forcing_dataarray":forcing_dataarray,
                                       "forcing_dataset":forcing_dataset}
            tmp = root.find("field_interp_method").text
            params["fields_interp_method"] = "nearest" if tmp in [None, ""] else tmp

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

            delta_time_seapodym = int(float(root.find('deltaT').attrib['value'])*86400)
            params["delta_time_seapodym"] = delta_time_seapodym

        def readMortality(ikamoana_root:ET.Element, params:dict):
            if (ikamoana_root.find("mortality") is not None
                    and ikamoana_root.find("mortality").find("effort_file") is not None) :
                effort_file = ikamoana_root.find("mortality").find("effort_file")
                params['import_effort'] = effort_file.attrib.pop('import', None) in ["true","True"]
                params['export_effort'] = effort_file.attrib.pop('export', None) in ["true","True"]
                effort_file_txt = effort_file.text if effort_file.text is not None else ""
                if effort_file_txt == "" :
                    effort_file_txt = "{}_effort".format(params["run_name"])
                else :
                    effort_file_txt = os.path.splitext(effort_file_txt)[0]
                if not os.path.isabs(effort_file_txt) :
                    effort_file_txt = os.path.join(params['forcing_dir'],
                                                   effort_file_txt)
                params['effort_file'] = "{}.nc".format(effort_file_txt)

            tree = ET.parse(params['seapodym_file'])
            seapodym_root = tree.getroot()
            species_name = seapodym_root.find("sp_name").text

            params["mortality_constants"] = {
                'MPmax': float(seapodym_root.find('Mp_mean_max').attrib[species_name]),
                'MPexp': float(seapodym_root.find('Mp_mean_exp').attrib[species_name]),
                'MSmax': float(seapodym_root.find('Ms_mean_max').attrib[species_name]),
                'MSslope': float(seapodym_root.find('Ms_mean_slope').attrib[species_name]),
                'Mrange': float(seapodym_root.find('M_mean_range').attrib[species_name])}

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
# NOTE : Do we have to add these parameters only when NaturalMortality
# is in kernels list ?
        readMortality(root, params)

        return params

    def loadFields(
            self, from_habitat: xr.DataArray = None,
            fields_interp_method: str = None,
            landmask_interp_methode: str = 'nearest',
            allow_time_extrapolation: bool = True):
        """Compute (or load) a feeding habitat using the FeedingHabitat
        class then call the IkaFields class to generate Diffusion and
        Advection fields.
        Finaly loads all these fields into a Parcels.FieldSet structure.

        Parameters
        ----------
        from_habitat : xr.DataArray, optional
            If you already have computed the habitat, it can be passed
            directly to the function using this argument.
        fields_interp_method : str, optional
            Interpolation method used to create the parcels FieldSet.
            Please refer to the Parcels documentation.
        landmask_interp_methode : str, optional
            Interpolation method used to create the parcels landmask.
            Please refer to the Parcels documentation.
        allow_time_extrapolation : bool, optional
            This is a Parcels parameter passed at FieldSet creation.
            Please refer to the Parcels documentation.

        """

        def loadFieldsFromFiles():
# TODO : reindex new fields ?
            forcing_files = self.ika_params["forcing_files"]
            forcing = {}
            for _, filepath in forcing_files["forcing_dataset"].items() :
                ds = xr.load_dataset(filepath)
                for dr_name in ds :
                    forcing[dr_name] = ds[dr_name]
            for name, filepath in forcing_files["forcing_dataarray"].items() :
                forcing[name] = seapodymFieldConstructor(filepath, dym_varname=name)
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
            import_effort = (self.ika_params['effort_file']
                             if self.ika_params.pop('import_effort', False) else None)
            export_effort = (self.ika_params['effort_file']
                             if self.ika_params.pop('export_effort', False) else None)

            return generator.computeIkamoanaFields(
                from_habitat=from_habitat, evolve=evolve,
                cohort_start=ages[0], cohort_end=None,
                time_start=start, time_end=end,
                lon_min=lonlims[0], lon_max=lonlims[1],
                lat_min=latlims[1], lat_max=latlims[0],
                import_effort=import_effort, export_effort=export_effort)

        def readCohortDt():
            tree = ET.parse(self.ika_params['seapodym_file'])
            root = tree.getroot()
            deltaT = float(root.find('deltaT').attrib["value"])
            return deltaT*24*60*60

        if fields_interp_method is None :
            fields_interp_method = self.ika_params['fields_interp_method']

        files_forcing = loadFieldsFromFiles()

        generated_forcing = {} if self.ika_params["files_only"] else generateFields()

        # If a key is defined in both generated_forcing and files_forcing,
        # only the value in files_forcing is conserved.
        forcing = {**generated_forcing, **files_forcing}

        super().loadFields(fields=forcing, inplace=True,
                           landmask_interp_methode=landmask_interp_methode,
                           fields_interp_method=fields_interp_method,
                           allow_time_extrapolation=allow_time_extrapolation)

        self.ocean.add_constant('cohort_dt', readCohortDt())

        for cst, value in self.ika_params['mortality_constants'].items():
            self.ocean.add_constant(name=cst, value=value)

    def _fromCellToStartField(self) -> xr.DataArray:
        """Generate a DataArray full of 0 except for one cell ("start_cell"
        tag in configuration file) which is initialized with value 1."""

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
            self, start_dist, particles_number, area_scale=True
            ) -> Tuple[np.ndarray, np.ndarray]:
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
        """Sets the spatial boundaries of a field in the same way as
        `ocean.U`."""
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
            particles_number: int = None,
            particles_starting_time: Union[np.datetime64,List[np.datetime64]] = None,
            particles_variables: Dict[str,List[Any]] = {}):
        """Initialise the ParticleSet (`fish` attribut) with four
        differents way :
        - From lists of longitude and latitude.
        - Using `start_cell` method. N particles are randomly
        distributed in a single cell.
        - Using `start_static_file` method. Simple function returning
        random particle start positions using the density distribution
        saved in `start_dist` (2D field).
        - Using `start_dynamic_file` method. Prety similare to
        `start_static_file` method. Will use cohort age and start date
        to select the right density distribution in start_distribution
        directory.

        If both `start_cell`, `start_static_file` and
        `start_dynamic_file` are defined in configuration file, specify
        which one you want to use in `method` attribut.

        Parameters
        ----------
        particles_longitude : Union[list,np.ndarray], optional
            The longitudinal position of the particles.
        particles_latitude : Union[list,np.ndarray], optional
            The latitudinal position of the particles.
        particles_class : JITParticle, optional
            The class of particles to be used in this simulation.
            See also the ikafish module.
        method : str, optional
            Choose among `start_cell`, `start_static_file` and
            `start_dynamic_file`. Define which method will be used.
        particles_number : int, optional
            Default is the number of particle defined in the
            configuration file.
        particles_starting_time : Union[np.datetime64,List[np.datetime64]], optional
            Optional list of start time values for particles. If None,
            ika_params['start_time'] will be used.
        particles_variables : Dict[str,List[Any]], optional
            Variables to add to particles. {variable name : list of
            values for each particle}.

        Warning
        -------
        loadFields() must be called before this function.

        Raises
        ------
        ValueError
            In case of `start_dynamic_file` usage :
            `start_age` must be greater than 0. Initial distribution
            will use previous age cohort file.
        ValueError
            Both start_cell, start_filestem and start_dynamic_file are
            defined in configuration file. Specify which one you want
            to use in method attribut.
        ValueError
            If you don't use start_cell, start_static_file or
            start_dynamic_file methods, you must specify longitude and
            latitude positions for each particle using
            `particles_longitude` and `particles_latitude` attributs.
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
        if particles_number is None :
            particles_number = self.ika_params['number_of_cohorts']

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
                            * self.ika_params["delta_time_seapodym"]]*nb_particles,
                     **particles_variables}

        super().initializeParticleSet(
            particles_longitude, particles_latitude, particles_class,
            particles_starting_time, variables)

    def runKernels(
            self, kernels: Union[KernelType, Dict[str, KernelType]] = None,
            recovery: Dict[int, KernelType] = None,
            sample_kernels: list = [], delta_time: int = None,
            duration_time: int = None, save: bool = False, output_name: str = None,
            output_delta_time: int = None, verbose: bool = False, **kargs):
        """Execute a list of kernels functions defined in
        `ikafish.behaviours` over the particle set for multiple timesteps.
        Selected kernels are listed in the configuration file and
        compared to `ikafish.behaviours.AllKernels`.
        Optionally also provide sub-timestepping for particle output.

        Parameters
        ----------
        kernels : Union[KernelType, Dict[str, KernelType]], optional
            Keys are kernels name and values are kernels functions. If
            None, kernels in ika_params['kernels'] are loaded from
            `ikafish.behaviours.AllKernels`.
            Default value is extract from the configuration file.
        recovery : Dict[int, KernelType], optional
            Dictionary with additional `parcels.tools.error` recovery
            kernels to allow custom recovery behaviour in case of kernel
            errors.
        delta_time : int, optional
            It is either a timedelta object or a double. Use a negative
            value for a backward-in-time simulation.
            Default value is extract from the configuration file.
        duration_time : int, optional
            Length of the timestepping loop. Use instead of endtime. It
            is either a timedelta object or a positive double.
            Default value is extract from the configuration file.
        save : bool, optional
            Specify if you want to save particles history into a NetCDF
            file.
        output_name : str, optional
            Name of the `parcels.particlefile.ParticleFile` object from
            the ParticleSet. Default is then `run_name`.
        output_delta_time : int, optional
            Interval which dictates the update frequency of file output.
            It is either a timedelta object or a positive double.
            Default value is extract from the configuration file.
        verbose : bool, optional
            Boolean for providing a progress bar for the kernel
            execution loop.
        kargs : Any
            kargs is passed directly to ParticleSet.execute().

        Raises
        ------
        ValueError
            A kernel (in the configuration file) is not defined by
            behaviours.AllKernels.
        """

        if delta_time is None :
            delta_time = self.ika_params['delta_time']
        if duration_time is None :
            duration_time = self.ika_params['duration_time']
        if output_delta_time is None :
            output_delta_time = self.ika_params['output_delta_time']

        if kernels is None :
            behaviours_dict = {}
            interactions_dict = {}
            for k in self.ika_params['kernels'] :
                if k in behaviours.AllKernels :
                    behaviours_dict[k] = behaviours.AllKernels[k]
                elif k in behaviours.AllInteractions :
                    interactions_dict[k] = behaviours.AllInteractions[k]
                else :
                    raise ValueError(("{} kernel is not defined by "
                                      "behaviours.AllKernels.").format(k))

        if recovery is None :
            recovery = {parcels.ErrorCode.ErrorOutOfBounds:behaviours.KillFish}

        super().runKernels(
            kernels=behaviours_dict, interactions=interactions_dict,
            sample_kernels=sample_kernels, delta_time=delta_time, duration_time=duration_time,
            recovery=recovery, save=save, output_name=output_name,
            output_delta_time=output_delta_time, verbose=verbose, *kargs)
