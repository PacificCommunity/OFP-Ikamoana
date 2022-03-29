import os
import xml.etree.ElementTree as ET

import numpy as np
import parcels as prcl
import xarray as xr
from numba import jit

from ikamoana.utils.feedinghabitatutils import coordsAccess, seapodymFieldConstructor
from ikamoana.utils.ikamoanafieldsutils import latitudeDirection

from .ikafish import behaviours, ikafish
from .ikamoanafields import IkamoanaFields


class IkaSim :

    def __init__(self, xml_parameterfile: str):

        self.ika_params = self._readParams(xml_filepath=xml_parameterfile)
        self.forcing_gen = IkamoanaFields(xml_parameterfile)
        ## A few key class variables that are required throughout
        # Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'time':'time', 'lat':'lat', 'lon':'lon'}
        ages = self.forcing_gen.feeding_habitat_structure.data_structure.findCohortByLength(self.ika_params['start_length'])
        self.start_age = ages[0]

        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

# -------------------------------------------------------------------- #

# TODO : DocString can be used on variables too. It can be useful to
# fully understand each parameter.
    def _readParams(self, xml_filepath: str) -> dict :
        """Reads the parameters from a XML parameter file and stores
        them in a dictionary."""

        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        params = {}

        params['run_name'] = root.find('run_name').text
        params['random_seed'] = root.find('random_seed').text
        if params['random_seed'] == '':
            params['random_seed'] = None

        directories = root.find('directories')
        params['start_distribution'] = directories.find('start_distribution').text
        params['seapodym_file'] = directories.find('seapodym_file').text
        params['forcing_dir'] = directories.find('forcing_dir').text

        cohort = root.find('cohort_info')
        params['start_length'] = float(cohort.find('start_length').text)
        tmp = cohort.find('ageing').text
        params['ageing_cohort'] = (tmp == 'True') or (tmp == 'true')
        params['start_cell_lon'] = cohort.find('start_cell_lon')
        if params['start_cell_lon'] is not None:
            params['start_cell_lon'] = float(params['start_cell_lon'].text)
            params['start_cell_lat'] = float(cohort.find('start_cell_lat').text)
        params['start_filestem'] = cohort.find('start_filestem')
        if params['start_filestem'] is not None:
            params['start_filestem'] = params['start_distribution'] + params['start_filestem'].text
            if "file_extension" in cohort.find('start_filestem').attrib :
                tmp = cohort.find(
                    'start_filestem').attrib["file_extension"]
                if tmp == '' or tmp is None :
                    params['start_filestem_extension'] = "nc"
                else :
                    params['start_filestem_extension'] = tmp

        domain = root.find('domain')

        time = domain.find('time')
        params['start_time'] = np.datetime64(time.find('start').text)
        # T in days ?
        params['T'] = int(time.find('sim_time').text)
        # TODO : Is it a repetition of the SEAPODYM deltaT ?
        # dt in days before conversion in seconds ?
        params['dt'] = int(time.find('dt').text)*86400
        # output_dt in days before conversion in seconds ?
        params['output_dt'] = int(time.find('output_dt').text)*86400

        params['spatial_lims'] = {
            'lonlim': (np.float32(domain.find('lon').find('min').text),
                       np.float32(domain.find('lon').find('max').text)),
            'latlim': (np.float32(domain.find('lat').find('min').text),
                       np.float32(domain.find('lat').find('max').text))
        }

        params['kernels'] = [i.text for i in root.find('kernels').findall('kernel')]

        return params

    def _readMortalityXML(
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
                                    'Ms_mean_slope').attrib[species_name]),
                   'Mrange': float(root.find(
                                    'M_mean_range').attrib[species_name])}

        return n_param

    def _setConstant(self, name, val):
        self.ocean.add_constant(name, val)

    def _addField(
            self, field, dims=None, name=None, time_extra=True,
            interp_method='nearest'):
        """Add a field to the self ocean attribut."""
        
        self.ocean.add_field(prcl.Field.from_xarray(
            field, name=field.name if name is None else name,
            dimensions=self.forcing_dims if dims is None else dims,
            allow_time_extrapolation=time_extra,
            interp_method=interp_method))

    # TODO : rewrite documentation.
    def _startDistribution(
            self, start_filestem:str, start_age:int, extension: str,
            field_resize: xr.DataArray
            ) -> xr.DataArray :
        """
        Returns the first two time steps of the particles starting
        distribution according to domain specifications.
        
        Warning
        -------
        `start_age` > 0. The previous age cohort distribution is used
        for the initialization.
        """
        
        if start_age-1 < 0:
            raise ValueError("start_age must be greater than 0. Initial "
                             "distribution will use previous age cohort file.")
        start_age -= 1
        start_time = self.ika_params["start_time"]
        
        filepath = "{}{}.{}".format(start_filestem, start_age, extension)
        dist = seapodymFieldConstructor(
            filepath, dym_varname="{}{}".format(start_filestem, start_age))
        dist = latitudeDirection(dist, south_to_north=True)
        
        # Clip dimensions to the same as the feeding habitats, but only
        timefun, latfun, lonfun  = coordsAccess(dist)
        mintime_idx = timefun(start_time)
        if field_resize is not None :
            minlon_idx = lonfun(min(field_resize.coords['lon'].data))
            maxlon_idx = lonfun(max(field_resize.coords['lon'].data))
            minlat_idx = latfun(min(field_resize.coords['lat'].data))
            maxlat_idx = latfun(max(field_resize.coords['lat'].data))
            return dist[mintime_idx-1:mintime_idx+1, minlat_idx:maxlat_idx+1, minlon_idx:maxlon_idx+1]
        else :
            return dist[mintime_idx-1:mintime_idx+1]

# -------------------------------------------------------------------- #

# TODO : to_file : None | True | nom_de_run
    def generateForcing(self, from_habitat: xr.DataArray = None, to_file=False):

        data_structure = self.forcing_gen.feeding_habitat_structure.data_structure
        ages = data_structure.findCohortByLength(self.ika_params['start_length'])
        start = data_structure.findIndexByDatetime(self.ika_params['start_time'])[0]
        end = data_structure.findIndexByDatetime(
            self.ika_params['start_time']+ self.ika_params['T'])[0]
        lonlims = self.ika_params['spatial_lims']['lonlim']
        lonlims = data_structure.findCoordIndexByValue(lonlims, coord='lon')
        lonlims = np.int32(lonlims)
        latlims = self.ika_params['spatial_lims']['latlim']
        latlims = data_structure.findCoordIndexByValue(latlims, coord='lat')
        latlims = np.int32(latlims)
        evolve = self.ika_params['ageing_cohort']

        self.forcing = self.forcing_gen.computeIkamoanaFields(
            from_habitat=from_habitat, evolve=evolve,
            # NOTE : must select one cohort
            cohort_start=ages[0], cohort_end=None, time_start=start, time_end=end,
            lon_min=lonlims[0], lon_max=lonlims[1], lat_min=latlims[1], lat_max=latlims[0],
        )

        if self.ika_params['start_filestem'] is not None:
           self.start_distribution = self._startDistribution(
               self.ika_params['start_filestem'], self.start_age,
               self.ika_params['start_filestem_extension'], self.forcing['H'])

        # Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'time':'time', 'lat':'lat', 'lon':'lon'}
        self.forcing_vars = dict([(i,i) for i in self.forcing.keys()])

        if to_file:
            for (var, forcing) in self.forcing.items():
                forcing.name=var
                forcing.to_netcdf(
                    path=os.path.join(self.ika_params['forcing_dir'],
                                      self.ika_params['run_name']+'_'+var+'.nc'))
            self.start_distribution.to_netcdf(
                os.path.join(self.ika_params['forcing_dir'],
                             self.ika_params['run_name']+'_start_distribution.nc'))


# TODO : comment éviter l'utilisation de allow_time_extrapolation=True ?
    def createFieldSet(
            self, from_disk: bool = False, variables: dict = None,
            landmask_interp_methode: str = 'nearest'):
        """[summary]

        Parameters
        ----------
        from_disk : bool, optional
            [description], by default False
        variables : dict, optional
            If None, names are automaticly created using `forcing_dir`
            and `run_name`.

            Example :
                `start_distribution` is optional.

                variables = {
                    "dKx_dx":"<run_name>_dKx_dx.nc",
                    "dKy_dy":"<run_name>_dKy_dy.nc",
                    "H":"<run_name>_H.nc",
                    "Kx":"<run_name>_Kx.nc",
                    "Ky":"<run_name>_Ky.nc",
                    "landmask":"<run_name>_landmask.nc",
                    "start_distribution":"<run_name>start_distribution.nc",
                    "Tx":"<run_name>_Tx.nc",
                    "Ty":"<run_name>_Ty.nc",
                    "U":"<run_name>_U.nc",
                    "V":"<run_name>_V.nc",
                    "F":"<run_name>_F.nc"}
        """

        if from_disk:
            if variables is None :
                list_var = ["dKx_dx", "dKy_dy", "H", "Kx", "Ky", "landmask",
                            "F", "Tx", "Ty", "U", "V"]
                if self.ika_params['start_filestem'] is not None:
                    self.start_distribution = self._startDistribution(
                        "{}{}.{}".format(self.ika_params['start_filestem'],
                                         self.start_age,
                                         self.ika_params['start_filestem_extension']))

                    list_var.append("start_distribution")
                variables = {
                    var: os.path.join(self.ika_params['forcing_dir'],
                                      self.ika_params['run_name']+'_'+var+'.nc')
                    for var in list_var}
            else :
                variables = {k: os.path.join(self.ika_params['forcing_dir'],v)
                            for k, v in variables.items()}

            if self.ika_params['start_filestem'] is not None:
                self.start_distribution = xr.load_dataarray(
                    variables.pop("start_distribution"))

            self.ocean = prcl.FieldSet.from_netcdf(
                variables, {k:k for k in variables.keys()},
                {'time':'time', 'lat':'lat', 'lon':'lon'},
                allow_time_extrapolation=True)

        else:
            dict_fields = {}
            landmask = self.forcing.pop("landmask")
            for k, v in self.forcing.items() :
                dict_fields[k] = prcl.Field.from_xarray(
                    v, k, self.forcing_dims, allow_time_extrapolation=True)
            self.ocean = prcl.FieldSet(
                dict_fields.pop('U'), dict_fields.pop('V'), dict_fields) 
            self._addField(landmask,name='landmask',
                          interp_method=landmask_interp_methode)

        if self.ika_params['start_filestem'] is not None:
            self.start_distribution = self._startDistribution(
                self.ika_params['start_filestem'], self.start_age,
                self.ika_params['start_filestem_extension'], self.forcing['H'])
            self.start_coords = self.start_distribution.coords
            self.start_distribution = prcl.Field.from_xarray(
                self.start_distribution, name='start_distribution',
                dimensions=self.forcing_dims)
            
        if self.ika_params['start_cell_lon'] is not None:
            self.start_distribution = self.createStartField(
                self.ika_params['start_cell_lon'], self.ika_params['start_cell_lat'])

        #Add necessary field constants
        #(constants easily accessed by particles during kernel execution)
        timestep = self.forcing_gen.ikamoana_fields_structure.timestep
        if 'NaturalMortality' in self.ika_params['kernels']:
            N_params = self._readMortalityXML(self.ika_params['seapodym_file'])
            self._setConstant('SEAPODYM_dt', timestep)
            for (p, val) in N_params.items():
                self._setConstant(p, val)
        if 'Age' in self.ika_params['kernels']:
            self._setConstant('cohort_dt', timestep)

    def startDistPositions(self, N, area_scale=True):
        """Simple function returning random particle start positions using
        the density distribution saved in self.start_dist. Includes option
        for scaling density by grid cell size (default true)."""

        # TODO : verify that this is a the desired behaviour
        print(self.start_distribution)
        self.start_distribution.data = self.start_distribution.data[0,:,:]

        data = self.start_distribution.data
        grid = self.start_distribution.grid

        #Assuming regular grid
        lonwidth = (grid.lon[1] - grid.lon[0]) / 2
        latwidth = (grid.lat[1] - grid.lat[0]) / 2

        # For distributions from a density on a spherical grid, we need to
        # rescale to a flat mesh
        def cell_area(lat,dx,dy):
            R = 6378.1
            Phi1 = lat*np.pi/180.0
            Phi2 = (lat+dy)*np.pi/180.0
            dx_radian = (dx)*np.pi/180
            S = R*R*dx_radian*(np.sin(Phi2)-np.sin(Phi1))
            return S

        if area_scale:
            for l in range(len(grid.lat)):
                area = cell_area(grid.lat[l],lonwidth,latwidth)
                data[l,:] *= area

        def add_jitter(pos, width, min, max):
            value = pos + np.random.uniform(-width, width)
            while not (min <= value <= max):
                value = pos + np.random.uniform(-width, width)
            return value

        p = np.reshape(data, (1, data.size))
        inds = np.random.choice(data.size, N, replace=True, p=p[0] / np.sum(p))
        lat, lon = np.unravel_index(inds, data.shape)
        lon = grid.lon[lon]#self.ocean.U.grid.lon[lon]
        lat = grid.lat[lat]#self.ocean.U.grid.lat[lat]

        for i in range(lon.size):
            lon[i] = add_jitter(lon[i], lonwidth, grid.lon[0], grid.lon[-1])
            lat[i] = add_jitter(lat[i], latwidth, grid.lat[0], grid.lat[-1])

        return lon, lat

    def createStartField(
            self, lon, lat, grid_lon=None, grid_lat=None) -> prcl.Field:
        """Function to create a particle starting distribution around a
        given cell vertex at a particular resolution"""
        
        if grid_lon is None:
            grid_lon = self.ocean.U.grid.lon
        if grid_lat is None:
            grid_lat = self.ocean.U.grid.lat
        start = np.zeros([len(grid_lat), len(grid_lon)],dtype=np.float32)
        coords = {'time': [self.ika_params['start_time']],
                  'lat': grid_lat,
                  'lon': grid_lon}
        start_dist = xr.DataArray(name = "start",
                            data = start[np.newaxis,:,:],
                            coords = coords,
                            dims=('time','lat','lon'))
        self.start_coords = start_dist.coords #For density calculation later
        start_dist = prcl.Field.from_xarray(start_dist, name='start_dist',
                                            dimensions=self.forcing_dims,
                                            interp_method='nearest')
        latidx = np.argmin(np.abs(start_dist.lat-lat))
        lonidx = np.argmin(np.abs(start_dist.lon-lon))
        start_dist.data[:,latidx,lonidx] = 1
        return start_dist

    def initialiseFishParticles(
            self, start: np.ndarray = None, n_fish:int = 10,
            pclass: prcl.JITParticle = prcl.JITParticle):
        """`start` has shape = (2, x) where [0,:] are longitude positions
        and [1,:] are latitude positions."""

        if isinstance(start, np.ndarray) :
            if start.shape[1] != n_fish :
                raise ValueError('Number of fish and provided initial positions'
                                 ' not equal!')
            plon = start[0]
            plat = start[1]
        else:
            if self.start_distribution is None :
                raise ValueError('No starting distribution field in ocean fieldset!')
            plon, plat = self.startDistPositions(n_fish)
        
        # NOTE : I'm not sure that 'cohorts_sp_unit' uses is robust.
        # Initialise fish
        cohort_dt = self.forcing_gen.feeding_habitat_structure.data_structure.\
            species_dictionary['cohorts_sp_unit'][0] * 86400
        
        self.fish = prcl.ParticleSet.from_list(
            fieldset=self.ocean, time=self.ika_params['start_time'],
            lon=plon, lat=plat, pclass=pclass,
            age_class = [self.start_age]*len(plon),
            age = [self.start_age*cohort_dt]*len(plon)
        )

        self._setConstant('cohort_dt', cohort_dt)

    def fishDensity(self):
        """Function calculating particle density at current time, using
        the same grid resolution given by the start field coordinates.
        Method replicates Seapodym biomass density ouputs, returning as
        an xarray dataarray"""

        lons = self.start_distribution.grid.lon
        lats = self.start_distribution.grid.lat
        #Assuming regular grid
        lonwidth = (lons[1] - lons[0]) / 2
        latwidth = (lats[1] - lats[0]) / 2
        #this is very slow with using JIT to compile to C
        @jit(nopython=True)
        def calc_D(lons, lats, grid_lon, grid_lat, D):
            for i in np.arange(1,len(grid_lon)):
                for j in np.arange(1,len(grid_lat)):
                    x = np.logical_and(lons>=grid_lon[i-1], lons<grid_lon[i])
                    y= np.logical_and(lats>=grid_lat[j-1], lats<grid_lat[j])
                    idx = np.logical_and(x, y)
                    D[j-1,i-1] = np.sum(idx)
            return(D)

        D = np.zeros((len(lats), len(lons)))
        Density = calc_D(np.array([f.lon for f in self.fish]),
                      np.array([f.lat for f in self.fish]),
                      lons-lonwidth, lats-latwidth, D)

        Density = Density[np.newaxis,:,:]
        density_coords = {
            'time': [np.array(self.ocean.time_origin.fulltime(
                self.fish.time[0]),dtype=np.datetime64)],
            'lat': self.start_coords['lat'].data,
            'lon': self.start_coords['lon'].data}
        PDensity = xr.DataArray(name = "Pdensity", data = Density,
                                coords = density_coords, dims=('time','lat','lon'))
        return PDensity

# TODO : vérifier si une particule dépasse le dernier timestep (WARNING)
# TODO : T -> valeur par défaut est égale à self.ika_params['T'] * nb sec par jour
    def runKernels(self, T=None, pfile_suffix='', verbose=True):
        """`T` in days."""

        if T is None :
            T = self.ika_params['T']
        T *= 86400 # converted to seconds

        pfile = self.fish.ParticleFile(
            name=self.ika_params['run_name']+pfile_suffix+'.nc',
            outputdt=self.ika_params['output_dt'])

        Behaviours = [self.fish.Kernel(behaviours.AllKernels[b])
                      for b in self.ika_params['kernels']]

        # Run an initial field sampling kernel loop
        if 'getRegion' in self.ika_params['kernels']:
            self.fish.execute(self.fish.Kernel(behaviours.getRegion), dt=0)

        KernelString = ''.join(
            [('Behaviours[{}]+').format(i) for i in range(len(Behaviours))])
        run_kernels = eval(KernelString[:-1])

        self.fish.execute(
            run_kernels, runtime=T, dt=self.ika_params['dt'], output_file=pfile,
            recovery={prcl.ErrorCode.ErrorOutOfBounds:behaviours.KillFish},
            verbose_progress=verbose)
