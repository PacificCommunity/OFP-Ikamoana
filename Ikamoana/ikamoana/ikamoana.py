import xml.etree.ElementTree as ET

import xarray as xr
import numpy as np
from numba import jit
import ikamoana as ika
from .ikafish import ikafish, behaviours
import parcels as prcl

# from ..ikamoana.ikamoanafields import IkamoanaFields
# from .ikafish import behaviours, ikafish

class IkaSim :

    def __init__(self, xml_parameterfile: str):

        self.ika_params = self._readParams(xml_filepath=xml_parameterfile)
        self.forcing_gen = ika.ikamoanafields.IkamoanaFields(IKAMOANA_config_filepath=xml_parameterfile,
                                                             SEAPODYM_config_filepath=self.ika_params['SEAPODYM_file'])

        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

# TODO : This version of the generateForcing will be removed
    def generateForcing(self, to_file=False):

        data_structure = self.forcing_gen.feeding_habitat_structure.data_structure
        ages = data_structure.findCohortByLength(self.ika_params['start_length'])
        start = data_structure.findIndexByDatetime(self.ika_params['start_time'])[0]
        end = data_structure.findIndexByDatetime(
            self.ika_params['start_time']+ self.ika_params['T'])[0]
        self.start_age = ages[0]

        lonlims = self.ika_params['spatial_lims']['lonlim']
        lonlims = data_structure.findCoordIndexByValue(lonlims, coord='lon')
        #lonlims = [int(l) for l in lonlims]
        # NOTE : Is equivalent to
        lonlims = np.int32(lonlims)
        latlims = self.ika_params['spatial_lims']['latlim']
        latlims = data_structure.findCoordIndexByValue(latlims, coord='lat')
        # latlims = [int(l) for l in latlims]
        latlims = np.int32(latlims)

        self.forcing = {}

        if self.ika_params['ageing_cohort']:
            self.forcing['Tx'], self.forcing['Ty'] = self.forcing_gen.computeEvolvingTaxis(
                cohort_start=ages[0], time_start=start, time_end=end,
                lon_min=lonlims[0], lon_max=lonlims[1], lat_min=latlims[1],
                lat_max=latlims[0])
        else:
            self.forcing['Tx'], self.forcing['Ty'] = self.forcing_gen.computeTaxis(
                cohort=ages, time_start=start, time_end=end, lon_min=lonlims[0],
                lon_max=lonlims[1], lat_min=latlims[1], lat_max=latlims[0])

        self.forcing['landmask'] = self.forcing_gen.landmask(
            use_SEAPODYM_global_mask=True, field_output=True,
            habitat_field=self.forcing_gen.feeding_habitat)

        self.forcing['H'] = self.forcing_gen.feeding_habitat

        if self.ika_params['start_filestem'] is not None:
            print(self.ika_params['start_filestem'])
            self.forcing['start'] = self.forcing_gen.start_distribution(
                self.ika_params['start_filestem']+str(self.start_age)+'.dym')
            print(self.ika_params['start_filestem']+str(self.start_age)+'.dym')

        ### Mortality fields to do

        self.forcing['U'], self.forcing['V'] = self.forcing_gen.current_forcing()

        self.forcing['K'] = self.forcing_gen.diffusion(self.forcing_gen.feeding_habitat)

        self.forcing['dK_dx'], self.forcing['dK_dy'] = self.forcing_gen.gradient(
            self.forcing['K'], self.forcing_gen.landmask(
                self.forcing_gen.feeding_habitat, lon_min=lonlims[0],
                lon_max=lonlims[1], lat_min=latlims[1], lat_max=latlims[0]),
            name='K')


        if to_file:
            for (var, forcing) in self.forcing.items():
                forcing.to_netcdf(path='%s/%s_%s.nc' % (
                    self.ika_params['forcing_dir'],
                    self.ika_params['run_name'], var))

        #Parcels will need a mapping of dimension coordinate names
        self.forcing_vars = {}
        for f in self.forcing:
            self.forcing_vars.update({f:f})
        self.forcing_dims = {'lon':'lon', 'lat':'lat', 'time':'time'}

    def generateForcingNEW(self, from_habitat=None, to_file=False):

        data_structure = self.forcing_gen.feeding_habitat_structure.data_structure
        ages = data_structure.findCohortByLength(self.ika_params['start_length'])
        start = data_structure.findIndexByDatetime(self.ika_params['start_time'])[0]
        end = data_structure.findIndexByDatetime(
            self.ika_params['start_time']+ self.ika_params['T'])[0]
        self.start_age = ages[0]
        lonlims = self.ika_params['spatial_lims']['lonlim']
        lonlims = data_structure.findCoordIndexByValue(lonlims, coord='lon')
        lonlims = np.int32(lonlims)
        latlims = self.ika_params['spatial_lims']['latlim']
        latlims = data_structure.findCoordIndexByValue(latlims, coord='lat')
        latlims = np.int32(latlims)
        evolve = self.ika_params['ageing_cohort']

        self.forcing = self.forcing_gen.computeIkamoanaFields(
            # TODO : Mortality is coming soon ---------------------------------
            effort_filepath=None,fisheries_xml_filepath=None, time_reso=None,
            space_reso=None, skiprows=None, removeNoCatch=None, predict_effort=None,
            remove_fisheries=None, convertion_tab=None,
            # ----------------------------------------------------------
            from_habitat=from_habitat,
            evolve=evolve,
            # NOTE : must select one cohort
            cohort_start= ages[0],
            cohort_end= None,
            time_start=start, time_end=end,
            lon_min=lonlims[0], lon_max=lonlims[1],
            lat_min=latlims[1], lat_max=latlims[0],
        )

        # TODO : this is temporary generation of the start_distribution
        # Note that we distribute fish using seapodym density from the PRIOR age class
        if self.ika_params['start_filestem'] is not None:
            self.start_dist = self.forcing_gen.start_distribution(
                self.ika_params['start_filestem']+str(self.start_age-1)+'.dym')

        # Mortality hasn't the same coordinates as others.
        if 'mortality' in self.forcing.keys():
            self.mortality = self.forcing.pop('mortality')
        # Landmask hasn't the same coordinates as others.
        self.landmask = self.forcing.pop('landmask')

        self.forcing = xr.Dataset(self.forcing)

        self.forcing_vars = dict([(i,i) for i in self.forcing.keys()])
        # Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'time':'time', 'lat':'lat', 'lon':'lon'}

        # TODO : finish this part
        # if to_file:
        #     for (var, forcing) in self.forcing.items():
        #         forcing.to_netcdf(path='%s/%s_%s.nc' % (
        #             self.ika_params['forcing_dir'],
        #             self.ika_params['run_name'], var))

    def createFieldSet(self, from_disk: bool = False):
        if from_disk:
            filestem = '%s/%s_*.nc' % (self.ika_params['forcing_dir'],
                                       self.ika_params['run_name'])
            self.ocean = prcl.FieldSet.from_netcdf(
                filestem, variables=self.forcing_vars,
                dimensions=self.forcing.forcing_dims,
                # NOTE : This argument is unexpected
                # deferred_load=False
            )
        else:
            self.ocean = prcl.FieldSet.from_xarray_dataset(
                self.forcing, variables=self.forcing_vars,
                dimensions=self.forcing_dims, allow_time_extrapolation=True)
            self.ocean.add_field(prcl.Field.from_xarray(
                self.landmask, name='landmask', dimensions=self.forcing_dims,
                allow_time_extrapolation=True, interp_method='nearest',))
            if self.ika_params['start_filestem'] is not None:
                self.start_coords = self.start_dist.coords
                self.start_dist = prcl.Field.from_xarray(
                    self.start_dist, name='start_dist',
                    dimensions=self.forcing_dims, interp_method='nearest')
            if self.ika_params['start_cell_lon'] is not None:
                self.start_dist = self.createStartField(self.ika_params['start_cell_lon'],
                                                        self.ika_params['start_cell_lat'])

        #Add necessary field constants
        #(constants easily accessed by particles during kernel execution)
        data_structure = self.forcing_gen.feeding_habitat_structure.data_structure
        if 'NaturalMortality' in self.ika_params['kernels']:
            N_params = self.forcing_gen.readMortalityXML(self.ika_params['SEAPODYM_file'])
            self._setConstant('SEAPODYM_dt',
                              data_structure.parameters_dictionary['deltaT']*24*60*60)
            for (p, val) in N_params.items():
                self._setConstant(p, val)
        if 'Age' in self.ika_params['kernels']:
            self._setConstant('cohort_dt',
                              data_structure.parameters_dictionary['deltaT']*24*60*60)

    def _setConstant(self, name, val):
        self.ocean.add_constant(name, val)

# NOTE : If there is a possibility to start a simulation without start
# argument, maybe it should be switch to optional :
#   start: np.ndarray = None
    def initialiseFishParticles(
            self,start,n_fish=10, pclass:prcl.JITParticle=prcl.JITParticle):

        if isinstance(start, np.ndarray) :
            if start.shape[1] != n_fish :
                raise ValueError('Number of fish and provided initial positions'
                                 ' not equal!')
            self.fish = prcl.ParticleSet.from_list(
                fieldset=self.ocean, lon=start[0],
                time=self.ika_params['start_time'], lat=start[1], pclass=pclass)
        else: # we will distribute according to a start field distribution
            if self.start_dist is None :
                raise ValueError('No starting distribution field in ocean fieldset!')
            plon, plat = self.startDistPositions(n_fish)
            self.fish = prcl.ParticleSet.from_list(
                fieldset=self.ocean, lon=plon,
                time=self.ika_params['start_time'], lat=plat, pclass=pclass)
            #Note previous method created interpolation errors
            #self.fish = prcl.ParticleSet.from_field(
            #    fieldset=self.ocean, start_field=self.start_dist,
            #    time=self.ika_params['start_time'], size=n_fish, pclass=pclass)
                # fieldset=self.ocean, start_field=self.ocean.start,
                # time=None, size=n_fish, pclass=pclass)


        #Initialise fish
        cohort_dt = self.forcing_gen.feeding_habitat_structure.data_structure.\
            species_dictionary['cohorts_sp_unit'][0]

        for f in range(len(self.fish)):
            self.fish[f].age_class = self.start_age
            self.fish[f].age = self.start_age*cohort_dt

        self._setConstant('cohort_dt', cohort_dt)

    def createStartField(self, lon, lat, grid_lon=None, grid_lat=None):
        #Function to create a particle starting distribution around
        #a given cell vertex at a particular resolution
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


    def startDistPositions(self, N, area_scale=True):

        #Simple function returning random particle start positions
        #using the density distribution saved in self.start_dist.
        #Includes option for scaling density by grid cell size (default true)

        self.start_dist.data = self.start_dist.data[0,:,:]
        #Assuming regular grid
        lonwidth = (self.start_dist.grid.lon[1] - self.start_dist.grid.lon[0]) / 2
        latwidth = (self.start_dist.grid.lat[1] - self.start_dist.grid.lat[0]) / 2
        # For distributions from a density on a spherical grid, we need to
        #rescale to a flat mesh
        def cell_area(lat,dx,dy):
            R = 6378.1
            Phi1 = lat*np.pi/180.0
            Phi2 = (lat+dy)*np.pi/180.0
            dx_radian = (dx)*np.pi/180
            S = R*R*dx_radian*(np.sin(Phi2)-np.sin(Phi1))
            return S

        if area_scale:
            for l in range(len(self.start_dist.grid.lat)):
                area = cell_area(self.start_dist.grid.lat[l],lonwidth,latwidth)
                self.start_dist.data[l,:] *= area

        def add_jitter(pos, width, min, max):
            value = pos + np.random.uniform(-width, width)
            while not (min <= value <= max):
                value = pos + np.random.uniform(-width, width)
            return value

        p = np.reshape(self.start_dist.data, (1, self.start_dist.data.size))
        inds = np.random.choice(self.start_dist.data.size, N, replace=True, p=p[0] / np.sum(p))
        lat, lon = np.unravel_index(inds, self.start_dist.data.shape)
        lon = self.ocean.U.grid.lon[lon]
        lat = self.ocean.U.grid.lat[lat]
        for i in range(lon.size):
            lon[i] = add_jitter(lon[i], lonwidth, self.start_dist.grid.lon[0], self.start_dist.grid.lon[-1])
            lat[i] = add_jitter(lat[i], latwidth, self.start_dist.grid.lat[0], self.start_dist.grid.lat[-1])

        return lon, lat

    def fishDensity(self):
        #Function calculating particle density at current time,
        #using the same grid resolution given by the start field coordinates.
        #Method replicates Seapodym biomass density ouputs,
        #returning as an xarray dataarray
        lons = self.start_dist.grid.lon
        lats = self.start_dist.grid.lat
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
                      lons-1, lats-1, D)

        Density = Density[np.newaxis,:,:]
        density_coords = {'time': [np.array(self.ocean.time_origin.fulltime(self.fish.time[0]),dtype=np.datetime64)],
                          'lat': self.start_coords['lat'].data,
                          'lon': self.start_coords['lon'].data}
        PDensity = xr.DataArray(name = "Pdensity",
                            data = Density,
                            coords = density_coords,
                            dims=('time','lat','lon'))
        return PDensity


    # NOTE : Arguments names may be more explicite ? Otherwise a good
    # documentation is needed.
    def runKernels(self, T, pfile_suffix='', verbose=True):
        pfile = self.fish.ParticleFile(
            name=self.ika_params['run_name']+pfile_suffix+'.nc',
            outputdt=self.ika_params['output_dt'])

        Behaviours = [self.fish.Kernel(ika.ikafish.behaviours.AllKernels[b])
                      for b in self.ika_params['kernels']]

        KernelString = ''.join(
            [('Behaviours[{}]+').format(i) for i in range(len(Behaviours))])
        run_kernels = eval(KernelString[:-1])

        self.fish.execute(
            run_kernels, runtime=T, dt=self.ika_params['dt'], output_file=pfile,
            recovery={
                prcl.ErrorCode.ErrorOutOfBounds:ika.ikafish.behaviours.KillFish},
            verbose_progress=verbose)

    def _readParams(self, xml_filepath: str) -> dict :
        """Reads the parameters from a XML parameter file and stores
        them in a dictionary."""

        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        params = {}

        params['run_name'] = root.find('run_name').text
        params['SEAPODYM_file'] = root.find('seapodym_parameters').text
        params['forcing_dir'] = root.find('forcing_dir').text
        params['random_seed'] = root.find('random_seed').text
        if params['random_seed'] == 'None':
            params['random_seed'] = None

        cohort = root.find('cohort_info')
        params['start_length'] = float(cohort.attrib['length'])
        #params['ageing_cohort'] = True if int(cohort.attrib['ageing']) == 1 else False
        # NOTE : Is equivalent to
        params['ageing_cohort'] = int(cohort.attrib['ageing']) == 1
        params['start_cell_lon'] = float(cohort.attrib['start_cell_lon'])
        params['start_cell_lat'] = float(cohort.attrib['start_cell_lat'])
        params['start_filestem'] = (
            params['forcing_dir']
            + cohort.attrib['start_filestem'] if 'start_filestem' in cohort.attrib else None)

        time = root.find('time')
        params['start_time'] = np.datetime64(time.attrib['start'])
        params['T'] = int(time.attrib['sim_time'])
        params['dt'] = int(time.attrib['dt'])*86400
        params['output_dt'] = int(time.attrib['output_dt'])*86400

        domain = root.find('domain')
        params['spatial_lims'] = {
            # 'lonlim': np.float32([domain.find('lon').text.split()[0],
            #                       domain.find('lon').text.split()[1]]),
            # 'latlim': np.float32([domain.find('lat').text.split()[0],
            #                       domain.find('lat').text.split()[1]])}
            # NOTE : Is equivalent to
            'lonlim': np.float32(domain.find('lon').text.split())[:2],
            'latlim': np.float32(domain.find('lat').text.split())[:2],
        }

        params['kernels'] = root.find('kernels').text.split()
        return params
