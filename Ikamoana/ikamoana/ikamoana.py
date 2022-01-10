import numpy as np
import ikamoana as ika
from .ikafish import ikafish, behaviours
import parcels as prcl
import xml.etree.ElementTree as ET

class IkaSim :

    def __init__(self,
                 xml_parameterfile: str):
        self.ika_params = self._readParams(xml_filepath=xml_parameterfile)
        self.forcing_gen = ika.ikamoanafields.IkamoanaFields(xml_feeding_habitat=self.ika_params['forcing_dir']+self.ika_params['SEAPODYM_file'])
        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

    def generateForcing(self,to_file=False):
        ages = self.forcing_gen.feeding_habitat_structure.data_structure.findCohortByLength(self.ika_params['start_length'])
        start = self.forcing_gen.feeding_habitat_structure.data_structure.findIndexByDatetime(self.ika_params['start_time'])[0]
        end = self.forcing_gen.feeding_habitat_structure.data_structure.findIndexByDatetime(self.ika_params['start_time']+
                                                                             self.ika_params['T'])[0]
        lonlims = self.ika_params['spatial_lims']['lonlim']
        lonlims = self.forcing_gen.feeding_habitat_structure.data_structure.findCoordIndexByValue(lonlims,coord='lon')
        lonlims = [int(l) for l in lonlims]
        latlims = self.ika_params['spatial_lims']['latlim']
        latlims = self.forcing_gen.feeding_habitat_structure.data_structure.findCoordIndexByValue(latlims,coord='lat')
        latlims = [int(l) for l in latlims]

        self.forcing = {}
        self.forcing_vars = {}

        if self.ika_params['ageing_cohort']:
            self.forcing['Tx'], self.forcing['Ty'] = self.forcing_gen.computeEvolvingTaxis(cohort_start=ages[0],
                                                                       time_start=start,
                                                                       time_end=end,
                                                                       lon_min=lonlims[0],
                                                                       lon_max=lonlims[1],
                                                                       lat_min=latlims[1],
                                                                       lat_max=latlims[0])
        else:
            self.forcing['Tx'], self.forcing['Ty'] = self.forcing_gen.computeTaxis(cohort=ages,
                                                               time_start=start,
                                                               time_end=end,
                                                               lon_min=lonlims[0],
                                                               lon_max=lonlims[1],
                                                               lat_min=latlims[1],
                                                               lat_max=latlims[0])
        self.forcing_vars.update({'Tx': 'Tx',
                                  'Ty': 'Ty'})

        ### Mortality fields to do

        self.forcing['U'], self.forcing['V'] = self.forcing_gen.current_forcing()
        self.forcing_vars.update({'U': 'U',
                                  'V': 'V'})

        self.forcing['K'] = self.forcing_gen.diffusion(self.forcing_gen.feeding_habitat)
        self.forcing_vars.update({'K': 'K'})
        self.forcing['dK_dx'], self.forcing['dK_dy'] = self.forcing_gen.gradient(self.forcing['K'],
                                                          self.forcing_gen.landmask(self.forcing_gen.feeding_habitat,
                                                          lon_min=lonlims[0],
                                                          lon_max=lonlims[1],
                                                          lat_min=latlims[1],
                                                          lat_max=latlims[0]),
                                                          name='K')
        self.forcing_vars.update({'dK_dx': 'dK_dx',
                                  'dK_dy': 'dK_dy'})

        if to_file:
            for (var, forcing) in self.forcing.items():
                forcing.to_netcdf(path='%s/%s_%s.nc' % (self.ika_params['forcing_dir'],
                                                  self.ika_params['run_name'], var))
        #Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'lon':'lon',
                             'lat':'lat',
                             'time':'time'}
        self.start_age = ages[0]


    def createFieldSet(self,from_disk=False):
        if from_disk:
            filestem = '%s/%s_*.nc' % (self.ika_params['forcing_dir'],
                                       self.ika_params['run_name'])
            self.ocean = prcl.FieldSet.from_netcdf(filestem,
                                                   variables=self.forcing_vars,
                                                   dimensions=self.forcing.coords)
        else:
            self.ocean = prcl.FieldSet.from_xarray_dataset(self.forcing,
                                       variables=self.forcing_vars,
                                       dimensions=self.forcing_dims)
        print(self.ocean.Tx.grid.time_origin)
        print(self.ocean.Tx.grid.time_origin.fulltime(self.ocean.Tx.grid.time[0]))

    def _setConstant(self, name, val):
        self.ocean.add_constant(name, val)

    def initialiseFishParticles(self,start,n_fish=10,pclass=prcl.JITParticle):
        if type(start) is np.ndarray:
            assert start.shape[1] == n_fish, 'Number of fish and provided initial positions not equal!'
            self.fish = prcl.ParticleSet.from_list(fieldset=self.ocean,
                                     lon=start[0],
                                     time=self.ika_params['start_time'],
                                     lat=start[1],
                                     pclass=pclass)
        else:
            assert self.ocean.start is not None, 'No starting distribution field in ocean fieldset!'
            self.fish = prcl.ParticleSet.from_field(fieldset=ocean,
                                      start=self.ocean.start,
                                      time=self.ika_params['start_time'],
                                      size=n_fish,
                                      pclass=pclass)
        #Initialise fish
        cohort_dt = self.forcing_gen.feeding_habitat_structure.data_structure.species_dictionary['cohorts_sp_unit'][0]
        for f in range(len(self.fish.particles)):
            self.fish.particles[f].age_class = self.start_age
            self.fish.particles[f].age = self.start_age*cohort_dt
        self._setConstant('cohort_dt', cohort_dt)


    def runKernels(self,T,pfile_suffix='',verbose=True):
        pfile = self.fish.ParticleFile(name=self.ika_params['run_name']+pfile_suffix+'.nc',
                                       outputdt=self.ika_params['output_dt'])


        Behaviours = [self.fish.Kernel(ika.ikafish.behaviours.AllKernels[b]) for b in self.ika_params['kernels']]
        KernelString = sum([['Behaviours[',str(i),']','+'] for i in range(len(Behaviours))], [])[:-1]
        KernelString = ''.join(KernelString)
        run_kernels = eval(KernelString)

        self.fish.execute(run_kernels, runtime=T,
                     dt=self.ika_params['dt'], output_file=pfile,
                     recovery={prcl.ErrorCode.ErrorOutOfBounds:
                                ika.ikafish.behaviours.KillFish},
                     verbose_progress=verbose)


    def _readParams(self,xml_filepath) :
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
        params['ageing_cohort'] = True if int(cohort.attrib['ageing']) is 1 else False

        time = root.find('time')
        params['start_time'] = np.datetime64(time.attrib['start'])
        params['T'] = int(time.attrib['sim_time'])
        params['output_dt'] = int(time.attrib['dt'])
        params['dt'] = int(time.attrib['output_dt'])

        domain = root.find('domain')
        params['spatial_lims'] = {'lonlim': np.float32([domain.find('lon').text.split()[0],
                                             domain.find('lon').text.split()[1]]),
                                  'latlim': np.float32([domain.find('lat').text.split()[0],
                                             domain.find('lat').text.split()[1]])}

        params['kernels'] = root.find('kernels').text.split()
        return params
