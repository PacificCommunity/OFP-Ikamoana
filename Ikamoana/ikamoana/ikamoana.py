import xml.etree.ElementTree as ET

import xarray as xr
import numpy as np
import ikamoana as ika
from .ikafish import ikafish, behaviours
import parcels as prcl

# from ..ikamoana.ikamoanafields import IkamoanaFields
# from .ikafish import behaviours, ikafish

class IkaSim :

    def __init__(self, xml_parameterfile: str):

        self.ika_params = self._readParams(xml_filepath=xml_parameterfile)
        self.forcing_gen = ika.ikamoanafields.IkamoanaFields(self.ika_params['SEAPODYM_file'])

        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])


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

        self.forcing['landmask'] = self.forcing_gen.landmask(use_SEAPODYM_global_mask=True,
                                                             field_output=True,
                                                             habitat_field=self.forcing_gen.feeding_habitat)

        self.forcing['H'] = self.forcing_gen.feeding_habitat

        if self.ika_params['start_filestem'] is not None:
            self.forcing['start'] = self.forcing_gen.start_distribution(
                                    self.ika_params['start_filestem']+str(self.start_age)+'.dym')

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


# TODO : WORK IN PROGRESS
# - Reverse dataArray latitude before convert them to Field ? Yes

    def generateForcingNEW(self, to_file=False):

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
            # Mortality is coming soon ---------------------------------
            effort_filepath=None,fisheries_xml_filepath=None, time_reso=None,
            space_reso=None, skiprows=None, removeNoCatch=None, predict_effort=None,
            remove_fisheries=None, convertion_tab=None,
            # ----------------------------------------------------------
            evolve=evolve,
            # NOTE : must select one cohort
            cohort_start= ages[0],
            cohort_end= None,
            time_start=start, time_end=end,
            lon_min=lonlims[0], lon_max=lonlims[1],
            lat_min=latlims[1], lat_max=latlims[0],
        )
        
        # Mortality hasn't the same coordinates as others.
        if 'mortality' in self.forcing.keys():
            mortality = self.forcing.pop('mortality')
        self.forcing = xr.Dataset(self.forcing)

        self.forcing_vars = dict([(i,i) for i in self.forcing.keys()])
        #Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'lon':'lon', 'lat':'lat', 'time':'time'}

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
            landmask = self.forcing.pop('landmask')
            self.forcing_vars.pop('landmask')
            self.ocean = prcl.FieldSet.from_xarray_dataset(self.forcing,
                                       variables=self.forcing_vars,
                                       dimensions=self.forcing_dims,
                                       deferred_load=False)
            self.ocean.add_field(prcl.Field.from_netcdf(None, landmask,
                                                        var_name='landmask',
                                                        dimensions= {'lon':'lon',
                                                                     'lat':'lat',
                                                                     'time':'time'},
                                                        allow_time_extrapolation=True,
                                                        interp_method='nearest',
                                                        deferred_load=False))

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
    # NOTE : Can use isinstance() function
        #if type(start) is np.ndarray:
        if isinstance(start, np.ndarray) :
    # NOTE : raise exception rather than assert for a better readability
            if start.shape[1] != n_fish :
                raise ValueError('Number of fish and provided initial positions'
                                 ' not equal!')
            self.fish = prcl.ParticleSet.from_list(
                fieldset=self.ocean, lon=start[0],
                time=self.ika_params['start_time'], lat=start[1], pclass=pclass)
        else:
            if self.ocean.start is None :
                raise ValueError('No starting distribution field in ocean fieldset!')
            self.fish = prcl.ParticleSet.from_field(
                fieldset=self.ocean, start_field=self.ocean.start,
                time=self.ika_params['start_time'], size=n_fish, pclass=pclass)

        #Initialise fish
        cohort_dt = self.forcing_gen.feeding_habitat_structure.data_structure.\
            species_dictionary['cohorts_sp_unit'][0]

        # NOTE : This wont work if the pclass is wrong. Should test with
        # isinstance ?
        for f in range(len(self.fish.particles)):
            self.fish.particles[f].age_class = self.start_age
            self.fish.particles[f].age = self.start_age*cohort_dt

        self._setConstant('cohort_dt', cohort_dt)

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
