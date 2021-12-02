import numpy as np
import ikamoana as ika
import parcels as prcl

class IkaSim :
    def __init__(self,
                 xml_parameterfile: str):

        self.ika_params = self._readParams(xml_parameterfile)
        self.habitat_gen = ika.feedinghabitat.FeedingHabitat(xml_filepath=ika_params['forcing_dir']+ika_params['SEAPODYM_file'])
        self.forcing_gen = ika.ikamoanafields.IkamoanaFields(xml_filepath=ika_params['forcing_dir']+xml_feeding_habitat=ika_params['SEAPODYM_file'])
        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

    def generateForcing(self,to_file=False):
        ages = self.habitat_gen.findCohortByLength(self.ika_params['start_length'])
        start = self.habitat_gen.findIndexByDatetime(self.ika_params['start_time'])
        end = self.habitat_gen.findIndexByDatetime(self.ika_params['start_time']+
                                                   self.ika_params['T'])
        lonlims = self.ika_params['spatial_lims']['lonlim']
        latlims = self.ika_params['spatial_lims']['latlim']

        self.forcing = {}
        self.forcing_vars = {}
        if self.ika_params['ageing_cohort']:
            self.forcing['H'] = self.habitat_gen.computeEvolvingFeedingHabitat(
                                                  cohort_start=ages,
                                                  time_start=start,
                                                  time_end=end,
                                                  lon_min=lonlims[0],
                                                  lon_max=lonlims[1],
                                                  lat_min=latlims[0],
                                                  lat_min=latlims[1])
        else:
            self.forcing['H'] = self.habitat_gen.computeFeedingHabitat(
                                                 cohort=ages,
                                                 time_start=start,
                                                 time_end=end,
                                                 lon_min=lonlims[0],
                                                 lon_max=lonlims[1],
                                                 lat_min=latlims[0],
                                                 lat_min=latlims[1])
        self.forcing_vars.update({'H': 'H'})


        ## Waiting for remaining field generation code
        ######## CREATE AND COMPILE TO XARRAY DATASET self.forcing #####
        dHdx, dHdy = self.forcing_gen.gradient(self.forcing['H'],
                                               self.forcing_gen.landmask['H'])
        self.forcing['Tx'], self.forcing['Ty'] = self.forcing_gen.taxis(dHdx, dHdy)
        self.forcing_vars.update({'Tx': 'Tx',
                                  'Ty': 'Ty'})


        if to_file:
            for (var, forcing) in self.forcing.items():
                forcing.to_netcdf(path='%s/%s_%s.nc' % (self.ika_params['forcing_dir'],
                                                  self.ika_params['run_name'], var))
        #Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'lon':'lon',
                             'lat':'lat',
                             'time':'time'}


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
                                       dimensions=self.forcing.coords)


    def initialiseFishParticles(self,pclass=prcl.JITParticle,
                                n_fish,start):

    def _readParams(xml_filepath) :
        """Reads the parameters from a XML parameter file and stores
        them in a dictionary."""
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        params = {}

        params['run_name'] = root.find('run').text
        params['SEAPODYM_file'] = root.find('seapodym_parameters').text
        params['forcing_dir'] = root.find('forcing_dir').text

        params['random_seed'] = root.find('random_seed').text'
        params['random_seed'] = None if params['random_seed'] is 'None'

        cohort = root.find('cohort_info')
        params['start_length'] = cohort.attrib['length']
        params['ageing_cohort'] = True if int(cohort.attrib['ageing']) is 1 else False

        time = root.find['time']
        params['start_time'] = time.attrib['start']
        params['T'] = int(time.attrib['sim_time'])

        domain = root.find('domain')
        params['spatial_lims'] = {'lonlim': [domain.find('lon').text.split()[0],
                                             domain.find('lon').text.split()[1]],
                                  'latlim': [domain.find('lat').text.split()[0],
                                             domain.find('lat').text.split()[1]]}

        return params
