import numpy as np
import ikamoana as ika
import parcels as prcl

class IkaSim :
    def __init__(self,
                 xml_parameterfile: str):

        self.ika_params = self._readParams(xml_parameterfile)
        self.forcing_gen = ika.feedinghabitat.FeedingHabitat(ika_params['SEAPODYM_file'])
        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

    def generateForcing(self,to_file=False):
        ages = self.forcing_gen.findCohortByLength(ika_params['start_length'])
        start = self.forcing_gen.findIndexByDatetime(self.ika_params['start_time'])
        end = self.forcing_gen.findIndexByDatetime(self.ika_params['start_time']+
                                                   self.ika_params['T'])
        lonlims = self.ika_params['spatial_lims']['lonlim']
        latlims = self.ika_params['spatial_lims']['latlim']

        self.forcing = {}
        self.forcing_vars = {}
        if self.ika_params['ageing_cohort']:
            self.forcing['H'] = self.forcing_gen.computeEvolvingFeedingHabitat(
                                                  cohort_start=ages,
                                                  time_start=start,
                                                  time_end=end,
                                                  lon_min=lonlims[0],
                                                  lon_max=lonlims[1],
                                                  lat_min=latlims[0],
                                                  lat_min=latlims[1])
        else:
            self.forcing['H'] = self.forcing_gen.computeFeedingHabitat(
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

        if to_file:
            self.H.to_netcdf(path='%s/%s_H.nc' % (self.ika_params['forcing_dir'],
                                                  self.ika_params['run_name']))
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

    
