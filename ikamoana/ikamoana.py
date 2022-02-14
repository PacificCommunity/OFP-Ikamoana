import os
import xml.etree.ElementTree as ET

import numpy as np
import parcels as prcl
import xarray as xr

from .ikafish import behaviours, ikafish
from .ikamoanafields import IkamoanaFields


class IkaSim :

    def __init__(self, xml_parameterfile: str):

        self.ika_params = self._readParams(xml_filepath=xml_parameterfile)
        self.forcing_gen = IkamoanaFields(xml_parameterfile)

        if self.ika_params['random_seed'] is None:
            np.random.RandomState()
            self.ika_params['random_seed'] = np.random.get_state()
        else:
            np.random.RandomState(self.ika_params['random_seed'])

# -------------------------------------------------------------------- #

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
        params['start_filestem'] = (params['start_distribution']
                                    + cohort.find('start_filestem').text)

        domain = root.find('domain')
        
        time = domain.find('time')
        params['start_time'] = np.datetime64(time.find('start').text)
        params['T'] = int(time.find('sim_time').text)
        params['dt'] = int(time.find('dt').text)*86400
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
                                    'Ms_mean_max').attrib[species_name]),
                   'Mrange': float(root.find(
                                    'Ms_mean_max').attrib[species_name])}

        return n_param

    def _setConstant(self, name, val):
        self.ocean.add_constant(name, val)

# -------------------------------------------------------------------- #

    def generateForcing(self, from_habitat=None, to_file=False):

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
            from_habitat=from_habitat,
            evolve=evolve,
            # NOTE : must select one cohort
            cohort_start= ages[0],
            cohort_end= None,
            time_start=start, time_end=end,
            lon_min=lonlims[0], lon_max=lonlims[1],
            lat_min=latlims[1], lat_max=latlims[0],
        )

        # TODO : Use latitudeDirection to ensure that the latitude is
        # from south to north?
        if self.ika_params['start_filestem'] is not None:
            self.start_distribution = self.forcing_gen.start_distribution(
                self.ika_params['start_filestem']+str(self.start_age)+'.dym')

        # Landmask hasn't the same coordinates as others.
        self.landmask = self.forcing.pop('landmask')

        # TODO : We will not use Dataset because it shared coordinates
        # among all DataArray. 
        self.forcing = xr.Dataset(self.forcing)

        self.forcing_vars = dict([(i,i) for i in self.forcing.keys()])
        # Parcels will need a mapping of dimension coordinate names
        self.forcing_dims = {'time':'time', 'lat':'lat', 'lon':'lon'}

        if to_file:
            for (var, forcing) in self.forcing.items():
                forcing.to_netcdf(
                    path=os.path.join(self.ika_params['forcing_dir'],
                                      self.ika_params['run_name']+'_'+var+'.nc'))
            self.start_distribution.to_netcdf(
                os.path.join(self.ika_params['forcing_dir'],
                             self.ika_params['run_name']+'_start_distribution.nc'))
            self.landmask.to_netcdf(
                os.path.join(self.ika_params['forcing_dir'],
                             self.ika_params['run_name']+'_landmask.nc'))
            self.mortality.to_netcdf(
                os.path.join(self.ika_params['forcing_dir'],
                             self.ika_params['run_name']+'_mortality.nc'))

    # TODO : Add mortality
    def createFieldSet(self, from_disk: bool = False, variables: dict = None):
        """[summary]

        Parameters
        ----------
        from_disk : bool, optional
            [description], by default False
        variables : dict, optional
            If None, names are automaticly created using `forcing_dir`
            and `run_name`.
            
            Example : 
                variables = {
                    "dK_dx":"<run_name>_dK_dx.nc",
                    "dK_dy":"<run_name>_dK_dy.nc",
                    "H":"<run_name>_H.nc",
                    "K":"<run_name>_K.nc",
                    "landmask":"<run_name>_landmask.nc",
                    "start_distribution":"<run_name>start_distribution.nc",
                    "Tx":"<run_name>_Tx.nc",
                    "Ty":"<run_name>_Ty.nc",
                    "U":"<run_name>_U.nc",
                    "V":"<run_name>_V.nc"}
        """

        if from_disk:
            if variables is None :
                list_var = ["dK_dx", "dK_dy", "H", "K", "landmask", "mortality",
                            "start_distribution", "Tx", "Ty", "U", "V"]
                variables = {
                    var: os.path.join(self.ika_params['forcing_dir'],
                                      self.ika_params['run_name']+'_'+var+'.nc')
                    for var in list_var}
            else :
                variables = {k: os.path.join(self.ika_params['forcing_dir'],v)
                            for k, v in variables.items()}
            self.ocean = prcl.FieldSet.from_netcdf(
                variables,
                {k:k for k in variables.keys()},
                {'time':'time', 'lat':'lat', 'lon':'lon'}
                )
        else:
            self.ocean = prcl.FieldSet.from_xarray_dataset(
                self.forcing, variables=self.forcing_vars,
                dimensions=self.forcing_dims)
            self.ocean.add_field(prcl.Field.from_xarray(
                self.landmask, name='landmask', dimensions=self.forcing_dims,
                allow_time_extrapolation=True, interp_method='nearest'))
            self.start_distribution = prcl.Field.from_xarray(
                self.start_distribution, name='start_distribution', dimensions=self.forcing_dims)

        #Add necessary field constants
        #(constants easily accessed by particles during kernel execution)
        data_structure = self.forcing_gen.feeding_habitat_structure.data_structure
        if 'NaturalMortality' in self.ika_params['kernels']:
            N_params = self._readMortalityXML(self.ika_params['seapodym_file'])
            self._setConstant('SEAPODYM_dt',
                              data_structure.parameters_dictionary['deltaT']*24*60*60)
            for (p, val) in N_params.items():
                self._setConstant(p, val)
        if 'Age' in self.ika_params['kernels']:
            self._setConstant('cohort_dt',
                              data_structure.parameters_dictionary['deltaT']*24*60*60)

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
        else:
            if self.start_distribution is None :
                raise ValueError('No starting distribution field in ocean fieldset!')
            self.fish = prcl.ParticleSet.from_field(
                fieldset=self.ocean, start_field=self.start_distribution,
                time=self.ika_params['start_time'], size=n_fish, pclass=pclass)
                # fieldset=self.ocean, start_field=self.ocean.start,
                # time=None, size=n_fish, pclass=pclass)


        #Initialise fish
        cohort_dt = self.forcing_gen.feeding_habitat_structure.data_structure.\
            species_dictionary['cohorts_sp_unit'][0]

        for f in range(len(self.fish)):
            self.fish[f].age_class = self.start_age
            self.fish[f].age = self.start_age*cohort_dt

        self._setConstant('cohort_dt', cohort_dt)

# NOTE : Arguments names may be more explicite (T) ? Uppercases are
# usualy for constants or classes.
    def runKernels(self, T, pfile_suffix='', verbose=True):
        pfile = self.fish.ParticleFile(
            name=self.ika_params['run_name']+pfile_suffix+'.nc',
            outputdt=self.ika_params['output_dt'])

        Behaviours = [self.fish.Kernel(behaviours.AllKernels[b])
                      for b in self.ika_params['kernels']]

        KernelString = ''.join(
            [('Behaviours[{}]+').format(i) for i in range(len(Behaviours))])
        run_kernels = eval(KernelString[:-1])

        self.fish.execute(
            run_kernels, runtime=T, dt=self.ika_params['dt'], output_file=pfile,
            recovery={
                prcl.ErrorCode.ErrorOutOfBounds:behaviours.KillFish},
            verbose_progress=verbose)



