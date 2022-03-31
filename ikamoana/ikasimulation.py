import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Union
from copy import deepcopy
from functools import reduce

import numpy as np
import parcels
import xarray as xr
from parcels.particle import JITParticle
from ikamoana.utils.feedinghabitatutils import seapodymFieldConstructor
from ikamoana.utils.ikamoanafieldsutils import fieldToDataArray
import os

# Unity used here are seconds and meters.

KernelType = Callable[[JITParticle,parcels.FieldSet,datetime], None]

class IkaSimulation :

    def __init__(self, run_name: str = None, random_seed: float = None):
        """
        `run_name` is used to identify the simulation.
        `random_seed` automatically computed if None."""
        
        if run_name is None :
            run_name = str(uuid.uuid4())[:8]
        if random_seed is None:
            np.random.RandomState()
            random_seed = np.random.get_state()
        
        self.run_name = run_name
        self.random_seed = random_seed
        
    def loadFields(
            self, fields: Union[Dict[str,Union[str,parcels.Field,xr.DataArray]],
                                xr.Dataset,
                                parcels.FieldSet],
            inplace: bool = False, allow_time_extrapolation: bool = True,
            landmask_interp_methode: str = 'nearest',
            fields_interp_method: str = 'nearest', 
            ):
        """`fields` Dict can be field_name:filepath, field_name:DataArray
        or field_name:Field.
        `fields` must contains U and V. If landmask is not inside, it is
        calculated automatically using Nan values of both U and V."""
        
        # NOTE : spatial_limits:Dict is not needed. We consider that all fields 
        # are already at the right size.
        
        def loadFilepaths(fields):
            """In the case where `fields` contains path to NetCDF/DYM
            files, they are loaded before anything else."""
            if isinstance(fields, dict) :
                for k, v in fields.items() :
                    if isinstance(v, str) :
                        fields[k] = seapodymFieldConstructor(v, k)
        
        def checkUVFields(fields):
            """U and V fields must by passed to parcels.FieldSet generator."""
            if isinstance(fields, parcels.FieldSet) :
                if not hasattr(fields, "U") :
                    raise AttributeError("fields argument must contains both U "
                                         "and V but U is missing.")
                if not hasattr(fields, "V") :
                    raise AttributeError("fields argument must contains both U "
                                         "and V but V is missing.")
            else :
                if 'U' not in fields :
                    raise KeyError("fields argument must contains both U and V "
                                   "but U is missing.")
                if 'V' not in fields :
                    raise KeyError("fields argument must contains both U and V "
                                   "but V is missing.")

        def checkLandmask(fields):
            """If `fields` doesn't contain landmask, it is created using
            U and V Nan values."""
            if isinstance(fields, parcels.FieldSet) :
                if not hasattr(fields, "landmask") :
                    landmask = np.full(shape=fields.U.data.shape, fill_value=True)
                    for f in fields.get_fields() :
                        landmask &= np.isfinite(f.data)
                    landmask = parcels.Field(
                        name="landmask", data=landmask, grid=fields.U.grid,
                        interp_method=landmask_interp_methode)
                    fields.add_field(landmask, name="landmask")
                        
            else :
                if 'landmask' not in fields :
                    landmask = np.full(shape=fields['U'].data.shape, fill_value=True)
                    for key in fields :
                        landmask &= np.isfinite(fields[key].data)
                    fields['landmask'] = xr.DataArray(data=landmask,
                                                      coords=fields['U'].coords)
        
        def convertToFieldSet(fields):
            """Convert to Parcels.FieldSet."""
            if isinstance(fields, parcels.FieldSet) :
                return fields
            
            if isinstance(fields, xr.Dataset) :
                landmask = fields["landmask"]
                landmask = parcels.Field.from_xarray(
                    landmask, name="landmask",
                    dimensions={k:k for k in landmask.coords.keys()},
                    interp_method=landmask_interp_methode)
                fields = fields.drop_vars("landmask")
                fields = parcels.FieldSet.from_xarray_dataset(
                    ds=fields, variables={k:k for k in fields.keys()},
                    dimensions={k:k for k in fields.coords.keys()},
                    allow_time_extrapolation=allow_time_extrapolation,
                    interp_method=fields_interp_method)
            
            else :
                landmask = fields.pop("landmask")
                if not isinstance(landmask, parcels.Field) :
                    landmask = parcels.Field.from_xarray(
                        landmask, name="landmask",
                        dimensions={k:k for k in landmask.coords.keys()},
                        interp_method=landmask_interp_methode)
                for k, v in fields.items() :
                    if not isinstance(v, parcels.Field) :
                        fields[k] = parcels.Field.from_xarray(
                            v, name=k, dimensions={k:k for k in v.coords.keys()},
                            allow_time_extrapolation=allow_time_extrapolation,
                            interp_method=fields_interp_method)
                
                fields = parcels.FieldSet(
                    U=fields.pop('U'), V=fields.pop('V'), fields=fields)
            
            fields.add_field(landmask, name='landmask')
            return fields
        
        if not inplace :
            if not isinstance(fields, parcels.FieldSet):
                copied_fields = fields.copy()
            else :
                copied_fields = deepcopy(fields)
        else :
            copied_fields = fields
        
        loadFilepaths(copied_fields)
        checkUVFields(copied_fields)
        checkLandmask(copied_fields)
        self.ocean = convertToFieldSet(copied_fields)
        
    def initializeParticleSet(
            self, particles_longitude:Union[list,np.ndarray],
            particles_latitude:Union[list,np.ndarray],
            particles_class: JITParticle = JITParticle,
            particles_starting_time: Union[np.datetime64,List[np.datetime64]] = None,
            particles_variables: Dict[str,List[Any]] = {}):
        
        # The verification of the sizes of particles_latitude and
        # particles_longitude is done later by parcels.ParticleSet.from_list.
        
        particles_number = len(particles_longitude)
        if particles_variables != {} :
            for k, v in particles_variables.items() :
                if len(v) != particles_number :
                    raise ValueError(("Each list in particles_variables must have "
                                      "a length equal to the number of particles. "
                                      "But {} length is equal to {}"
                                      ).format(k, len(v)))
        
        self.fish = parcels.ParticleSet.from_list(
            fieldset=self.ocean, lon=particles_longitude, lat=particles_latitude,
            pclass=particles_class, time=particles_starting_time, **particles_variables)
    
    def runKernels(
            self, kernels: Union[KernelType, Dict[str, KernelType]],
            recovery: Dict[int, KernelType] = None,
            delta_time: int = 1, duration_time: int = None, end_time: np.datetime64 = None,
            save: bool = False, output_name: str = None, output_delta_time: int = 1,
            verbose: bool = False):
        """
        `recovery` is a dict that contains error code (int) as keys and
        function as values. See also parcels.ErrorCode class to know
        which codes can be handled.
        """

        # The verification of the duration_time and end_time (only one
        # can be chosen) is done later by parcels.ParticleSet.execute.
        
        if save :
            if output_name is None :
                output_name = self.run_name
            particles_file = self.fish.ParticleFile(
                name=("{}.nc").format(output_name), outputdt=output_delta_time)
        else :
            particles_file = None

        if isinstance(kernels, dict) :
            kernels = list(kernels.values())
        if not callable(kernels) : # then it should be a list or tuple
            run_kernels = reduce(lambda a, b : a+b, 
                                 [self.fish.Kernel(k_fun) for k_fun in kernels])

        self.fish.execute(
            run_kernels, endtime=end_time, runtime=duration_time, dt=delta_time,
            recovery=recovery, output_file=particles_file, verbose_progress=verbose)
        
# TOOLS -------------------------------------------------------------- #

    def oceanToNetCDF(self, dir_path: str = None, to_dataset: bool = False):
        
        if dir_path is None :
            dir_path = "."
        dir_path = os.path.join(dir_path,self.run_name)
        try :
            os.mkdir(dir_path)
        except FileExistsError :
            print(("WARNING : The {} directory already exists. The files it "
                   "contains will be replaced.").format(dir_path))
        
        fields_dict = {}
        for field in self.ocean.get_fields() :
            if (not isinstance(field, parcels.VectorField)
                    and not field.name == "start_distribution") :
                print(field.name)
                fields_dict[field.name] = fieldToDataArray(field)
                     
        if to_dataset :
            file_name = "{}.nc".format(self.run_name)
            xr.Dataset(fields_dict).to_netcdf(os.path.join(dir_path, file_name))
        else :
            for field_name, field_value in fields_dict.items() :
                file_name = "{}_{}.nc".format(self.run_name, field_name)
                field_value.to_netcdf(os.path.join(dir_path, file_name))