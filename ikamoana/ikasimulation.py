import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Union
from copy import deepcopy
from functools import reduce

import numpy as np
import parcels
import xarray as xr
from parcels.particle import JITParticle
from ikamoana.utils import seapodymFieldConstructor
from ikamoana.utils import convertToDataArray
from ikamoana.ikafish import behaviours
import os

# NOTE : Unity used here are seconds and meters.

KernelType = Callable[[JITParticle,parcels.FieldSet,datetime], None]

class IkaSimulation :
    """
    Encapsulates the simulation methods of the Parcels library. Is used
    as a template by the IkaSeapodym class.

    Attributes
    ----------
    run_name : str
        The name of the simulation.
    random_seed : float
        Seed to generate random values.
    ocean : parcels.FieldSet
        Contains all the fields necessary for the simulation.
    fish : parcels.ParticleSet
        Contains the state of all particles.

    Examples
    --------
    First example : Simply load the U and V fields followed by running
    the AdvectionRK4 plot method on a single particle. Finally, export
    the fields to a Dataset for later use.

    >>> my_sim = ikasim.IkaSimulation()
    >>> my_dataset = xr.Dataset({U:"U.nc", "V":"V.nc"})
    >>> my_sim.loadFields(fields=my_dataset)
    >>> my_sim.initializeParticleSet(
    ...     particles_longitude=[150],
    ...     particles_latitude=[10],
    ...     particles_class=parcels.particle.JITParticle,
    ...     particles_starting_time=np.datetime64('1979-01-15'),
    ...     particles_variables={"age":[0]})
    >>> my_sim.fish.show(field=my_sim.ocean.U)
    >>> my_sim.runKernels(
    ...     {"AdvectionRK4":parcels.AdvectionRK4},
    ...     # Duration in seconds (= 30 days)
    ...     duration_time=30*(24*60*60),
    ...     # Can also use a end date
    ...     #end_time=np.datetime64('1980-02-15'),
    ...     delta_time=24*60*60,
    ...     # Record day after day
    ...     output_delta_time=24*60*60)
    >>> my_sim.oceanToNetCDF(to_dataset=True)
    """

    def __init__(self, run_name: str = None, random_seed: float = None):
        """Initialize an IkaSimulation object.

        Parameters
        ----------
        run_name : str, optional
           It is used to identify the simulation. Automatically
           generated if None.
        random_seed : float, optional
            Automatically generated if None.
        """

        if run_name is None :
            run_name = str(uuid.uuid4())[:8]
        if random_seed is None:
            np.random.RandomState()
            random_seed = np.random.get_state()

        self.run_name = run_name
        self.random_seed = random_seed

    def _addField(
            self, field, dims={"time":"time","lat":"lat","lon":"lon"},
            name=None, time_extra=True, interp_method='nearest'
            ):
        """Add a field to the self ocean attribut."""

        self.ocean.add_field(parcels.Field.from_xarray(
            field, name=field.name if name is None else name,
            dimensions=dims, allow_time_extrapolation=time_extra,
            interp_method=interp_method))

    def loadFields(
            self, fields: Union[xr.Dataset, parcels.FieldSet,
                                Dict[str,Union[str,parcels.Field,xr.DataArray]]],
            inplace: bool = False, allow_time_extrapolation: bool = True,
            landmask_interp_methode: str = 'nearest',
            fields_interp_method: str = 'nearest',
            ):
        """Loads all the fields needed for the simulation into a
        Parcels.FieldSet structure.

        Parameters
        ----------
        fields : Union[xr.Dataset, parcels.FieldSet, Dict[str,Union[str,parcels.Field,xarray.DataArray]]]
            It must contains U and V. If  the "landmask" is not inside,
            it is calculated automatically using the Nan values of both
            U and V. If `fields` is a Dict, keys are fields name and
            values can be Fields, DataArray or filepath (str).
        inplace : bool, optional
            Choose if this function manipulates a copy of the `fields`
            structure or the structure itself.
        allow_time_extrapolation : bool, optional
            This is a Parcels parameter passed at FieldSet creation.
            Please refer to the Parcels documentation.
        landmask_interp_methode : str, optional
            Interpolation method used to create the parcels landmask.
            Please refer to the Parcels documentation.
        fields_interp_method : str, optional
            Interpolation method used to create the parcels FieldSet.
            Please refer to the Parcels documentation.

        Raises
        ------
        AttributeError
            `fields` (Dataset/FieldSet) argument must contains both U
            and V but U or V are missing.
        KeyError
            `fields` (Dict[str,DataArray/Field]) argument must contains
            both U and V but U or V are missing.
        """

        # NOTE : spatial_limits:Dict is not needed. We consider that
        # all fields are already at the right size.

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
        """Initialise the ParticleSet (`fish` attribut) from lists of
        longitude and latitude.

        Parameters
        ----------
        particles_longitude : Union[list,np.ndarray]
            The longitudinal position of the particles.
        particles_latitude : Union[list,np.ndarray]
            The latitudinal position of the particles.
        particles_class : JITParticle, optional
            The class of particles to be used in this simulation.
        particles_starting_time : Union[np.datetime64,List[np.datetime64]], optional
            Optional list of start time values for particles. Default is
            `ocean.U.time[0]`.
        particles_variables : Dict[str,List[Any]], optional
            Variables to add to particles. {variable name : list of
            values for each particle}.

        Raises
        ------
        ValueError
            Each list in particles_variables must have a length equal to
            the number of particles.
        """

        # NOTE : The verification of the sizes of particles_latitude and
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
            sample_kernels: list = [],
            delta_time: int = 1, duration_time: int = None, end_time: np.datetime64 = None,
            save: bool = False, output_name: str = None, output_delta_time: int = 1,
            verbose: bool = False):
        """Execute a given kernel (or a list of kernels) function(s)
        over the particle set for multiple timesteps. Optionally also
        provide sub-timestepping for particle output.

        Parameters
        ----------
        kernels : Union[KernelType, Dict[str, KernelType]]
            Kernel function to execute. This can be the name of a
            defined Python function or a parcels.kernel.Kernel object.
            If you define multiple kernels in a dict, they will be
            concatenated using the + operator.
        recovery : Dict[int, KernelType], optional
            Dictionary with additional `parcels.tools.error` recovery
            kernels to allow custom recovery behaviour in case of kernel
            errors.
        delta_time : int, optional
            It is either a timedelta object or a double. Use a negative
            value for a backward-in-time simulation.
        duration_time : int, optional
            Length of the timestepping loop. Use instead of endtime. It
            is either a timedelta object or a positive double.
        end_time : np.datetime64, optional
            End time for the timestepping loop. It is either a datetime
            object or a positive double.
        save : bool, optional
            Specify if you want to save particles history into a NetCDF
            file.
        output_name : str, optional
            Name of the `parcels.particlefile.ParticleFile` object from
            the ParticleSet. Default is then `run_name`.
        output_delta_time : int, optional
            Interval which dictates the update frequency of file output.
            It is either a timedelta object or a positive double.
        verbose : bool, optional
            Boolean for providing a progress bar for the kernel
            execution loop.
        """

        # NOTE : The verification of the duration_time and end_time
        # (only one can be chosen) is done later by parcels.ParticleSet.execute.

        if save :
            if output_name is None :
                output_name = "{}_particleFile".format(self.run_name)
            else :
                output_name, _ = os.path.splitext(output_name)
            particles_file = self.fish.ParticleFile(
                name=("{}.nc").format(output_name), outputdt=output_delta_time)
        else :
            particles_file = None

        for sk in sample_kernels:
            if sk in behaviours.AllKernels :
                sampler = behaviours.AllKernels[sk]
                self.fish.execute(self.fish.Kernel(sampler),dt=0)
            else :
                raise ValueError(("{} kernel is not defined by "
                                  "behaviours.AllKernels.").format(sk))


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
        """Export the `ocean` FieldSet into a single NetCDF (as a Dataset)
        or multiple NetCDF (as DataArrays).

        Parameters
        ----------
        dir_path : str, optional
            The directory where Fields will be saved. Created if it
            doesn't exist.
        to_dataset : bool, optional
            Save as one Dataset (True) or multiple DataArrays (False).
        """

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
                fields_dict[field.name] = convertToDataArray(field)

        if to_dataset :
            file_name = "{}.nc".format(self.run_name)
            xr.Dataset(fields_dict).to_netcdf(os.path.join(dir_path, file_name))
        else :
            for field_name, field_value in fields_dict.items() :
                file_name = "{}_{}.nc".format(self.run_name, field_name)
                field_value.to_netcdf(os.path.join(dir_path, file_name))
