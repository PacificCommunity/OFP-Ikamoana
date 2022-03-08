from typing import Union, Dict, Any
import xarray as xr
import numpy as np
import parcels

# Unity used here are seconds and meters.

class IkaSimulation :

    def __init__(self, run_name:str, start_time:str, random_seed: float = None):
        """
        `run_name` is used to identify the simulation.
        `start_time` will be used to initialize particles."""
        
        pass

    def loadFields(
            self, fields: Union[Dict[str,Union[parcels.Field,xr.DataArray]],
                                xr.Dataset,
                                parcels.FieldSet]
            ):
        # NOTE : spatial_limits:Dict is not needed. We consider that all fields 
        # are already at the right size.
        
        def checkUVFields(fields):
            """U and V fields must by passed to parcels.FieldSet generator."""
            pass
        
        def convertToFields(fields):
            """Convert currents and landmask to parcels.Field."""
            pass
        
        checkUVFields(fields)
        
        U, V, landmask = convertToFields(fields)
        
        # self.ocean = ...
        
        pass
    
    def initializeParticleSet(
            self, particles_longitude:Union[list,np.ndarray],
            particles_latitude:Union[list,np.ndarray],
            particles_class: parcels.JITParticle = parcels.JITParticle,
            particles_attributs: Dict[str,Any] = None):
        pass
    
    def runKernels(
            self, kernels: Dict[str, function], delta_time: int = 1,
            duration_time: int = None, output_name: str = None,
            output_delta_time: int = None):
        pass