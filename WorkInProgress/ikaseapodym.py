from typing import Union, Dict, Any
import xarray as xr
import numpy as np
import parcels

from .ikasimulation import IkaSimulation

class IkaSeapodym(IkaSimulation) :
    
    def __init__(self, filepath: str):
        """Overrides `IkaSimulation.__init__()` by first reading a
        configuration file and then passing it all parameters."""
        
        parameter_file_struct = self._readConfigFile(filepath)
        
        super().__init__(parameter_file_struct.pop("run_name"),
                         parameter_file_struct.pop("start_time"),
                         parameter_file_struct.pop("random_seed"))
        
        # Add other parameters
        
        pass
    
    def _readConfigFile(self, filepath:str) -> dict :
        """Reads a configuration file and returns a dictionary with all
        the parameters necessary for the initialization of this class."""
        pass
    
    def loadFields(self):
        
        # Creates fields using IkaFields class or read it from directory
        # filepath + run name. Then it creates a FieldSet using :
        # super().loadFields(U,V,landmask)
        # 
        # NOTE : Should we use an argument from_file or a parameter in
        # the XML configuration file ?
        
        pass
    
    def initializeParticleSet(
            self, particles_longitude:Union[list,np.ndarray],
            particles_latitude:Union[list,np.ndarray],
            particles_class: parcels.JITParticle = parcels.JITParticle,
            particles_attributs: Dict[str,Any] = None):
        
        # This part depends on what we want to do. It can stay unchanged
        # or be override. For example, IkaTag will need a specific
        # method to create multiple releases in same place.
        # It can also has different behaviours according to the particle
        # class used to initialize particles (Tag, Mix, etc...).
        
        pass
    
    def runKernels(self, kernels: Dict[str, function]):
        
        # All kernels here are already writed in an other file
        # (behaviours). Configuration file will specify the ones we want
        # to use in this simulation.
        
        pass