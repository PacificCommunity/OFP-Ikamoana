from .forcingfield import ForcingField
from typing import Union, List
import xarray as xr
import numpy as np


def gradient(field : ForcingField) -> ForcingField :
    """Compute the gradient of a given ForcingField"""
    pass

def vMax() -> Union[float, List[float]] :
    pass

def mortality() -> float:
    pass

def getPopFromDensityField() -> float :
    pass