import xml.etree.ElementTree as ET
from os.path import exists
from typing import List, Union

import numpy as np
import xarray as xr

from .. import dymfiles as df


def seapodymFieldConstructor(
        filepath: str, dym_varname : str = None, dym_attributs : dict = None
        ) -> xr.DataArray :
    """
    Return a Seapodym field as a DataArray using NetCDF or Dym method
    according to the file extension : 'nc', 'cdf' or 'dym'.

    Parameters
    ----------
    filepath : str
        The path to the NetCDF or DYM.
    dym_varname : str, optional
        If the file is a DYM, dym_varname is the name of the variable
        represented inside the file. By default None which is replaced
        by the filepath.
    dym_attributs : str, optional
        If the file is a DYM, dym_attributs is , by default None

    Returns
    -------
    xr.DataArray
        [description]
    """
    if exists(filepath) :
        #NetCDF
        if filepath.lower().endswith(('.nc', '.cdf')) :
            return xr.open_dataarray(filepath)
        #DymFile
        if filepath.lower().endswith('.dym') :
            if dym_varname is None :
                dym_varname = filepath
            return df.dym2ToDataArray(infilepath = filepath,
                                      varname = dym_varname,
                                      attributs = dym_attributs)
    else :
        raise ValueError("No such file : {}".format(filepath))
    
def tagReading(
        root: ET.Element, tags: Union[str,List[str]],
        default: Union[str,int,float] = None, attribute: str = None
        ) -> Union[int,float,str]:
    """Move through a chain of XML `tags` to read a parameter. Return
    `default` value if this parameter `text` (or a specific `attribute`)
    is empty."""
    
    tags = np.ravel(tags)
    elmt = root.find(tags[0])
    for tag_name in tags[1:]:
        elmt = elmt.find(tag_name)
        
    elmt = elmt.text if attribute is None else elmt.attrib[attribute]
    return default if (elmt == '') or (elmt is None) else elmt
