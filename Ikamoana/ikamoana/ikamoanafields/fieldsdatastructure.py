from typing import List, Union
import xml.etree.ElementTree as ET
from os.path import dirname
import numpy as np

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
    return default if elmt == '' else elmt



class FieldsDataStructure :

    def __init__(
            self, IKAMOANA_config_filepath: str, root_directory: str = None,
            SEAPODYM_config_filepath: str = None
            ):
        """Create a data structure to store all informations needed to
        compute the IKAMOANA fields.

        Parameters
        ----------
        IKAMOANA_config_filepath : str
            Path to the IKAMOANA configuration XML file.
        root_directory : str, optional
            If the SEAPODYM configuration file is not in the root of the
            working directory, this directory path must be specified.
        SEAPODYM_config_filepath : str, optional
            SEAPODYM configuration filepath can also be specified by
            user.
        """        

        # IKAMOANA --------------------------------------------------- #
        tree = ET.parse(IKAMOANA_config_filepath)
        root = tree.getroot()
        
        if SEAPODYM_config_filepath is None :
            SEAPODYM_config_filepath = root.find('seapodym_parameters').text
        self.SEAPODYM_config_filepath = SEAPODYM_config_filepath
        
        self.diffusion_boost=float(tagReading(
            root,['forcing','diffusion_boost'],0))
        self.diffusion_scale=float(tagReading(root,['forcing','diffusion_scale'],1))
        self.sig_scale=float(tagReading(root,['forcing','sig_scale'],1))
        self.c_scale=float(tagReading(root,['forcing','c_scale'],1))
        self.taxis_scale=float(tagReading(root,['forcing','taxis_scale'],1))
        self.units=tagReading(root,['forcing','units'],'m_per_s')
        
        tmp = tagReading(root,['forcing','shallow_sea_to_ocean'], 'False')
        self.shallow_sea_to_ocean = (tmp == 'True') or (tmp == 'true')
        tmp = tagReading(root,['forcing','landmask_from_habitat'], 'False')
        self.landmask_from_habitat = (tmp == 'True') or (tmp == 'true')
        """Specify if the landmask is based on the SEAPODYM mask used
        to compute feeding habitat or not."""


        # SEAPODYM --------------------------------------------------- #
        tree = ET.parse(SEAPODYM_config_filepath)
        root = tree.getroot()
        
        if root_directory is None :
            prefix = dirname(SEAPODYM_config_filepath)
            # Support windows paths using backslash ("\")
            prefix += "\\" if "\\" in prefix else "/"
        else :
            prefix = dirname(SEAPODYM_config_filepath) if root_directory is None else root_directory
        root_directory = prefix + root.find('strdir').attrib['value']
        sp_name = root.find('sp_name').text
        deltaT = float(root.find('deltaT').attrib["value"])
        
        self.timestep=deltaT*24*60*60
        
        ## TAXIS ####################################
        self.vmax_a=float(root.find('MSS_species').attrib[sp_name])
        """SEAPODYM name is MSS_species. Velocity at maximal habitat
        gradient and `A = 1, BL/s`."""
        self.vmax_b=float(root.find('MSS_size_slope').attrib[sp_name])
        """SEAPODYM name is MSS_size_slope. Slope coefficient in
        allometric function for tuna velocity."""
        
        ## CURRENTS #################################
        # TODO : For now, only the first layer is used.
        self.u_file = 'po_interim_historic_2x30d_u_L1_1979_2010.dym'
        self.v_file = 'po_interim_historic_2x30d_v_L1_1979_2010.dym'
        
        ## DIFFUSION ####################################
        self.sigma_K=float(root.find('sigma_species').attrib[sp_name])
        """SEAPODYM name is sigma_species. Multiplier for the theoretical
        diffusion rate `((V̄**2) * ∆T)/4`"""
        self.c=float(root.find('c_diff_fish').attrib[sp_name])
        """SEAPODYM name is c_diff_fish. Coefficient of diffusion
        variability with habitat index."""
        self.P=3
        """The constant (`p=3`) is chosen to limit the reduction of `D0` in
        the lowest habitat indices `Ha < 0.5`."""

