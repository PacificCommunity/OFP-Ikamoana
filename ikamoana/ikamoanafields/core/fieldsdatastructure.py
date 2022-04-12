import xml.etree.ElementTree as ET
import numpy as np
import os 

from ikamoana.utils.ikamoanafieldsutils import tagReading

class IkamoanaFieldsDataStructure :

    def __init__(
            self, IKAMOANA_config_filepath: str, SEAPODYM_config_filepath: str = None,
            root_directory: str = None
            ):
        """Create a data structure to store all informations needed to
        compute the IKAMOANA fields.

        Parameters
        ----------
        IKAMOANA_config_filepath : str
            Path to the IKAMOANA configuration XML file.
        SEAPODYM_config_filepath : str, optional
            SEAPODYM configuration filepath can also be specified by
            user.
        root_directory : str, optional
            If the SEAPODYM configuration file is not at the root of the
            working directory, this (working) directory path must be
            specified.
            
            Example : ~/SEAPODYM/run-test/ which contains `parameter
            files` and the `data` directory.
        """        

        tree = ET.parse(IKAMOANA_config_filepath)
        root = tree.getroot()
        
        if SEAPODYM_config_filepath is None :
            SEAPODYM_config_filepath = tagReading(
                root,['directories','seapodym_file'])
        
        self.SEAPODYM_config_filepath = SEAPODYM_config_filepath
        self.IKAMOANA_config_filepath = IKAMOANA_config_filepath
        if root_directory is None :
            root_directory = os.path.dirname(SEAPODYM_config_filepath)
        
        self._readIkamoana(root)
        
        if root.find("mortality") is not None :
            self._readMortality(root)
        
        tree = ET.parse(SEAPODYM_config_filepath)
        root = tree.getroot()
        
        self._readSeapodym(root)
        
        if hasattr(self, "selected_fisheries") :
            self._readFisheries(root, root_directory)

# ---------------------- IKAMOANA CONFIGURATION FILE ----------------- #
        
    def _readIkamoana(self, root: ET.Element) :
        self.diffusion_boost=float(tagReading(root,['forcing','diffusion_boost'],0))
        self.diffusion_scale=float(tagReading(root,['forcing','diffusion_scale'],1))
        self.c_scale=float(tagReading(root,['forcing','c_scale'],1))
        self.taxis_scale=float(tagReading(root,['forcing','taxis_scale'],1))
        
        self.units=tagReading(root,['forcing','units'],'m_per_s')
        """only `m_per_s` and `nm_per_timestep` are supported"""
        
        tmp = tagReading(root,['forcing','shallow_sea_to_ocean'], 'False')
        self.shallow_sea_to_ocean = tmp in ["True", "true"]
        tmp = tagReading(root,['forcing','landmask_from_habitat'], 'False')
        self.landmask_from_habitat = tmp in ["True", "true"]
        """Specify if the landmask is based on the SEAPODYM mask used
        to compute feeding habitat or not."""
        
        tmp = tagReading(root,['forcing','correct_epi_temp_with_vld'], 'False')
        self.correct_epi_temp_with_vld = tmp in ["True", "true"]
        """Correct the epipelagic layer (L1) using the vertical gradient.
        See Also : FeedingHabitat.correct_epi_temp_with_vld()"""
        
        tmp = tagReading(root,['forcing','indonesian_filter'], 'False')
        self.indonesian_filter = tmp in ["True", "true"]
        """Apply the indonesian filter.
        See Also : FeedingHabitat.indonesianFilter()"""
        
        tmp = tagReading(root,['forcing','vertical_movement'], 'False')
        self.vertical_movement = tmp in ["True", "true"]
        """Correction of rho by passive advection. Maybe temporary."""
        

    def _readMortality(self, root: ET.Element) :
        iter_fisheries = root.find("mortality").find(
            "selected_fisheries").findall("fishery")
        if len(iter_fisheries) < 1 :
            raise ValueError("You must specify at least one fishery in "
                             "<selected_fisheries> tag or totaly remove "
                             "<mortality> tag if you don't want to compute F.")
        selected_fisheries = {}
        for fishery in iter_fisheries :
            seapodym_name = fishery.attrib['name']
            if 'effort_file_name' in fishery.attrib :
                effort_file_name = fishery.attrib['effort_file_name']
            else :
                effort_file_name = seapodym_name
            selected_fisheries[seapodym_name] = effort_file_name
        self.selected_fisheries = selected_fisheries
        tmp = tagReading(root, ["mortality","predict_effort"], "False")
        self.predict_effort = True if tmp in ["True", "true"] else False

        skiprows = tagReading(root, ["mortality","skiprows"], '2')
        self.skiprows = np.int32(skiprows.split())
        
# ---------------------- SEAPODYM CONFIGURATION FILE ----------------- #

    def _readSeapodym(self, root: ET.Element) :

        sp_name = root.find('sp_name').text
        deltaT = float(root.find('deltaT').attrib["value"])
        
        self.timestep=deltaT*24*60*60
        """Delta time from SEAPODYM configuration file. May be different
        than `dt` in IKAMOANA configuration file."""
        
        ## TAXIS ####################################
        self.vmax_a=float(root.find('MSS_species').attrib[sp_name])
        """SEAPODYM name is MSS_species. Velocity at maximal habitat
        gradient and `A = 1, BL/s`."""
        self.vmax_b=float(root.find('MSS_size_slope').attrib[sp_name])
        """SEAPODYM name is MSS_size_slope. Slope coefficient in
        allometric function for tuna velocity."""
        
        ## CURRENTS #################################
        # TODO : For now, only the first layer is used.
        self.u_file = root.find('strfile_u').attrib['layer0']
        self.v_file = root.find('strfile_v').attrib['layer0']
        
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

    def _readFisheries(self, root: ET.Element, root_directory) :
        """Read a XML file to get all parameters needed for mortality
        field production."""

        species_name = root.find("sp_name").text

        ## FISHERY FILES #################################
        directory_fishery = os.path.join(
            root_directory, root.find('strdir_fisheries').attrib['value'])
        files_list = os.listdir(directory_fishery)
        if files_list == [] :
            raise ValueError('The directory that should contain the fisheries '
                             'effort files is empty.')
        self.fishery_filepaths = [
            os.path.join(directory_fishery,path)for path in files_list]

        ## PARAMETERS ####################################
        nb_fishery = int(root.find('nb_fishery').attrib['value'])
        list_fishery_name = root.find('list_fishery_name').text.split()

        # fisheries name
        if len(list_fishery_name) != nb_fishery :
            raise ValueError((
                "nb_fishery is {} but list_fishery_name contains {} elements."
                ).format(nb_fishery, len(list_fishery_name)))

        f_param = {}
        for f in list_fishery_name :
            tmp_dict = {
                "function_type":int(root.find("s_sp_fishery").find(f).find(
                    "function_type").attrib["value"]),
                "q":float(root.find("q_sp_fishery").find(f).attrib[species_name]),
                "variable":float(root.find(
                    "s_sp_fishery").find(f).attrib[species_name]),
                "length_threshold":float(root.find('s_sp_fishery').find(f).find(
                    "length_threshold").attrib[species_name])}

            if tmp_dict['function_type'] == 3 :
                tmp_dict['right_asymptote'] = float(
                    root.find("s_sp_fishery").find(f).find(
                        "right_asymptote").attrib[species_name])
            f_param[f] = tmp_dict

        self.f_param = f_param