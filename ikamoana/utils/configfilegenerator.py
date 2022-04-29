from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import os

class ConfigFileGenerator :
    """
    This class generates a configuration file for the Ikamoana
    simulation. All functions will generate the necessary tags except
    for mortality which is optional.
    
    Examples
    --------
    First example : Generates a configuration file with default settings.
    
    >>> my_cfg = ConfigFileGenerator()
    >>> my_cfg.directories("~/forcing/", "~/start_distribution/",
    ...                    "~/seapodym_file/")
    >>> my_cfg.domain()
    >>> my_cfg.cohortInfo()
    >>> my_cfg.mortality({"fishery_P1":"1", "fishery_S2":"2"})
    >>> my_cfg.forcing()
    >>> my_cfg.kernels(["IkAdvectionRK4","IkaDymMove","LandBlock","Age])
    >>> my_cfg.write("~/my_config_file.xml")
    
    See also
    --------
    Documentation : `Ikamoana_configuration_file.md` in
    `Ikamoana/doc/Configuration Files/`.
    """
    
    def __init__(
            self, run_name: str = "", random_seed: float = ""
            ) -> None:
        """Generates the basic structure of the configuration file.

        Parameters
        ----------
        run_name : str, optional
            Randomly generated if left empty.
            
        random_seed : float, optional
            Randomly generated if left empty.
        """

        self.root = ET.Element("ika_params")
        se_run_name = ET.SubElement(self.root, "run_name")
        se_run_name.text = run_name
        se_random_seed = ET.SubElement(self.root, "random_seed")
        se_random_seed.text = random_seed
    
    def directories(
            self, forcing_dir:str, start_distribution:str, seapodym_file:str):
        """
        Parameters
        ----------
        forcing_dir : str
            Path to the directory containing the forcing fields (forcing
            used by the kernels). Also used to export fields to NetCDF.
            
        start_distribution : str
            Path to the directory containing the starting distribution
            of the particles.
            
        seapodym_file : str
            Path to the SEAPODYM configuration file.
        """
        directories = ET.Element("directories")
        
        se_forcing_dir = ET.SubElement(directories,"forcing_dir")
        se_forcing_dir.text = str(forcing_dir)
        se_start_distribution = ET.SubElement(directories,"start_distribution")
        se_start_distribution.text = str(start_distribution)
        se_seapodym_file = ET.SubElement(directories,"seapodym_file")
        se_seapodym_file.text = str(seapodym_file)
        
        self.root.append(directories)
    
    def domain(
            self,
            dt: float = 1, start: str = "", sim_time: float = 1, output_dt: float = 0,
            lat_min: float = -90, lat_max: float = 90,
            lon_min: float = 0, lon_max: float = 360
            ) :
        """
        Parameters
        ----------
        dt : float, optional
            The IKAMOANA simulation delta time in days. This is not the
            delta time of the SEAPODYM forcing fields (defined in
            SEAPODYM configuration file). They can be different.
            
        start : str, optional
            The start date of the simulation in days. The format is
            YYYY-MM-DD and will be converted to numpy.datetime64.
            
        sim_time : float, optional
            Total simulation time in days.
            
        output_dt : float, optional
            If you choose to export the particle history, you can choose
            a delta time different from the IKAMOANA dt.
            
        lat_min, lat_max, lon_min, lon_max : float, optional
            These values will limit the domain in space. Latitude is
            South to North ([-90, 90]) and longitude is West to Est
            ([0, 360])
            
        Note
        ----
        `dt`, `start` and `output_dt` are expressed in days.
        """
        domain = ET.Element("domain")
        
        e_time = ET.Element("time")
        se_dt = ET.SubElement(e_time, "dt")
        se_dt.text = str(dt)
        se_start = ET.SubElement(e_time, "start")
        se_start.text = str(start)
        se_sim_time = ET.SubElement(e_time, "sim_time")
        se_sim_time.text = str(sim_time)
        se_output_dt = ET.SubElement(e_time, "output_dt")
        se_output_dt.text = str(output_dt)
        domain.append(e_time)
        
        e_lat = ET.Element("lat")
        se_lat_min = ET.SubElement(e_lat, "min")
        se_lat_min.text = str(lat_min)
        se_lat_max = ET.SubElement(e_lat, "max")
        se_lat_max.text = str(lat_max)
        domain.append(e_lat)
        
        e_lon = ET.Element("lon")
        se_lon_min = ET.SubElement(e_lon, "min")
        se_lon_min.text = str(lon_min)
        se_lon_max = ET.SubElement(e_lon, "max")
        se_lon_max.text = str(lon_max)
        domain.append(e_lon)
        
        self.root.append(domain)
    
    def cohortInfo(
            self, start_length: float = 0, ageing: bool = False,
            number_of_cohorts: int = 0, start_dynamic_file: str = None,
            file_extension: str = "nc", start_static_file: str = None,
            start_cell_lon: float = None, start_cell_lat: float = None
            ):
        """
        Parameters
        ----------
        start_length : float, optional
            The size of the first cohort. It refers to the <length> tag
            in the SEAPODYM configuration file.
            
        ageing : bool, optional
            Define if the cohort age will evolve through time.
            
        number_of_cohorts : int, optional
            The total number of particles in the simulation at the first
            time step.
            
        start_dynamic_file : str, optional
            This kind of file is 3D. Ikamoana will select previous
            start_date/cohort_age based on the <start_length> and
            (<time>) <start> tags.
            
        file_extension : str, optional
            This is the extension of the start_dynamic_file chosen 
            between "nc" and "dym". If none is passed, the default 
            value will be "nc".
            
        start_static_file : str, optional
            This type of file is 2D (no time axis) or only the first
            time step will be chosen.
            
        start_cell_lon, start_cell_lat : float, optional
            The particles will be randomly distributed in a specific
            cell. This method is used in tag release simulations.
        """
        cohort_info = ET.Element("cohort_info")
        
        se_start_length = ET.SubElement(cohort_info, "start_length")
        se_start_length.text = str(start_length)
        se_ageing = ET.SubElement(cohort_info, "ageing")
        se_ageing.text = str(ageing)
        se_number_of_cohorts = ET.SubElement(cohort_info, "number_of_cohorts")
        se_number_of_cohorts.text = str(number_of_cohorts)
        if start_dynamic_file is not None :
            se_start_dynamic_file = ET.SubElement(cohort_info, "start_dynamic_file")
            se_start_dynamic_file.text = str(start_dynamic_file)
            se_start_dynamic_file.attrib["file_extension"] = str(file_extension)
        if start_static_file is not None :
            se_start_static_file = ET.SubElement(cohort_info, "start_static_file")
            se_start_static_file.text = str(start_static_file)

        if (start_cell_lon is not None) and (start_cell_lat is not None) :
            e_start_cell = ET.Element("start_cell")
            se_lon = ET.SubElement(e_start_cell, "lon")
            se_lon.text = str(start_cell_lon)
            se_lat = ET.SubElement(e_start_cell, "lat")
            se_lat.text = str(start_cell_lat)
            cohort_info.append(e_start_cell)
        
        self.root.append(cohort_info)
    
    def mortality(
            self, fishery_dict: Dict[str,str], skiprows: int = 0,
            predict_effort: bool = False, effort_file: str = "",
            import_effort: bool = False, export_effort: bool = False
            ):
        """OPTIONAL function.

        Parameters
        ----------
        fishery_dict : Dict[str,str]
            Used to select all fisheries that the user wishes to include
            in the calculation of the mortality field.
            {fishery name in SEAPODYM : fishery name in effort file}
            
        skiprows : int, optional
            By default, in SEAPODYM, two lines are added at the begining
            of the file. These lines indicate, respectively, the number
            of fisheries and, for each of these fisheries, the number of
            entries in the text file. According to this, specify how 
            many lines have been added.
            
        predict_effort : bool, optional
            Effort can be predicted using the regression method when it
            is equal to 0 but catch is a positiv value. `No longer
            supported`. Can easily be reimplemented if necessary.
            
        effort_file : str, optional
            Default value is `forcing_dir+run_name+"_effort.nc"`. This
            is the filepath to the effort dataset. It help to reduce the
            computation time of the mortality.
            
        import_effort : bool, optional

        export_effort : bool, optional

            
        Note
        ----
        This part is optional. If you don't need mortality field
        computation you can skip it.
        
        At high (spatial) resolution, this part takes a lot of time to
        compute. It is recommended to calculate it once and then save
        it in a NetCDF file.
        """
        
        mortality = ET.Element("mortality")
        
        effort_file_attrs = {"import":str(import_effort),
                             "export":str(export_effort)}
        se_effort_file = ET.SubElement(mortality, "effort_file",
                                       attrib=effort_file_attrs)
        se_effort_file.text = str(effort_file)
                
        e_selected_fisheries = ET.Element("selected_fisheries")
        for name, effort_file_name in fishery_dict.items() :
            se_fishery = ET.SubElement(e_selected_fisheries, "fishery")
            se_fishery.attrib['name'] = str(name)
            if effort_file_name not in [None, ""] :
                se_fishery.attrib['effort_file_name'] = str(effort_file_name)
        mortality.append(e_selected_fisheries)
        
        if predict_effort :
            raise ValueError("The prediction of effort is no longer supported.")
        se_predict_effort = ET.SubElement(mortality, "predict_effort")
        se_predict_effort.text = str(predict_effort)
        
        se_skiprows = ET.SubElement(mortality, "skiprows")
        se_skiprows.text = str(skiprows)
        
        self.root.append(mortality)

    def forcing(
            self,
            files_dict: Dict[str, Tuple[bool, str]] = {},
            files_only: bool = False,
            home_directory: bool = "",
            correct_epi_temp_with_vld: bool = False,
            landmask_from_habitat: bool = False,
            shallow_sea_to_ocean: bool = False, indonesian_filter: bool = False,
            vertical_movement: bool = False, diffusion_boost: float = 0,
            diffusion_scale: float = 1, c_scale: float = 1,
            taxis_scale: float = 1, units: str = "m_per_s",
            field_interp_method: str = "nearest"
            ) :
        """
        Parameters
        ----------
        files_dict : Dict[str, Tuple[bool, str]], optional
            All the files you want to add to the ocean fieldset. The
            dictionary is structured as follow :
            `{field_name : (is_a_dataset, filepath)}`
            
            Where :
            - `file_name` is the name of the field in the fieldset.
            - `is_a_dataset` is `True` if the file is a Dataset (contains
            several DataArray) and `False` otherwise.
            - `filepath` is the path to the NetCDF.
            
        files_only : bool, optional
            Specify if you want to generate feeding habitat and others
            (taxis, diffusion etc...) ("False") or not ("True").
            
        home_directory : bool, optional
            The directory where all the fields below are stored.
            
        correct_epi_temp_with_vld : bool, optional
            Specify if you want to apply the
            `FeedingHabitat.correctEpiTempWithVld` method to the habitat
            computation.
            
        landmask_from_habitat : bool, optional
            Specify if you want to use Seapodym landmask or a landmask
            extract from habitat Nan values.
            
        shallow_sea_to_ocean : bool, optional
            Specify if you want to consider shallow sea as ocean ("True")
            or land ("False").
            
        indonesian_filter : bool, optional
            Specify if you want to apply the
            `FeedingHabitat.indonesianFilter` method to the habitat
            computation.
            
        vertical_movement : bool, optional
            Specify if you want to add a correction by the passive
            advection to the diffusion fields.
            
        diffusion_boost : float, optional
            Is added to Diffusion.
            
        diffusion_scale : float, optional
            Multiply the diffusion.
            
        c_scale : float, optional
            Multiply the Seapodym c_diff_fish parameter in diffusion
            computation.
            
        taxis_scale : float, optional
            Multiply the taxis.
            
        units : str, optional
            Only m_per_s and nm_per_timestep are supported for now.
            
        field_interp_method : str, optional
            String, default is "nearest". Only "linear", "nearest",
            "linear_invdist_land_tracer", "cgrid_velocity",
            "cgrid_tracer" and "bgrid_velocity" are supported for now.
        """
        
        forcing = ET.Element("forcing")
        
        attributes = {"files_only":str(files_only),
                      "home_directory":str(home_directory)}
        e_files = ET.Element("files", attrib=attributes)
        
        for file_name, (dataset_bool, filepath) in files_dict.items() :
            se_file = ET.SubElement(e_files, file_name)
            se_file.attrib['dataset'] = str(dataset_bool)
            # TODO include home_directory
            se_file.text = os.path.join(str(home_directory), str(filepath))
        forcing.append(e_files)
        
        se_correct_epi_temp_with_vld = ET.SubElement(forcing,"correct_epi_temp_with_vld")
        se_correct_epi_temp_with_vld.text = str(correct_epi_temp_with_vld)
        se_landmask_from_habitat = ET.SubElement(forcing,"landmask_from_habitat")
        se_landmask_from_habitat.text = str(landmask_from_habitat)
        se_shallow_sea_to_ocean = ET.SubElement(forcing,"shallow_sea_to_ocean")
        se_shallow_sea_to_ocean.text = str(shallow_sea_to_ocean)
        se_indonesian_filter = ET.SubElement(forcing,"indonesian_filter")
        se_indonesian_filter.text = str(indonesian_filter)
        se_vertical_movement = ET.SubElement(forcing,"vertical_movement")
        se_vertical_movement.text = str(vertical_movement)
        se_diffusion_boost = ET.SubElement(forcing,"diffusion_boost")
        se_diffusion_boost.text = str(diffusion_boost)
        se_diffusion_scale = ET.SubElement(forcing,"diffusion_scale")
        se_diffusion_scale.text = str(diffusion_scale)
        se_c_scale = ET.SubElement(forcing,"c_scale")
        se_c_scale.text = str(c_scale)
        se_taxis_scale = ET.SubElement(forcing,"taxis_scale")
        se_taxis_scale.text = str(taxis_scale)
        se_units = ET.SubElement(forcing,"units")
        se_units.text = str(units)
        se_field_interp_method = ET.SubElement(forcing,"field_interp_method")
        se_field_interp_method.text = str(field_interp_method)
        self.root.append(forcing)
    
    def kernels(self, kernels_list: List[str]) :
        """
        Parameters
        ----------
        kernels_list : List[str]
             Contains all the kernels name the user want to use in
             simulation execution. Must be contained in
             `ikafish.behaviours.AllKernels`. The kernels declaration
             order is the one Parcels will use to update particles. Some
             kernels are dependent on others and will have to be
             executed downstream. The user must pay attention and act
             accordingly.
        """
        kernels = ET.Element("kernels")
        
        for kernel in kernels_list :
            se_kernel = ET.SubElement(kernels, "kernel")
            se_kernel.text = str(kernel)
        
        self.root.append(kernels)

    def write(self, filepath:str):
        """Write the final configuration file."""
        
        dirname, filename = os.path.split(filepath)
        if dirname == "" :
            dirname = "./"
        pref, _ = os.path.splitext(filename)
        filepath = os.path.join(dirname, pref+".xml")
        
        parser = parseString(ET.tostring(self.root))
        pretty_xml = parser.toprettyxml()
        
        with open(filepath, "w") as files :
                files.write(pretty_xml)
    