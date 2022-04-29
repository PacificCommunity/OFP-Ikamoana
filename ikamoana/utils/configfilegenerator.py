from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import os

class ConfigFileGenerator :

    def __init__(
            self, run_name: str = "", random_seed: float = ""
            ) -> None:
        """
        Generate :
        - base tags and file.
        - run_name -> Externally (All values in CSV are chained)
        - random_seed -> default is empty (="")
        """
        self.root = ET.Element("ika_params")
        se_run_name = ET.SubElement(self.root, "run_name")
        se_run_name.text = run_name
        se_random_seed = ET.SubElement(self.root, "random_seed")
        se_random_seed.text = random_seed

    def directories(
            self, forcing_dir:str, start_distribution:str, seapodym_file:str):

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
            lat_min: float = 0, lat_max: float = 0,
            lon_min: float = 0, lon_max: float = 0
            ) :
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
        """OPTIONAL function

        fishery_dict :
            {fishery name in SEAPODYM : fishery name in effort file}
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
        files_dict : {file_name : (is_a_dataset, filepath)}
        """

        forcing = ET.Element("forcing")

        attributes = {"files_only":str(files_only),
                      "home_directory":str(home_directory)}
        e_files = ET.Element("files", attrib=attributes)

        for file_name, (dataset_bool, filepath) in files_dict.items() :
            se_file = ET.SubElement(e_files, file_name)
            se_file.attrib['dataset'] = str(dataset_bool)
            # TODO include home_directory
            se_file.text = str(filepath)#os.path.join(str(home_directory), str(filepath))
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
        kernels = ET.Element("kernels")

        for kernel in kernels_list :
            se_kernel = ET.SubElement(kernels, "kernel")
            se_kernel.text = str(kernel)

        self.root.append(kernels)

    def write(self, filepath:str):

        dirname, filename = os.path.split(filepath)
        if dirname == "" :
            dirname = "./"
        pref, _ = os.path.splitext(filename)
        filepath = os.path.join(dirname, pref+".xml")

        parser = parseString(ET.tostring(self.root))
        pretty_xml = parser.toprettyxml()

        with open(filepath, "w") as files :
                files.write(pretty_xml)

        return filepath
