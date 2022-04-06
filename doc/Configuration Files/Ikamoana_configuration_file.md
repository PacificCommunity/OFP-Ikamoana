# Presentation

## Table of contents
1. [Main architecture](#main_arch)
2. [Directories](#directories)
3. [Domain](#domain)
4. [Cohort informations](#cohort_info)
5. [Mortality](#mortality)
6. [Forcing](#forcing)
7. [Kernels](#kernels)

## Main architecture <a name="main_arch"></a>

```xml
<?xml version="1.0"?>
<ika_params>
    <run_name>...</run_name>
    <random_seed>...</random_seed>
    <directories>...<directories>
    <domain>...</domain>
    <cohort_info>...</cohort_info>
    <mortality>...</mortality>
    <forcing>...</forcing>
    <kernels>...</kernels>
  
<ika_params>
```

- `<run_name>...</run_name>` : Contains the name of the simulation that will be randomly generated if not given.
- `<random_seed>...</random_seed>` : Contains a float that will be used to generate random artifacts (example : particles position).

## Directories <a name="directories"></a>

```xml
<directories>
    <forcing_dir>...</forcing_dir>
    <start_distribution>...</start_distribution>
    <seapodym_file>...</seapodym_file>
<directories>
```

- `<forcing_dir>` : Path to the directory containing the forcing fields (forcing used by the kernels). Also used to export fields to NetCDF.
- `<start_distribution>` : Path to the directory containing the starting distribution of the particles.
- `<seapodym_file>` : Path to the SEAPODYM configuration file.

## Domain <a name="domain"></a>

```xml
<domain>
    <time>
        <dt>...</dt>
        <start>...</start>
        <sim_time>...</sim_time>
        <output_dt>...</output_dt>
    </time>
    <lat>
        <min>...</min>
        <max>...</max>
    </lat>
    <lon>
        <min>...</min>
        <max>...</max>
    </lon>
</domain>
```

- `<dt>` : The IKAMOANA simulation delta time in *days*. This is not the delta time of the SEAPODYM forcing fields (defined in SEAPODYM configuration file). They can be different.
- `<start>` : The start date of the simulation in *days*. The format is YYYY-MM-DD and will be converted to *numpy.datetime64*.
- `<sim_time>` : Total simulation time in *days*.
- `<output_dt>` : If you choose to export the particle history, you can choose a delta time different from the IKAMOANA `dt`.
- `<lat>`, `<lon>` : These values will limit the domain in space. Latitude is South to North ([-90, 90]) and longitude is West to Est ([0, 360])


> Note : `dt`, `start` and `output_dt` are expressed in days.

## Cohort informations <a name="cohort_info"></a>

```xml
<cohort_info>
    <start_length>...</start_length>
    <ageing>...</ageing>
    <number_of_cohorts>...</number_of_cohorts>
    <start_dynamic_file file_extension="...">...</start_dynamic_file>
    <start_static_file>...</start_static_file>
    <start_cell>
        <lon>...</lon>
        <lat>...</lat>
    </start_cell>
</cohort_info>
```

- `<start_length>` : The size of the first cohort. It refers to the `<length>` tag in the SEAPODYM configuration file.
- `<ageing>` : Define if the cohort age will evolve through time.
- `<number_of_cohorts>` : The total number of particles in the simulation at the first time step.

### Cohorts initialization
At least one of these methods must be chosen. If there is more than one tag in the configuration file, user will need to define the one he want to use in the `initializeParticleSet` function using the `method` argument.

- `<start_dynamic_file>` : This kind of file is 3D. Ikamoana will select previous start_date/cohort_age based on the `<start_length>` and (*\<time\>*) `<start>` tags.
  - Attribut `file_extension` : Chosen among "nc" and "dym". If no file_extension is passed, default will be "nc".
- `<start_static_file>` : This type of file is 2D (no time axis) or only the first time step will be chosen.
- `<start_cell>` : The particles will be randomly distributed in a specific cell. This method is used in tag release simulations.

## Mortality <a name="mortality"></a>

```xml
  <mortality>
    <selected_fisheries>
      <fishery name="..." effort_file_name="..."/>
    </selected_fisheries>
    <predict_effort>...</predict_effort>
    <skiprows>...</skiprows>
  </mortality>
```

- `<mortality>` (*Optional*) : If there is no mortality tag in the configuration file, mortality will not be calculated by IkaField.
- `<selected_fisheries>` : Used to select all fisheries that the user wishes to include in the calculation of the mortality field.
  - `<fishery>` :
    - Attribut `name` : Fishery name in SEAPODYM configuration file.
    - Attribut `effort_file_name` (*Optional*) : Fishery name in effort file.
- `<predict_effort>` : Default is False. Effort can be predicted using the regression method when it is equal to 0 but catch is a positiv value. **No longer supported.** Can easily be reimplemented if necessary.
- `<skiprows>` : Default value is 0. By default, in SEAPODYM, two lines are added at the begining of the file. These lines indicate, respectively, the number of fisheries and, for each of these fisheries, the number of entries in  the text file. According to this, specify how many lines have been added.
  
> **Example** :
>```xml
><selected_fisheries>
>    <fishery name="F1" effort_file_name="1"/>
>    <fishery name="2"/>
></selected_fisheries>
>```

## Forcing fields <a name="forcing"></a>
```xml
<forcing>

    <files files_only="...">
      <... dataset="..." >...</...>
    </files>

    <correct_epi_temp_with_vld>>...</correct_epi_temp_with_vld>
    <landmask_from_habitat>...</landmask_from_habitat>
    <shallow_sea_to_ocean>...</shallow_sea_to_ocean>
    <indonesian_filter>...</indonesian_filter>
    <vertical_movement>...</vertical_movement>
    <diffusion_boost>...</diffusion_boost>
    <diffusion_scale>...</diffusion_scale>
    <c_scale>...</c_scale>
    <taxis_scale>...</taxis_scale>
    <units>...</units>
    <field_interp_method>...</field_interp_method>
  </forcing>
  ```

- `<files>` (*Optional*) : 
  - Attribut `files_only` : Specify if you want to generate feeding habitat and others (taxis, diffusion etc...) ("False") or not ("True").
  - `<...>` : "..." is the name of the Field/DataArray you want to add to the ocean FieldSet. This tag is followed by the filepath to the NetCDF (or Dymfile).
    - Attribut `dataset` : If you want to add all fields from a Dataset you must specify this attribut as True. For example, you may want to export your ocean once it is computed to reuse it later. To do so, export as a DataSet using `oceanToNetCDF` function with argument `to_dataset=True`. Then use this tag this way :
    ```xml
    <files files_only="True">
      <ocean dataset="True">/path/to/the/file.nc</ocean>
    </files>
    ```
- `<correct_epi_temp_with_vld>` : Boolean, default is False. Specify if you want to apply the `FeedingHabitat.correctEpiTempWithVld` method to the habitat computation.
- `<landmask_from_habitat>` : Boolean, default is False. Specify if you want to use Seapodym landmask or a landmask extract from habitat Nan values.
- `<shallow_sea_to_ocean>` : Boolean, default is False. Specify if you want to consider shallow sea as ocean ("True") or land ("False").
- `<indonesian_filter>` : Boolean, default is False. Specify if you want to apply the `FeedingHabitat.indonesianFilter` method to the habitat computation.
- `<vertical_movement>` : Boolean, default is False. Specify if you want to add a correction by the passive advection to the diffusion fields.
- `<diffusion_boost>` : Float, default is 0. Is added to Diffusion.
- `<diffusion_scale>` : Float, default is 1. Multiply the diffusion.
- `<c_scale>` : Float, default is 1. Multiply the Seapodym `c_diff_fish` parameter in diffusion computation.
- `<taxis_scale>` : Float, default is 1. Multiply the taxis.
- `<units>` : String, default is "m_per_s". Only m_per_s and nm_per_timestep are supported for now.
- `<field_interp_method>` : String, default is "nearest". Only "linear", "nearest", "linear_invdist_land_tracer", "cgrid_velocity", "cgrid_tracer" and "bgrid_velocity" are supported for now.

## Kernels <a name="kernels"></a>

```xml
  <kernels>
    <kernel>...</kernel>
  </kernels>
```

- `<kernels>` : Contains all the kernels the user want to use in simulation execution.
  - `<kernel>` : Kernel name. Must be contained in `ikafish.behaviours.AllKernels`. The kernels declaration order is the one Parcels will use to update particles. Some kernels are dependent on others and will have to be executed downstream. The user must pay attention and act accordingly.