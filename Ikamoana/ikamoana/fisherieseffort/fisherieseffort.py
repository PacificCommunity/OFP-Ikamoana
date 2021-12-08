from typing import List, Tuple, Union
import pandas as pd
import xarray as xr
import numpy as np

# TODO ; Take into account fisheries with severals "gear".

def readFile(filepath: str, header_position: int = None,
             columns_name: List[str] = None) -> pd.DataFrame :
    return pd.read_table(filepath, header=header_position, names=columns_name)

def separateFisheries(fisheries : pd.DataFrame,
                      fisheries_label: str = "f") -> dict :
    f_dict = {}
    for f_name in fisheries[fisheries_label].unique() :
        f_dict[f_name] = fisheries.loc[fisheries[fisheries_label] == f_name]
    return f_dict

def rescaleFisheries(
    fisheries: dict, resolution: float = None, resolution_label: str = "res",
    latitude_label: str = "lat", longitude_label: str = "lon",
    effort_label: str = "E", catch_label: str = "C", inplace=True
    ) -> dict :

    if resolution is None :
        resolution = min([min(fishery[resolution_label]) for fishery in fisheries.values()])
    
    def computeSubfisheriesPositions(resolution, min_resolution, latitude, longitude) :
            
        assert latitude >= -90 and latitude <= 90, "Warning : -90 <= latitude <= 90"
        assert longitude >= 0 and longitude <= 360, "Warning : 0 <= longitude <= 360"
        
        X = int(resolution / min_resolution)
        if X % 2 == 1 :
            middle = X // 2
            padding = 0
        else :
            middle = (X // 2) - 1
            padding = 1
        min_latitude, max_latitude = -90, 90
        max_longitude = 360
        
        lat_inf = max(latitude-((middle+padding)*min_resolution), min_latitude)
        lat_sup = min(latitude+(middle*min_resolution), max_latitude)
        nb_elem_lat = int(np.round((lat_sup - lat_inf) / min_resolution)) + 1
        mat_lat = np.repeat(np.linspace(lat_sup,lat_inf,nb_elem_lat), X)

        lon_inf = longitude-middle*min_resolution
        lon_sup = longitude+(middle+padding)*min_resolution
        nb_elem_lon = int(np.round((lon_sup - lon_inf) / min_resolution)) + 1
        mat_lon = np.tile(np.linspace(lon_inf, lon_sup, nb_elem_lon)%max_longitude,
                        nb_elem_lat)
        
        return mat_lat, mat_lon
    
    def createSubfisheries(row,  min_resolution) :
        resolution = row[resolution_label]
        effort = row[effort_label]
        catch = row[catch_label]
        
        latitude, longitude = computeSubfisheriesPositions(
            resolution, min_resolution, row[latitude_label], row[longitude_label])
        sub_fisheries_number = latitude.size
        sub_fisheries_list = [x for x in range(sub_fisheries_number)]
        effort /= sub_fisheries_number
        catch /= sub_fisheries_number
        
        # Returning a list is the most efficient way to compute sub-fisheries
        # Tried pandas.Series and partial pandas.Series with update.
        return [sub_fisheries_list,latitude,longitude,effort,catch]

    def computeRightPosition(row):
        return [row['sub_fishery_latitude'][row['sub_fishery']],
                row['sub_fishery_longitude'][row['sub_fishery']]]
    
    update_index = ['sub_fishery',
                    'sub_fishery_latitude', 'sub_fishery_longitude',
                    effort_label, catch_label]
    
    sub_fisheries = {}
    for name, fishery in fisheries.items() : 
        
        if max(fishery[resolution_label].unique()) > resolution :
            
            sub_fishery = fishery.copy()
            sub_fishery[update_index] = fishery.apply(
                createSubfisheries, args=(resolution,), axis=1, result_type='expand')

            sub_fishery = sub_fishery.explode('sub_fishery', ignore_index=True)
            
            sub_fishery[[latitude_label,longitude_label]] = (
                sub_fishery.apply(computeRightPosition, axis=1, result_type='expand'))
            sub_fishery = sub_fishery.drop(
                columns=['sub_fishery_latitude', 'sub_fishery_longitude', 'sub_fishery'])
            
            sub_fishery[resolution_label] = 1
            sub_fisheries[name] = sub_fishery

    if inplace :
        fisheries.update(sub_fisheries)
        return fisheries
    return sub_fisheries

def groupByFisheries(
    fisheries : dict, multi_index = False,
    fishery_label: str = "f", resolution_label: str = "res", gear_label: str = "gr",
    year_label: str = "yr", month_label: str = "mm", day_label: str = "dd",
    latitude_label: str = "lat", longitude_label: str = "lon",
    effort_label: str = "E", catch_label: str = "C"
    ) -> dict :
    
    # WARNING :
    # Group by is needed when there is multiple entries for a single [Fishery/Date/Pos].
    # Since 5° resolution fisheries are not separated by 5° distance, this step is required.
    
    for f_name, fishery in fisheries.items() :
        f_gb = fishery.groupby(by=[fishery_label,year_label,month_label,
                                   day_label,latitude_label,longitude_label])
        f_gb = f_gb.aggregate({gear_label:np.unique, resolution_label:np.unique,
                               effort_label:sum, catch_label:sum})
        fisheries[f_name] = f_gb if multi_index else f_gb.reset_index()
    
    return fisheries

def fisheriesToDataSet(
    fisheries : dict, resolution_label: str = "res", gear_label: str = "gr",
    year_label: str = "yr", month_label: str = "mm", day_label: str = "dd",
    latitude_label: str = "lat", longitude_label: str = "lon",
    effort_label: str = "E", catch_label: str = "C"
    ) -> Tuple[xr.Dataset, xr.Dataset] :
    
    fisheries_effort_da_dict = {}
    fisheries_catch_da_dict = {}
    for f_name, fishery in fisheries.items():
        
        fishery_time = pd.to_datetime(
            fishery.rename(
                columns={year_label:"year",month_label:"month",day_label:"day"}
                )[["year","month","day"]])
        update_dict = {gear_label:fishery[gear_label].unique(),
                       resolution_label:fishery[resolution_label].unique()}
        
        fishery_to_convert = pd.DataFrame({"time":fishery_time,
                                           "lat":fishery[latitude_label],
                                           "lon":fishery[longitude_label],
                                           effort_label:fishery[effort_label],
                                           catch_label:fishery[catch_label]})
        

        fishery_multi_index = fishery_to_convert.set_index(["time","lat","lon"])
        fishery_effort_xr = fishery_multi_index[effort_label].to_xarray()
        fishery_effort_xr.attrs.update(update_dict)
        fisheries_effort_da_dict[f_name] = fishery_effort_xr
        
        fishery_catch_xr = fishery_multi_index[catch_label].to_xarray()
        fishery_catch_xr.attrs.update(update_dict)
        fisheries_catch_da_dict[f_name] = fishery_catch_xr

    return (xr.Dataset(fisheries_effort_da_dict, attrs={"Type":effort_label}),
            xr.Dataset(fisheries_catch_da_dict, attrs={"Type":catch_label}))

def _completeCoords(data_array: xr.DataArray, deltaT: int = None,
                    space_reso: Union[float,int] = None) -> xr.DataArray :
    
    ## WARNING : space_reso and deltaT in effort file is independant from the space_reso in SEAPODYM configuration file.
    
    ## WARNING : deltaT == 30 in SEAPODYM mean that we use a monthly time resolution.
    
    time_coords = data_array.time.data
    lat_coords = data_array.lat.data
    lon_coords = data_array.lon.data
    data_set = data_array.to_dataset()
    
    ## TIME : Normalizing time format before reindexing
    if deltaT is not None :
        if deltaT == 30 : # MONTHLY
            time_coords = np.array(time_coords, dtype='datetime64[M]')
            linear_time_coords = np.arange(time_coords[0],time_coords[-1])
        else :
            time_coords = np.array(time_coords, dtype='datetime64[D]')
            linear_time_coords = np.arange(time_coords[0],time_coords[-1],step=deltaT)
        linear_time_coords = np.concatenate((linear_time_coords,[time_coords[-1]]))
        data_set['new_time'] = ('time',time_coords)
        data_set = data_set.reset_index('time',drop=True)
        data_set = data_set.set_index({'time':'new_time'})
    else :
        linear_time_coords = time_coords
    
    ## LATITUDE
    if space_reso is not None :
        linear_lat_coords = np.arange(lat_coords[0],lat_coords[-1],step=space_reso)
        linear_lat_coords = np.concatenate((linear_lat_coords, [lat_coords[-1]]))
        linear_lon_coords = np.arange(lon_coords[0],lon_coords[-1],step=space_reso)
        linear_lon_coords = np.concatenate((linear_lon_coords, [lon_coords[-1]]))
    else :
        linear_lat_coords = lat_coords
        linear_lon_coords = lon_coords
                
    return data_set[data_array.name].reindex(time=linear_time_coords,lat=linear_lat_coords, lon=linear_lon_coords)

def normalizeCoords(data_set: xr.Dataset, deltaT: Union[int,List[int]],
                    space_reso: Union[Union[float,int],List[Union[float,int]]] = None
                    ) -> xr.Dataset :
    
    deltaT = np.array(deltaT).ravel()
    if deltaT.size == 1 : deltaT = deltaT.repeat(len(data_set))
    elif deltaT.size != len(data_set) :
        raise ValueError("deltaT must be int or list of size len(data_set)")
    
    space_reso = np.array(space_reso).ravel()
    if space_reso.size == 1 : space_reso = space_reso.repeat(len(data_set))
    elif space_reso.size != len(data_set) :
        raise ValueError("space_reso must be int/float or list of size len(data_set)")
    
    f_effort_dict = {}
    for pos, f_name in enumerate(data_set) :
        f_effort_dict[f_name] = _completeCoords(data_set[f_name],
                                                deltaT[pos],
                                                space_reso[pos])
    
    return xr.Dataset(f_effort_dict)

def sumDataSet(fisheries: xr.Dataset) -> xr.DataArray :
    
    f_name_list = list(fisheries)
    sum = np.nan_to_num(fisheries[f_name_list[0]])
    
    for f in f_name_list[1:] :
        sum = sum + np.nan_to_num(fisheries[f])
    
    return xr.DataArray(data=sum, coords=fisheries.coords,attrs={"Type":"Sum of "+fisheries.attrs["Type"]})
