from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import linear_model

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
                               effort_label:np.sum, catch_label:np.sum})
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
        update_dict = {
            gear_label:fishery[gear_label].explode().unique(),
            resolution_label:fishery[resolution_label].explode().unique()}
        
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
    
    return xr.Dataset(f_effort_dict, attrs=data_set.attrs)

## TOOL ################################################################

def sumDataSet(fisheries: xr.Dataset) -> xr.DataArray :
    
    f_name_list = list(fisheries)
    sum = np.nan_to_num(fisheries[f_name_list[0]])
    
    for f in f_name_list[1:] :
        sum = sum + np.nan_to_num(fisheries[f])
    
    return xr.DataArray(data=sum, coords=fisheries.coords,attrs={"Type":"Sum of "+fisheries.attrs["Type"]})

def toTextFile(fisheries: Union[dict, pd.DataFrame],
               filepath: str = './ouput.txt', fishery_label: str = "f",
               resolution_label: str = "res", gear_label: str = "gr",
               year_label: str = "yr", month_label: str = "mm",
               day_label: str = "dd", latitude_label: str = "lat",
               longitude_label: str = "lon", effort_label: str = "E", catch_label: str = "C") :

    if ((not isinstance(fisheries, dict))
        or (not isinstance(fisheries, pd.DataFrame))) :
        raise ValueError(
            ("Argument fisheries must be a dict or a pandas DataFrame. Actual "
             "type is {}").format(type(fisheries)))

    if isinstance(fisheries, dict) :
        fisheries = pd.concat(fisheries.values())
    
    fisheries = fisheries.sort_values([fishery_label, year_label, month_label,
                                       day_label, latitude_label, longitude_label])
    fisheries = fisheries[fishery_label, year_label, month_label, day_label,
                          gear_label, latitude_label, longitude_label,
                          effort_label, catch_label]
    fisheries.to_csv(filepath, sep='\t', index= False)
        

## CLEANING DATA #######################################################

def removeEmptyEntries(fishery: pd.DataFrame, fisheries_label: str = 'f',
                       effort_label: str = 'E', catch_label: str = 'C',
                       verbose: bool = False) -> pd.DataFrame :
    if verbose :
        print("Removed %d empty entries."%(
            len(fishery[(fishery[catch_label]==0)
                        & (fishery[effort_label]==0)])))
        print('Number of empty entries per fishery :\n',
              fishery[(fishery[catch_label]==0) & (fishery[effort_label]==0)][
                  [fisheries_label, catch_label]].groupby(
                      [fisheries_label]).count().rename(
                          columns={catch_label:'empty'}).T,end="\n\n")

    return fishery[(fishery[catch_label]!=0) | (fishery[effort_label]!=0)]

def removeNoCatchEntries(fishery: pd.DataFrame,fisheries_label: str = 'f',
                         catch_label: str = 'C', verbose: bool = False
                         ) -> pd.DataFrame :
    if verbose :
        print("Removed %d entries without catch."%(
            len(fishery[fishery[catch_label]==0])))
        print('Number of entries without catch per fishery :\n',
              fishery[fishery[catch_label]==0][
                  [fisheries_label,catch_label]].groupby(
                      [fisheries_label]).count().rename(
                          columns={catch_label:'no catch'}).T)
        
    return fishery[fishery[catch_label]!=0]

def plotByGear(fishery: pd.DataFrame, choose_gear: List = None,
               gear_label: str = 'gr', effort_label: str = 'E',
               catch_label: str = 'C', figsize=(10,10), label_fontsize=15,
               legend_fontsize=15, title=None, title_fontsize=24) :
    """
    Warnings
    --------
    Don't use `groupByFisheries()` before using this function.
    """
    plt.subplots(1,1,figsize=figsize)
    if choose_gear is None: gr = fishery[gear_label].explode().unique()
    else : gr = np.array(choose_gear).ravel()
        
    #color = ['b','g','r','c','m','y','k']
    color = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    for g, c in zip(gr,color) :
        plt.plot(fishery[fishery[gear_label]==g][catch_label],
                 fishery[fishery[gear_label]==g][effort_label],
                 marker='o', markersize=2, color=c,linestyle="None")

    plt.legend(gr, fontsize=legend_fontsize)
    plt.xlabel("Catch", fontsize=label_fontsize)
    plt.ylabel("Effort", fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.grid(True)
    plt.show()

def predictEffort(fishery: pd.DataFrame, conserve_no_catch: bool = False,
                  conserve_empty: bool = False, gear_to_choose: str = 'K',
                  catch_label: str = 'C', gear_label: str = 'gr',
                  effort_label: str = 'E') -> pd.DataFrame :
    """
    Predict effort (where effort is null but catch is not) using linear
    regression.
    """
    
    model = fishery[(fishery[gear_label]==gear_to_choose)
                    & (fishery[effort_label]!=0) & (fishery[catch_label]!=0)]
    
    effort_to_predict = pd.DataFrame(fishery[~fishery.index.isin(model.index)])
    
    no_catch_but_effort = effort_to_predict[(effort_to_predict[effort_label]!=0)
                               & (effort_to_predict[catch_label]==0)]

    nothing = effort_to_predict[(effort_to_predict[effort_label]==0)
                               & (effort_to_predict[catch_label]==0)]
    
    if effort_to_predict.size != 0 :
        effort_to_predict = effort_to_predict[(effort_to_predict[catch_label]!=0)]
        
        regr = linear_model.LinearRegression()
        fun = regr.fit(np.array(model[catch_label])[:,np.newaxis],
                    np.array(model[effort_label]))
        
        effort_to_predict.drop(effort_label, axis=1)
        effort_to_predict[effort_label] = fun.predict(
            np.array(effort_to_predict[catch_label])[:,np.newaxis])
        
    df_to_return = [model, effort_to_predict]
    if conserve_no_catch : df_to_return.append(no_catch_but_effort)
    if conserve_empty :df_to_return.append(nothing)
    
    return pd.concat(df_to_return)

def predictEffortAllFisheries(
    fishery: dict, conserve_no_catch: bool = True,
    conserve_empty: bool = True, gear_to_choose: dict = {9:'K'},
    catch_label: str = 'C', gear_label: str = 'gr',
    effort_label: str = 'E') -> dict :
    
    dict_update = {}
    for key, value in fishery.items() :
        if key in gear_to_choose : gear = gear_to_choose[key]
        else :
            gear_list, index = np.unique(value[gear_label],
                                         return_counts=True)
            gear = gear_list[np.argmax(index)]

        new_value = predictEffort(value,conserve_no_catch,conserve_empty,gear,
                                  catch_label,gear_label,effort_label)
        dict_update[key] = new_value
    
    fishery.update(dict_update)
    
    return fishery

## WRAPPER #############################################################

def effortByFishery(filepath: str, time_reso: int, space_reso: float,
                    header_position: int = None, columns_name: List[str] = None,
                    removeNoCatch: bool = False, predict_effort: bool =False,
                    verbose: bool = False) -> dict :
    
    fisheries_effort = readFile(filepath, header_position, columns_name)
    fisheries_effort = removeEmptyEntries(fisheries_effort,verbose)
    if removeNoCatch :
        fisheries_effort = removeNoCatchEntries(fisheries_effort,verbose)
    fisheries_effort = separateFisheries(fisheries_effort)
    if predict_effort :
        fisheries_effort = predictEffortAllFisheries(fisheries_effort)
    fisheries_effort = rescaleFisheries(fisheries_effort)
    fisheries_effort = groupByFisheries(fisheries_effort)
    fisheries_effort, _ = fisheriesToDataSet(fisheries_effort)
    return normalizeCoords(fisheries_effort, time_reso, space_reso)