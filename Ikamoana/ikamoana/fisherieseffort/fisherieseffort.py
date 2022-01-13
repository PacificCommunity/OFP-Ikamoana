"""
This module was written to produce fishing effort fields from a text
file listing all the surveys of various fisheries. 


Routine Listings
----------------
The most common use of this set of functions is described in a wrapper
function : `effortByFishery`. It corresponds to successive use of :
- readFile
- removeEmptyEntries
- (Optional -> removeNoCatchEntries)
- separateFisheries
- (Optional -> predictEffortAllFisheries)
- rescaleFisheries
- groupByFisheries
- fisheriesToDataSet
- normalizeCoords

"""


from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import linear_model

def readFile(
        filepath: str, skiprows: int = 0, columns_name: List[str] = None
        ) -> pd.DataFrame :
    """Read a fisheries effort/catch file and return a dataframe
    containing all the entries.

    Parameters
    ----------
    filepath : str
        The path to the file.
    skiprows : int, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file.
    columns_name : List[str], optional
        You can specify the names of the columns using a list containing
        all these names.

    Returns
    -------
    pd.DataFrame
        A Dataframe containing all the fisheries entries.
    """
    
    return pd.read_table(filepath, skiprows=skiprows, names=columns_name)

def separateFisheries(
        fisheries : pd.DataFrame, fisheries_label: str = "f"
        ) -> dict :
    """Separate all entries according to the fishery name.

    Parameters
    ----------
    fisheries : pd.DataFrame
        The data to separate, should be the readFile output.

    Returns
    -------
    dict
        A dict containing fishery names as keys and entries (Dataframe)
        as values.
    """
    
    f_dict = {}
    for f_name in fisheries[fisheries_label].unique() :
        f_dict[f_name] = fisheries.loc[fisheries[fisheries_label] == f_name]
    return f_dict

def rescaleFisheries(
        fisheries: dict, resolution: float = None, inplace: bool = False, 
        resolution_label: str = "res", latitude_label: str = "lat",
        longitude_label: str = "lon", effort_label: str = "E",
        catch_label: str = "C"
        ) -> Union[dict, None] :
    """Rescale fisheries with the specified resolution or the minimal
    resolution if the `resolution` argument is `None`.

    Parameters
    ----------
    fisheries : dict
        A dictionary containing fisheries name as keys and entries as
        values.
    resolution : float, optional
        The resolution used to rescale all entries. Rescaling to a lower
        resolution is not supported for now. If this argument is None
        then the used resolution is the minimal resolution found in
        `fisheries` argument.
    inplace : bool, optional
        If inplace is False the returned value is the rescaled
        dictionary. Elsewhere, the fisheries argument is update and this
        function return None.

    Returns
    -------
    Union[dict, None]
        Return the rescaled fisheries dictionary or None value depending
        on the inplace argument value.
    """

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
        else :
            sub_fisheries[name] = fishery.copy()

    if inplace :
        fisheries.update(sub_fisheries)
        return None
    return sub_fisheries

def groupByFisheries(
        fisheries : dict, multi_index = False, fishery_label: str = "f",
        resolution_label: str = "res", gear_label: str = "gr",
        year_label: str = "yr", month_label: str = "mm", day_label: str = "dd",
        latitude_label: str = "lat", longitude_label: str = "lon",
        effort_label: str = "E", catch_label: str = "C"
        ) -> dict :
    """Group entries if they have same fishery name, date and position.

    Parameters
    ----------
    fisheries : dict
        Keys are fisheries name and values are Dataframe containing
        effort and catch values.
    multi_index : bool, optional
        If True then multi index is conserved after aggregation
        (name,time,space).

    Warnings
    --------
    Group by is needed when there is multiple entries for a single
    [Fishery/Date/Pos]. For example, since 5° resolution fisheries are
    not separated by 5° distance, this step is required.

    Returns
    -------
    dict
        Return the `fisheries` argument with grouped entries as a
        dictionnary.
    """
    
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
    """Converts the fisheries dictionary into a Dataset. Each value is
    converted into a DataArray whose name is the dictionary key.

    Parameters
    ----------
    fisheries : dict
        A dictionary containing all the fisheries entries. Make sure
        there are no entries with same coordinates. To do so, use the
        `groupByFisheries` function.

    See also
    --------
    - groupByFisheries()
    
    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        First Dataset contains effort entries for each fishery while
        second contains catch entries.
    """
    
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

def _completeCoords(
        data_array: xr.DataArray, deltaT: int = None,
        space_reso: Union[float,int] = None
        ) -> xr.DataArray :
    """Complete the coordinates. The objective is to fit with the rest
    of the data as Taxis or Fedding Habitat fields.
    
    Warning
    -------
    - space_reso and deltaT in effort file are independant from the
    space_reso in SEAPODYM configuration file.
    - deltaT == 30 in SEAPODYM mean that we use a monthly time
    resolution.
    """
    
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

def normalizeCoords(
        data_set: xr.Dataset, deltaT: Union[int,List[int]] = None,
        space_reso: Union[Union[float,int],List[Union[float,int]]] = None
        ) -> xr.Dataset :
    """Complete the time or/and space coordinates. The objective is to
    fit with the rest of the data as Taxis or Fedding Habitat fields.

    Parameters
    ----------
    data_set : xr.Dataset
        The data to normalize.
    deltaT : Union[int,List[int]], optional
        The number of day for each step. 30 is corresponding to month
        resolution.
    space_reso : Union[Union[float,int],List[Union[float,int]]], optional
        The space resolution in degrees.

    Returns
    -------
    xr.Dataset
        Same dataset as the `data_set` argument but with full
        coordinates on the axes chosen by the user.

    Raises
    ------
    ValueError
        Must specify at least one  of deltaT or space_reso arguments.
    ValueError
        deltaT must be int or list of size len(data_set).
    ValueError
        space_reso must be int/float or list of size len(data_set).
    """
    
    if (deltaT is None) and (space_reso is None):
        raise ValueError("If deltaT AND space_reso are None, nothing will happen.")
    
    deltaT = np.array(deltaT).ravel()
    if deltaT.size == 1 :
        deltaT = deltaT.repeat(len(data_set))
    elif deltaT.size != len(data_set) :
        raise ValueError("deltaT must be int or list of size len(data_set)")
    
    space_reso = np.array(space_reso).ravel()
    if space_reso.size == 1 :
        space_reso = space_reso.repeat(len(data_set))
    elif space_reso.size != len(data_set) :
        raise ValueError("space_reso must be int/float or list of size len(data_set)")
    
    f_effort_dict = {}
    for pos, f_name in enumerate(data_set) :
        f_effort_dict[f_name] = _completeCoords(data_set[f_name],
                                                deltaT[pos],
                                                space_reso[pos])
    
    return xr.Dataset(f_effort_dict, attrs=data_set.attrs)

## TOOL ################################################################

def sumDataSet(fisheries: xr.Dataset, name: str = None) -> xr.DataArray :
    """Sum all DataArray in the Dataset passed in argument.

    Parameters
    ----------
    fisheries : xr.Dataset
        Dataset to sum.
    name : str; optional
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        The sum of all the DataArray in `fisheries` argument.
    """
    
    f_name_list = list(fisheries)
    sum = np.nan_to_num(fisheries[f_name_list[0]])
    
    for f in f_name_list[1:] :
        sum = sum + np.nan_to_num(fisheries[f])
    
    return xr.DataArray(data=sum,
                        coords=fisheries.coords,
                        name=name,
                        attrs={"Type":"Sum of "+fisheries.attrs["Type"],
                               **fisheries.attrs})

def toTextFile(
        fisheries: Union[dict, pd.DataFrame], filepath: str = './ouput.txt',
        fishery_label: str = "f", gear_label: str = "gr",
        year_label: str = "yr", month_label: str = "mm", day_label: str = "dd",
        latitude_label: str = "lat", longitude_label: str = "lon",
        effort_label: str = "E", catch_label: str = "C"
        ) -> None :
    """Export a group of fisheries entries to a text file.

    Parameters
    ----------
    fisheries : Union[dict, pd.DataFrame]
        The fisheries entries. Must be a dictionary where values are
        fisheries entries or a pandas.Dataframe.
    filepath : str, optional
        The path to the file you want to create.

    Raises
    ------
    ValueError
        Argument fisheries must be a dict or a pandas DataFrame.
    """

    if ((not isinstance(fisheries, dict))
        and (not isinstance(fisheries, pd.DataFrame))) :
        raise ValueError(
            ("Argument fisheries must be a dict or a pandas DataFrame. Actual "
             "type is {}").format(type(fisheries)))

    if isinstance(fisheries, dict) :
        fisheries = pd.concat(fisheries.values())
    
    fisheries = fisheries.sort_values([fishery_label, year_label, month_label,
                                       day_label, latitude_label, longitude_label])
    fisheries = fisheries[[fishery_label, year_label, month_label, day_label,
                          gear_label, latitude_label, longitude_label,
                          effort_label, catch_label]]
    fisheries.to_csv(filepath, sep='\t', index= False)
        
## CLEANING DATA #######################################################

def removeEmptyEntries(
        fishery: pd.DataFrame, verbose: bool = False, fisheries_label: str = 'f',
        effort_label: str = 'E', catch_label: str = 'C'
        ) -> pd.DataFrame :
    """Remove entries in `fishery` Dataframe which have no effort and
    catch values.

    Parameters
    ----------
    fishery : pd.DataFrame
        Dataframe containing fisheries entries.
    verbose : bool, optional

    Returns
    -------
    pd.DataFrame
        Same as the `fishery` argument but cleaned of empty entries.
    """
    
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

def removeNoCatchEntries(
        fishery: pd.DataFrame, verbose: bool = False,
        fisheries_label: str = 'f', catch_label: str = 'C', 
        ) -> pd.DataFrame :
    """Remove entries in `fishery` Dataframe which have no catch values.

    Parameters
    ----------
    fishery : pd.DataFrame
        Dataframe containing fisheries entries.
    verbose : bool, optional

    Returns
    -------
    pd.DataFrame
        Same as the `fishery` argument but cleaned of "no catch" entries.
    """
    
    if verbose :
        print("Removed %d entries without catch."%(
            len(fishery[fishery[catch_label]==0])))
        print('Number of entries without catch per fishery :\n',
              fishery[fishery[catch_label]==0][
                  [fisheries_label,catch_label]].groupby(
                      [fisheries_label]).count().rename(
                          columns={catch_label:'no catch'}).T)
        
    return fishery[fishery[catch_label]!=0]

def plotByGear(
        fishery: pd.DataFrame, choose_gear: List = None,
        figsize: Tuple[float,float] = (10,10), label_fontsize: float = 15,
        legend_fontsize: float = 15, title: str = None, title_fontsize: int = 24,
        gear_label: str = 'gr', effort_label: str = 'E', catch_label: str = 'C'
        ) -> None :
    """Plot entries from a DataFrame as (x:catch, y:effort) for chosen
    gear.

    Parameters
    ----------
    fishery : pd.DataFrame
        Dataframe containing fisheries entries.
    choose_gear : List, optional
        List of gear you want to plot in the figure. If None, all the
        gear in the `fishery` Dataframe will be shown.
    figsize : Tuple[float,float], optional
        See Matplotlib arguments.
    label_fontsize : float, optional
        Size of the 'catch' and 'effort' labels. See Matplotlib
        arguments.
    legend_fontsize : float, optional
        Size of the Gears labels in legend. See Matplotlib arguments.
    title : str, optional
        Title of the figure. See Matplotlib arguments.
    title_fontsize : int, optional
        Size of the title. See Matplotlib arguments.
    
    Warnings
    --------
    Don't use `groupByFisheries()` before using this function.
    """
    
    plt.subplots(1,1,figsize=figsize)
    if choose_gear is None :
        gr = fishery[gear_label].explode().unique()
    else :
        gr = np.array(choose_gear).ravel()
        
    #color = ['b','g','r','c','m','y','k']
    color = ['blue','orange','green','red','purple','brown','pink','gray',
             'olive','cyan']

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

def predictEffort(
        fishery: pd.DataFrame, conserve_no_catch: bool = False,
        conserve_empty: bool = False, gear_to_choose: str = 'K',
        catch_label: str = 'C', gear_label: str = 'gr',
        effort_label: str = 'E'
        ) -> pd.DataFrame :
    """
    Predict effort (where effort is null but catch is not) using linear
    regression. Will use the `gear_to_choose` gear as model for the
    regression (only when effort and catch are not equal to zero).
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
    """Predict effort (where effort is null but catch is not) using linear
    regression.

    Parameters
    ----------
    fishery : dict
        Key are fisheries name and value are Dataframe containing
        catch and effort entries.
    conserve_no_catch : bool, optional
        Conserve or not entries with catch equal to zero.
    conserve_empty : bool, optional
        Conserve or not entries with catch and effort equal to zero.
    gear_to_choose : dict, optional
        Keys are fisheries name you want to perform the prediction to
        with a specific gear name. Values are gear to use for the
        prediction. If the name of a fishery is not in the keys of this
        argument, the gear chosen for that fishery is the most present
        in its entries.

    See Also
    --------
    - predictEffort()
    - sklearn.linear_model.LinearRegression()

    Returns
    -------
    dict
        Modified `fishery` dictionary with new effort values where
        effort was equal to zero and catch was not.
    """
    dict_update = {}
    for key, value in fishery.items() :
        if key in gear_to_choose :
            gear = gear_to_choose[key]
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

def effortByFishery(
        filepath: str, time_reso: int, space_reso: float, skiprows: int = 0,
        columns_name: List[str] = None, removeNoCatch: bool = False,
        remove_fisheries: List[Union[float,int,str]] = None,
        predict_effort: bool = False, verbose: bool = False,
        fishery_label: str = "f"
        ) -> xr.Dataset :
    """This wrapper will perform common manipulation which is :
    - readFile
    - Optional -> remove some fisheries (`remove_fisheries`)
    - removeEmptyEntries
    - Optional -> removeNoCatchEntries
    - separateFisheries
    - Optional -> predictEffortAllFisheries
    - rescaleFisheries
    - groupByFisheries
    - fisheriesToDataSet
    - normalizeCoords

    Parameters
    ----------
    filepath : str
        Path to the file containing all catch and effort fisheries
        entries.
    time_reso : int
        The time resolution in days. 30 is monthly. 
    space_reso : float
        The space resolution in degrees.
    skiprows : int, optional
        Line numbers to skip (0-indexed) or number of lines to skip
        (int) at the start of the file.
    removeNoCatch : bool, optional
        Remove entries where catch is equal to zero.
    remove_fisheries : List[Union[float,int,str]], optional
        A list of fisheries name you want to remove from the final
        DataSet.
    predict_effort : bool, optional
        Predict effort where effort is equal to zero and catch is not.
    verbose : bool, optional

    Returns
    -------
    xr.Dataset
        Contains a DataArray for each fishery in the text file passed as
        argument. Each DataArray corresponds to the effort of this
        fishery according to the temporal and spatial coordinates.
    """

    fisheries_effort = readFile(filepath, skiprows)
    if remove_fisheries is not None :
        for f in remove_fisheries :
            fisheries_effort = fisheries_effort.drop(
                fisheries_effort[fisheries_effort[fishery_label]==f].index)
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
