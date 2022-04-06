"""
This module was written to produce fishing effort fields from a text
file listing all the surveys of various fisheries. 


Routine Listings
----------------
The most common use of this set of functions is described in a wrapper
function : `effortByFishery`. It corresponds to successive use of :

- readFile
- removeEmptyEntries
- *(Optional -> removeNoCatchEntries)*
- separateFisheries
- *(Optional -> predictEffortAllFisheries)*
- rescaleFisheries
- groupByFisheries
- fisheriesToDataSet
- normalizeCoords

"""


import enum
from typing import List, Tuple, Union
from unittest import skip
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from sklearn import linear_model

_labels = {
    'fishery_name':'f',
    'year':'yr',
    'month':'mm',
    'day':'dd',
    'latitude':'lat',
    'longitude':'lon',
    'resolution':'res',
    'gear':'gr',
    'effort':'E',
    'catch':'C',
}

def readFiles(
        filepath: Union[str,List[str]], skiprows: Union[int,List[int]] = 2,
        ) -> pd.DataFrame :
    """Read multiple fisheries effort/catch files and return a dataframe
    containing all the entries aggregated.

    Parameters
    ----------
    filepath : str
        The path(s) to the file(s).
    skiprows : int, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file(s).

    Returns
    -------
    pd.DataFrame
        A Dataframe containing all the fisheries entries.
    """
    
    filepath = np.ravel(filepath)
    skiprows = np.ravel(skiprows)
    if (skiprows.size == 1) and (filepath.size != 1) :
        skiprows = np.tile(skiprows, filepath.size)
    elif skiprows.size != filepath.size :
        raise ValueError(('Specify a number of row to skip for each filepath '
                         'or one for every paths. filepath size is {} and '
                         'skiprows size is {}.').format(filepath.size, skiprows.size))
    
    df_list = []
    for path, skip in zip(filepath, skiprows) :
        df_list.append(pd.read_table(path, skiprows=skip))
    
    fishery = pd.concat(df_list)
    
    fishery_time = pd.to_datetime(
            fishery.rename(
                columns={_labels['year']:"year",
                         _labels['month']:"month",
                         _labels['day']:"day"}
                )[["year","month","day"]])
    
    
    return pd.DataFrame({
        _labels['fishery_name']:fishery[_labels['fishery_name']].astype(str),
        "time":fishery_time,
        _labels['latitude']:fishery[_labels['latitude']],
        _labels['longitude']:fishery[_labels['longitude']],
        _labels['resolution']:fishery[_labels['resolution']],
        _labels['gear']:fishery[_labels['gear']],
        _labels['effort']:fishery[_labels['effort']]
    })

def _reindexSpace(
        effort_df: pd.DataFrame, lat_coords: np.ndarray, lon_coords: np.ndarray,
        limit: float
        ) -> pd.DataFrame :
    """
    Parameters
    ----------
    limit : int
        The maximum difference in degrees between the original lat/lon
        and the reindexed one. Should be equal to space_reso.
    """
    
    def nearestSpace(element, coordinate, limit) :
        dist = np.absolute(coordinate-element)
        if dist.min() > limit :
            return element
        else :
            return coordinate[dist.argmin()]
    
    effort_df[_labels['latitude']] = effort_df[
        _labels['latitude']].apply(nearestSpace, args=(lat_coords,limit))
    effort_df[_labels['longitude']] = effort_df[
        _labels['longitude']].apply(nearestSpace, args=(lon_coords,limit))
    
    return effort_df

def _reindexTime(
        effort_df: pd.DataFrame, time_coords: np.ndarray, limit: int
        ) -> pd.DataFrame :
    """
    Parameters
    ----------
    limit : int
        The maximum difference in day between the original date and
        the reindexed one. Should be equal to detlaT.
    """
    
    def nearestTime(timestep, time_coords, limit) :
        timestep = np.datetime64(timestep, 'D')
        time_coords = time_coords.astype('datetime64[D]')
        dist = np.absolute(time_coords-timestep)
        if dist.min() > int(limit) :
            return timestep
        else :
            return time_coords[dist.argmin()]

    effort_df['time'] = effort_df['time'].apply(nearestTime, 
                                                args=(time_coords,limit))
    return effort_df

def reindexCoordinates(
        effort_df: pd.DataFrame, coords: xr.Coordinate, time_limit: float, 
        space_limit: int
        ):
    effort_df = _reindexTime(effort_df, coords['time'].data, time_limit)
    effort_df = _reindexSpace(effort_df, coords[_labels['latitude']].data,
                              coords[_labels['longitude']].data, space_limit)
    return effort_df

def selectFisheries(
        effort_df: pd.DataFrame, selected_fisheries: Union[str, List[str]],
        ) -> pd.DataFrame:
    
    for index, name in enumerate(np.ravel(selected_fisheries)) :
        if index == 0 :
            condition = effort_df[_labels['fishery_name']] == name
        else :
            condition |= effort_df[_labels['fishery_name']] == name
    
    return effort_df[condition]

def rescaleFisheries(effort_df: pd.DataFrame, resolution: float = None) :
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
    min_data_resolution = effort_df[_labels['resolution']].min()
    if resolution is None :
        resolution = min_data_resolution
    elif resolution > min_data_resolution :
        raise ValueError(('Downscaling is not supported for now. Use '
                         'reindexSpace instead. resolution = {} and '
                         'min_data_resolution = {}'
                         ).format(resolution, min_data_resolution))
    
    def computeSubfisheriesPositions(resolution, min_resolution, latitude, longitude) :
            
        # assert latitude >= -90 and latitude <= 90, "Warning : -90 <= latitude <= 90"
        # assert longitude >= 0 and longitude <= 360, "Warning : 0 <= longitude <= 360"
        
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
        mat_lon = np.tile(
            np.linspace(lon_inf, lon_sup, nb_elem_lon)%max_longitude, nb_elem_lat)
        
        return mat_lat, mat_lon
    
    def createSubfisheries(row,  min_resolution) :
        resolution = row[_labels['resolution']]
        effort = row[_labels['effort']]
        
        latitude, longitude = computeSubfisheriesPositions(
            resolution, min_resolution, row[_labels['latitude']], row[_labels['longitude']])
        sub_fisheries_number = latitude.size
        sub_fisheries_list = np.arange(sub_fisheries_number)
        effort /= sub_fisheries_number

        return [sub_fisheries_list,latitude,longitude,effort]

    def computeRightPosition(row):
        return [row['sub_fishery_latitude'][row['sub_fishery']],
                row['sub_fishery_longitude'][row['sub_fishery']]]
    
    update_index = ['sub_fishery', 'sub_fishery_latitude', 'sub_fishery_longitude',
                    _labels['effort'],]
    
    f_dict = {}
    for f_name in effort_df[_labels['fishery_name']].unique() :
        f_dict[f_name] = effort_df.loc[effort_df[_labels['fishery_name']] == f_name]
    
    sub_fisheries = {}
    for name, fishery in f_dict.items() : 
        
        if max(fishery[_labels['resolution']].unique()) > resolution :
            
            sub_fishery = fishery.copy()
            sub_fishery[update_index] = fishery.apply(
                createSubfisheries, args=(resolution,), axis=1, result_type='expand')

            sub_fishery = sub_fishery.explode('sub_fishery', ignore_index=True)
            
            sub_fishery[[_labels['latitude'],_labels['longitude']]] = (
                sub_fishery.apply(computeRightPosition, axis=1, result_type='expand'))
            sub_fishery = sub_fishery.drop(
                columns=['sub_fishery_latitude', 'sub_fishery_longitude', 'sub_fishery'])
            
            sub_fishery[_labels['resolution']] = resolution
            sub_fisheries[name] = sub_fishery
        else :
            sub_fisheries[name] = fishery.copy()

    for k, v in sub_fisheries.items() :
        v['f'] = k

    return pd.concat(sub_fisheries.values())

def groupByFisheries(
        effort_df : pd.DataFrame, multi_index = False
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
    
    effort_df = effort_df.groupby(
        [_labels['fishery_name'],'time',_labels['latitude'],_labels['longitude']]
        ).aggregate(
            {_labels['gear']:np.unique, _labels['resolution']:np.unique,
             _labels['effort']:np.sum})
    
    if not multi_index :
        effort_df = effort_df.reset_index()
    
    return effort_df

def fisheriesToDataSet(
        effort_df : pd.DataFrame
        ) -> xr.Dataset :
    """Converts the fisheries DataFrame into a Dataset. Each unique
    fishery name is converted into a DataArray.

    See also
    --------
    groupByFisheries
    
    Returns
    -------
    xr.Dataset
        A Dataset that contains effort entries for each fishery.
    """
    
    tmp_dict = {}
    for f in effort_df['f'].unique():

        tmp = effort_df[effort_df['f'] == f]
        update_dict = {
            _labels['gear']:list(effort_df[_labels['gear']].explode().unique()),
            _labels['resolution']:list(effort_df[_labels['resolution']].explode().unique())}

        tmp = tmp[
            ['time',_labels['latitude'],_labels['longitude'],
            _labels['effort']]]

        tmp = tmp.groupby(['time',_labels['latitude'],_labels['longitude']]
                        ).sum().to_xarray()['E']
        
        tmp.name = f
        tmp.attrs.update(update_dict)
        
        tmp_dict[f] = tmp
        
    return xr.Dataset(tmp_dict)

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
                        attrs={"Type":"Sum of Effort",
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
        fishery: pd.DataFrame, verbose: bool = False
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
        print("Removed %d empty entries."%(len(fishery[fishery[_labels['effort']]==0])))
    return fishery[fishery[_labels['effort']]!=0]

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

def _predictEffort(
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
    predictEffort
    sklearn.linear_model.LinearRegression

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

        new_value = _predictEffort(value,conserve_no_catch,conserve_empty,gear,
                                  catch_label,gear_label,effort_label)
        dict_update[key] = new_value
    
    fishery.update(dict_update)
    
    return fishery

## WRAPPER #############################################################

# TODO : since this module has changed, the doc string should be rewrited
def effortByFishery(
        filepath: Union[str,List[str]], space_reso: float, time_reso: int,
        coords: xr.Coordinate, skiprows: Union[int,List[int]] = 2,
        selected_fisheries: Union[str,List[str]] = None,
        predict_effort: bool = False, verbose: bool = False
        ) -> xr.Dataset :
    """This wrapper will perform common manipulation which is :
    - readFile
    - Optional -> remove some named fisheries (`remove_fisheries`)
    - removeEmptyEntries
    - Optional -> removeNoCatchEntries
    - separateFisheries
    
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
    remove_fisheries : List[Union[float,int,str]], optional
        A list of fisheries name you want to remove from the final
        DataSet.
    predict_effort : bool, optional
        Predict effort where effort is equal to zero and catch is not.

    Returns
    -------
    xr.Dataset
        Contains a DataArray for each fishery in the text file passed as
        argument. Each DataArray corresponds to the effort of this
        fishery according to the temporal and spatial coordinates.
    """

    fisheries_effort = readFiles(filepath, skiprows)
    if selected_fisheries is not None :
        fisheries_effort = selectFisheries(fisheries_effort, selected_fisheries)
    fisheries_effort = removeEmptyEntries(fisheries_effort, verbose)
    if predict_effort :
        raise ValueError('Effort prediction is not supported yet.')
        fisheries_effort = predictEffortAllFisheries(fisheries_effort)
    min_res = fisheries_effort[_labels['resolution']].min()
    if min_res < space_reso :
        fisheries_effort = rescaleFisheries(fisheries_effort)
    else :
        fisheries_effort = rescaleFisheries(fisheries_effort, resolution=space_reso)
    fisheries_effort = reindexCoordinates(fisheries_effort, coords, time_reso,
                                          space_reso)
    fisheries_effort = groupByFisheries(fisheries_effort)
    fisheries_effort = fisheriesToDataSet(fisheries_effort)
    fisheries_effort = fisheries_effort.reindex(coords)
    # print(fisheries_effort.sum())
    return fisheries_effort
