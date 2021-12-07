# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:39:38 2021
@author: Jules Lehodey
"""
from scipy.spatial.distance import cdist
from multiprocessing import cpu_count
from dask import dataframe as ddf
import pandas as pd
import xarray as xr
import numpy as np
import pygmt

class EffortRedistribution :
    """
    
    This class is used to distribute effort according to marine protected areas.

    Attributes
    ----------
    
    original_fisheries_dataframe : Pandas.Dataframe
    
    fisheries_dataframe : Pandas.Dataframe
    
    fisheries_dataframe_after_distribution : Pandas.Dataframe
    
    grouped_fisheries_dataframe_after_distribution : Pandas.Dataframe
    
    mask : Xarray.DataArray
    
    Examples
    --------
    
    Distribute effort and catch with MPA created by rectangular figures using the
    everywhere strategy and the Pandas.iterrows() method :
    
    >>> er = EffortRedistribution()
    >>> er.readCatchAndEffortFromFile(filepath)
    >>> figures = [{'lat':(0, 30), 'lon':(120, 160)},{'lat':(-10, 0), 'lon':(180, 200)}]
    >>> er.loadMaskAndDistributionZones(mask_method='rectangular_figures',
    ...     distribution_method='everywhere', figures=figures,
    ...     mask_parameters={'lat':(-50.5,60.5), 'lon':(80.5,300.5),'resolution':1.0})
    >>> er.rescaleFisheries()
    >>> er.classifyFisheries()
    >>> er.distributeEffort()
    >>> er.groupSubfisheries()
    >>> er.checkSummary()
    >>> er.saveInTextFile(filepath_2)
    
    Distribute effort and catch with MPA loaded from text file using the
    manhattan strategy and the Dask.apply() method :
    
    >>> er = EffortRedistribution()
    >>> er.readCatchAndEffortFromFile(filepath)
    >>> er.loadMaskAndDistributionZones(
    ...     mask_method='file', distribution_method='manhattan', filepath=mask_filepath,
    ...     masked_values=[-50], using_xml_config_values=True, manhattan_distance=10,
    ...     mask_parameters={'lat':(-54.625,65.375), 'lon':(88.375,290.375),'resolution':0.25})
    >>> er.rescaleFisheries()
    >>> er.classifyFisheries(strategy='parallel_dask')
    >>> er.distributeEffort()
    >>> er.groupSubfisheries()
    >>> er.checkSummary()
    >>> er.saveInTextFile(filepath_2)
    
    """

    def __init__(self):
        
        self.grouped_fisheries_dataframe_after_distribution = None
        self.fisheries_dataframe_after_distribution = None
        self.fisheries_dataframe = None
        self.original_fisheries_dataframe = None
        self.mask = None
    
    def __computeSubfisheriesPositions__(self, resolution, min_resolution, latitude, longitude) :
        
        # When a fishery has a low resolution, it is rescaled by creating several
        # sub-fisheries distributed throughout the area.
        # WARNING : Pole gestion is not implemented
        #           Actual effect is an aggregation (reflect on limits north and south)
        # TODO : Take into account these particular cases
        
        assert latitude >= -90 and latitude <= 90, "Warning : -90 <= latitude <= 90"
        assert longitude >= 0 and longitude <= 360, "Warning : 0 <= longitude <= 360"
        
        X = int(resolution / min_resolution)
        
        # ODD #################################################################
        if X % 2 == 1 : 
            middle = X // 2
        # EVEN ################################################################
        else :
            middle = (X // 2) - 1
    
        latitude_list = []
        min, max = -90, 90
        for layer in range(X) :
            layer_latitude_list = []
            for _ in range(X) :
                tmp = latitude + (middle - layer)
                if tmp > max :
                    tmp = max - (tmp % max)
                if tmp < min :
                    tmp = -(max - (abs(tmp) % max))
                layer_latitude_list.append(tmp)
            latitude_list.append(layer_latitude_list)
    
        longitude_list = []
        min, max = 0, 360
        for pos in range(X) :
            layer_longitude_list = []
            for layer in range(X) :
                tmp = longitude + (layer - middle)
                if tmp >= max :
                    tmp = min + (tmp % max)
                if tmp < min :
                    tmp = max + tmp
                layer_longitude_list.append(tmp)
            longitude_list.append(layer_longitude_list)
        
        # Flat lists are returned
        return ([item for sublist in latitude_list for item in sublist],
                [item for sublist in longitude_list for item in sublist])
    
    def __createSubfisheries__(self, row, min_resolution) :
        resolution = row['resolution']
        effort = row['effort']
        catch = row['catch']
        
        latitude, longitude = self.__computeSubfisheriesPositions__(
            resolution, min_resolution, row['latitude'], row['longitude'])
        sub_fisheries_number = int(resolution / min_resolution)**2
        sub_fisheries_list = [x for x in range(sub_fisheries_number)]
        effort /= sub_fisheries_number
        catch /= sub_fisheries_number
        
        # Returning a list is the most efficient way to compute sub-fisheries
        # Tried pandas.Series and partial pandas.Series with update.
        return [sub_fisheries_list,latitude,longitude,effort,catch]
    
    def __loadMaskFromTexte__(self, path, masked_values,
                              mask_parameters={'lat':(-90,90),
                                               'lon':(0,360),
                                               'resolution':1.,
                                               'using_xml_config_values':False}) :
        
        min_lat = mask_parameters['lat'][0]
        max_lat = mask_parameters['lat'][1]
        min_lon = mask_parameters['lon'][0]
        max_lon = mask_parameters['lon'][1]
        resolution = mask_parameters['resolution']
        
        # i.e. loadMaskAndDistributionZones Doc. string
        if mask_parameters['using_xml_config_values'] :
            min_lat += (resolution / 2)
            min_lon += (resolution / 2)
            max_lat -= (resolution / 2)
            max_lon -= (resolution / 2)
        
        # Flip the latitude is needed to avoid to inverse numpy array in xr.DataArray
        latitude_index = np.flip(np.arange(min_lat, max_lat+resolution, resolution))
        # Longitude = 360 is equivalent to longitude = 0
        longitude_index = np.arange(min_lon, max_lon+(resolution if max_lon < 360 else 0), resolution)
            
        initial_mask = np.loadtxt(path, dtype=np.int)
    
        
        contained_values = set(initial_mask.ravel())
        
        for value in masked_values :
            assert value in contained_values, "Value %f is not in the text file %s" % (np.float64(value), path)

        self.mask = xr.DataArray(data=np.isin(initial_mask, masked_values),
                                 dims=['lat','lon'],
                                 coords={'lat':latitude_index,'lon':longitude_index},
                                 attrs={'resolution':resolution})
    
    def __loadMaskFromrectangularFigures__(self, figures_list,
                                            mask_parameters={'lat':(-90,90),
                                                             'lon':(0,360),
                                                             'resolution':1.,
                                                             'using_xml_config_values':False}) :
    
        """
        Exemple : 
                    >>> figures = ({'lat':(-10, 30), 'lon':(100, 140)},
                    >>>            {'lat':(30,40), 'lon':(110, 120)})
                    >>> __loadMaskFromrectangularFigures__(figures,
                            mask_parameters={'lat':(-20,50), 'lon':(80,200),
                                'resolution':0.25, 'using_xml_config_values':False})
        """
        
        for figure, pos in zip(figures_list, range(1,len(figures_list)+1)) :
            assert figure['lat'][0] >= -90 and figure['lat'][1] <= 90, "Error with min lat in figure %d : Latitude min = -90 and max =90" % pos
            assert figure['lon'][0] >= 0 and figure['lon'][1] <= 360, "Error with min lat in figure %d : longitude min = 0 and max = 360" % pos
            assert figure['lat'][0] >= mask_parameters['lat'][0], "Error with min lat in figure %d : %f should be >= %f" % (pos, figure['lat'][0], mask_parameters['lat'][0])
            assert figure['lat'][1] <= mask_parameters['lat'][1], "Error with max lat in figure %d : %f should be <= %f" % (pos, figure['lat'][1], mask_parameters['lat'][1])
            assert figure['lon'][0] >= mask_parameters['lon'][0], "Error with min lon in figure %d : %f should be >= %f" % (pos, figure['lon'][0], mask_parameters['lon'][0])
            assert figure['lon'][1] <= mask_parameters['lon'][1], "Error with max lon in figure %d : %f should be <= %f" % (pos, figure['lon'][1], mask_parameters['lon'][1])
            assert figure['lat'][0] < figure['lat'][1], "Error with lat in figure %d : %f should be < %f" % (pos, figure['lat'][0], figure['lat'][1])
            assert figure['lon'][0] < figure['lon'][1], "Error with lon in figure %d : %f should be < %f" % (pos, figure['lon'][0], figure['lon'][1])
        
        min_lat = mask_parameters['lat'][0]
        max_lat = mask_parameters['lat'][1]
        min_lon = mask_parameters['lon'][0]
        max_lon = mask_parameters['lon'][1]
        resolution = mask_parameters['resolution']
        
        # i.e. Warning in Doc string
        if mask_parameters['using_xml_config_values'] :
            min_lat += (resolution / 2)
            min_lon += (resolution / 2)
            max_lat -= (resolution / 2)
            max_lon -= (resolution / 2)
        
        resolution_multiplicator = 1. / resolution
        
        # Adding 1 to take into account the last value
        lat_size = ((max_lat - min_lat) * resolution_multiplicator) + 1
        # If longitude max is 360 (which is the 0.0 position) adding is not needed
        lon_size = ((max_lon - min_lon) * resolution_multiplicator) + (1 if max_lon < 360 else 0)
        
        mask_np = np.full((int(lat_size), int(lon_size)), 0.0)
        
        for figure in figures_list :
            figure_lat_min = figure['lat'][0] - min_lat
            figure_lat_max = figure_lat_min + (figure['lat'][1] - figure['lat'][0])
            figure_lat_min *= resolution_multiplicator
            figure_lat_max *= resolution_multiplicator
            
            figure_lon_min = figure['lon'][0] - min_lon
            figure_lon_max = figure_lon_min + (figure['lon'][1] - figure['lon'][0])
            figure_lon_min *= resolution_multiplicator
            figure_lon_max *= resolution_multiplicator
            
            mask_np[int(figure_lat_min):int(figure_lat_max), int(figure_lon_min):int(figure_lon_max)] = 1.0
        
        # need to flip up and down because min (should be down) position is 0
        # which is the up position
        mask_np = np.flipud(mask_np)
        
        # Flip the latitude is needed to avoid to inverse numpy array in xr.DataArray
        latitude_index = np.flip(np.arange(min_lat, max_lat+resolution, resolution))
        # Longitude = 360 is equivalent to longitude = 0
        longitude_index = np.arange(min_lon, max_lon+(resolution if max_lon < 360 else 0), resolution)
        
        self.mask = xr.DataArray(data=mask_np,
                               dims=['lat','lon'],
                               coords={'lat':latitude_index,'lon':longitude_index},
                               attrs={'resolution':resolution})
    
    def __manhattanDistance__(self, mask, seed_value=1):
    
        seed_mask = (mask == seed_value)
        z         = np.argwhere(seed_mask)
        z_inv     = np.argwhere(~seed_mask)
    
        output = np.zeros(mask.shape, dtype=int)
        output[tuple(z_inv.T)] = cdist(z, z_inv, 'cityblock').min(0).astype(int)
        
        return output
    
    def __whoIsReceiving__(self, strategy, **kargs) :
        """
        strategy : String
                   {'everywhere' | 'manhattan' | 'expand'}
                   
        args     : distance as int if strategy is manhattan
        """
        if strategy == 'everywhere' :
            everywhere_mask = np.where(self.mask  == 0.0, 2.0, 1.0)
            self.mask = xr.DataArray(data=everywhere_mask,
                                     dims=self.mask .dims,
                                     coords=self.mask.coords,
                                     attrs=self.mask.attrs)
        
        elif strategy == 'manhattan' :
            if not 'manhattan_distance' in kargs :
                raise ValueError("Error : The strategy \'%s\' require manhattan_distance > 0." % strategy)
            
            manhattan_from_zones = ((self.__manhattanDistance__(self.mask.data, seed_value=True)
                                     <= (kargs['manhattan_distance'] / kargs['resolution']))
                                    * 2) - self.mask
            self.mask = xr.DataArray(data=manhattan_from_zones,
                                     dims=self.mask.dims,
                                     coords=self.mask.coords,
                                     attrs=self.mask.attrs)

        elif strategy == 'expand' :
            raise NotImplementedError('The %s strategy has not been implemented yet.')

        else :
            raise ValueError("Error : The strategy '%s' is not supported. Use {'everywhere' | 'manhattan' | 'expand'} instead." % strategy)

    def __classifyFisheriesUsingDask__(self, position_index=('sub_fisheries_latitude',
                                                             'sub_fisheries_longitude'),
                                   output_index='classification', scheduler='multiprocessing',
                                   partitions_number=None, workers_number=None, sort=False) :

        if partitions_number is None :
            partitions_number = cpu_count() * 2
        
        df_dask = ddf.from_pandas(self.fisheries_dataframe, npartitions=partitions_number, sort=sort)
        
        lat, lon = position_index[0], position_index[1]
        func = lambda row, mask : mask.sel(lat=row[lat], lon=row[lon]).data[()]
        
        dask_serie_to_compute = df_dask.apply(func, axis=1, args=(self.mask,), meta=(None, 'float64'))
        
        if workers_number is None :
            return dask_serie_to_compute.compute(scheduler=scheduler)
        else :
            return dask_serie_to_compute.compute(scheduler=scheduler, num_workers=workers_number)
    
    def __classifyFisheriesUsingIterrows__(self, position_index=('sub_fisheries_latitude','sub_fisheries_longitude'),
                                   output_index='classification', sort=True) :
        
        lat, lon = position_index[0], position_index[1]
        
        if sort :
            fisheries = self.fisheries_dataframe.sort_values([lat, lon])
        
        output_list = []
        index_list=[]
        last_lat = np.NaN
        last_lon = np.NaN
        last_value = np.NaN
        value = 0.0
        
        for index, row in fisheries.iterrows() :
            row_lat = row[lat]
            row_lon = row[lon]
            # Note : First condition is evaluated before the second, may cost less
            #        execution time when redondent.
            # Warning : Xarray sel() function is not really efficient with labels
            if last_lat == row_lat and last_lon == row_lon :
                value = last_value
            else :
                value = self.mask.sel(lat=row_lat, lon=row_lon, method="nearest").data[()]
            
            index_list.append(index)
            output_list.append(value)
            
            last_lat = row_lat
            last_lon = row_lon
            last_value = value
            
        return pd.Series(output_list, index=index_list)
    
    def __readTextFile__(self, filepath, lines_to_skip=0, verbose=True) :
        
        f = open(filepath, "r")
        lines_list = f.readlines()
        
        
        if verbose :
            header = lines_list[lines_to_skip if lines_to_skip > 0 else 0]
            print("Check that this is the expected result, otherwise the 'lines_to_skip' value may be changed.\nThe header (containing column names) is :", header)
        
        # Remove unused lines
        lines_list = lines_list[lines_to_skip+1:]
        
        # Data lists
        fisheries           = []
        year, month, day    = [], [], []
        latitude, longitude = [], []
        effort, catch       = [], []
        resolution          = []
        gear                = []
        
        for line in lines_list :
            data_in_line = line.split()
        
            fisheries.append(int(data_in_line[0]))
            year.append(int(data_in_line[1]))
            month.append(int(data_in_line[2]))
            day.append(int(data_in_line[3]))
            gear.append(data_in_line[4])
            latitude.append(float(data_in_line[5]))
            longitude.append(float(data_in_line[6]))
            resolution.append(float(data_in_line[7]))
            effort.append(float(data_in_line[8]))
            catch.append(float(data_in_line[9]))
        
        # Verify all lists have same size
        assert len(fisheries) == len(year)
        assert len(year) == len(month)
        assert len(month) == len(day)
        assert len(day) == len(gear)
        assert len(gear) == len(latitude)
        assert len(latitude) == len(longitude)
        assert len(longitude) == len(resolution)
        assert len(resolution) == len(effort)
        assert len(effort) == len(catch)
        
        # Combine dates to convert in datetime64 type
        date = pd.DataFrame({'year':year,
                             'month':month,
                             'day':day})
        date = pd.to_datetime(date)
        date.name = 'date'
        
        # Create Dataframe to store all this data
        read_df = pd.DataFrame({'date':date,
                                'latitude':latitude,
                                'longitude':longitude,
                                'fishery':fisheries,
                                'gear':gear,
                                'resolution':resolution,
                                'effort':effort,
                                'catch':catch})
        
        size_before = read_df.index.size
        
        # TODO : tester la modif
        read_df = read_df.groupby(
            by=['date','fishery','latitude','longitude','gear','resolution']).agg(
            effort=pd.NamedAgg(column='effort', aggfunc="sum"),
            catch=pd.NamedAgg(column='catch', aggfunc="sum"))
        
        read_df = read_df.reset_index([0,1,2,3,4,5])
        
        size_after = read_df.index.size
        
        if size_before != size_after :
            print("Warning :\n---------\nMultiple entries with same date, position and fisheries at the same resolution.\nA 'group by' is done on the data using sum.")
            print("Size before 'group by' was : %d\nSize after 'group by' is : %d" % (size_before, size_after), end="\n\n")
            
        gb_df = read_df.groupby(by=['fishery']).agg({'resolution':lambda x: np.unique(x)})
        gb_df['len'] = gb_df['resolution'].apply(lambda x : x.size)
        res = gb_df.loc[gb_df['len'] > 1]
        res = res.reset_index([0])
        fishery_to_upscale = set(res['fishery'])
        if len(fishery_to_upscale) > 0 :
            print("Warning :\n---------\nFishery with multiple resolution : ", fishery_to_upscale)
            print("List of resolutions is :\n", res[['fishery','resolution']].to_string(index=False))
        
        return read_df
    
###############################################################################
# ------------------------ Fisheries manipulation --------------------------- #
###############################################################################
    
    def readCatchAndEffortFromFile(self, filepath, lines_to_skip=0, verbose=True):
        """
        
        Load data from a text file. This function must be called first. 

        Parameters
        ----------
        filepath : string
            The filepath to the text file contening the catchs and effort for
            each fisheries.
        lines_to_skip : int, optional
            The number of line to skip before encountering the header.
            The default is 0.
        verbose : boolean, optional
            Prints some informations on screen if True.
            The default is True.

        Returns
        -------
        None.

        """
        
        self.fisheries_dataframe = self.__readTextFile__(filepath, lines_to_skip, verbose)
        
        self.original_fisheries_dataframe = self.fisheries_dataframe.copy()
    
    def rescaleFisheries(self):
        """
        If there are fisheries which have a lower resolution, they are divided
        into sub-fisheries. Those sub-fisheries are distributed around the
        initial position.

        Returns
        -------
        None.

        """
        
        # Transforme
        min_resolution = np.min(self.fisheries_dataframe['resolution'])
        update_index = ['sub_fishery',
                        'sub_fishery_latitude','sub_fishery_longitude',
                        'sub_fishery_effort','sub_fishery_catch']
        
        self.fisheries_dataframe[update_index] = self.fisheries_dataframe.apply(
            self.__createSubfisheries__, args=(min_resolution,), axis=1, result_type='expand')
        
        # Explode
        self.fisheries_dataframe = self.fisheries_dataframe.explode('sub_fishery', ignore_index=True)
        
        # Compute Right Position
        computeRightPosition = (lambda row :
                                [row['sub_fishery_latitude'][row['sub_fishery']],
                                 row['sub_fishery_longitude'][row['sub_fishery']]])
        
        self.fisheries_dataframe[['sub_fishery_latitude','sub_fishery_longitude']] = (
            self.fisheries_dataframe.apply(computeRightPosition, axis=1, result_type='expand'))

    def loadMaskAndDistributionZones(self, mask_method, distribution_method,
                                     mask_parameters={'lat':(-90,90),'lon':(0,360),'resolution':1.},
                                     using_xml_config_values=False, **kargs) :
        """
        
        Load a mask from a text file or create it from rectangular figures. This
        mask correspond to the marine protected area.
        After what the zone where to distribute the effort is compute according
        to two strategy : 'everywhere' or 'manhattan'.

        Parameters
        ----------
        mask_method : string
            Must be 'file' or 'rectangular_figures'.
            - 'file' : load a mask from a text file.
            - 'rectangular_figures' : reate a mask with from rectangular figures.
        distribution_method : string
            Must be 'everywhere' or 'manhattan'.
            - 'everywhere' :
            - 'manhattan' :
        mask_parameters : TYPE, optional
            DESCRIPTION. The default is {'lat':(-90,90),'lon':(0,360),'resolution':1.}.
        using_xml_config_values : boolean, optional
            using_xml_config_values report to the xml file used by SEAPODYM.
            It uses particular values for latitude and longitude minimum and maximum.
            - One cell is added on each sides of the grid to run advection diffusion model.
            - Initial values are center of cells.
            Result in adding (resolution/2) to minimums and removing (resolution/2)
            to maximums.
            The default is False.
        **kargs : arguments
            According to your strategy choice, some additionnal arguments are required.
            File :
                 - 'filepath' : string -> the location of the text file used
                                to compute the mask
                - 'masked_values' : float -> the value of the cells considered as MPA
            rectangular Figures :   
                - 'figures' : list of dict -> contains a list of figures as dict
                              composed by 'lat' and 'lon' fields as follow
                              {'lat':(min_lat,max_lat),'lon':(min_lon,max_lon)}
            Manhattan :
                - 'manhattan_distance' : int -> maximum distance (in degree) between
                                         an acceptable cell and the mask.

        Returns
        -------
        None.

        """
        
        # Verify mask_parameters validity ----------------------------------- #
        if not (('lat' in mask_parameters) or ('lon' in mask_parameters)
                or ('resolution' in mask_parameters)) :
            
            raise ValueError("mask_parameters must contains : {'lat':(min,max), 'lon':(min,max), 'resolution':float}")
        
        assert mask_parameters['lat'][0] >= -90 and mask_parameters['lat'][1] <= 90, "mask_parameters : Latitude min = -90 and max =90"
        assert mask_parameters['lon'][0] >= 0 and mask_parameters['lon'][1] <= 360, "mask_parameters : longitude min = 0 and max = 360"
        assert mask_parameters['lat'][0] < mask_parameters['lat'][1], "Error with lat in mask_parameters : %f should be < %f" % (mask_parameters['lat'][0], mask_parameters['lat'][1])
        assert mask_parameters['lon'][0] < mask_parameters['lon'][1], "Error with lon in mask_parameters : %f should be < %f" % (mask_parameters['lon'][0], mask_parameters['lon'][1])
        
        mask_parameters['using_xml_config_values'] = using_xml_config_values
        
        # Compute mask ------------------------------------------------------ #
        if mask_method == 'file' :
            
            if not ('filepath' in kargs) :
                raise AttributeError("If mask_method is 'file' a 'filepath' argument is required. Exemple : filepath='./mask_name.txt")
            if not ('masked_values' in kargs) :
                raise AttributeError("If mask_method is 'file' a 'masked_values' argument (as a list) is required. Exemple : masked_values=[-1,-2]")
            self.__loadMaskFromTexte__(kargs['filepath'], kargs['masked_values'],
                                       mask_parameters)
        
        elif mask_method == 'rectangular_figures' :
            
            if not ('figures' in kargs) :
                raise AttributeError("If mask_method is 'rectangular_figures' a 'figures' argument (as a list of dict) is required. Exemple : figures=[{'lat':(15, 18), 'lon':(105, 108)}]")
            
            self.__loadMaskFromrectangularFigures__(kargs['figures'], mask_parameters)
        
        else :
            raise ValueError("mask_method must be : {'file' | 'rectangular_figures'}")
        
        
        
        # Add cells who receive effort -------------------------------------- #
        if distribution_method == 'everywhere' :
            self.__whoIsReceiving__(distribution_method)
        
        elif distribution_method == 'manhattan' :
            if not ('manhattan_distance' in kargs) :
                raise AttributeError("If distribution_method is 'manhattan' a 'manhattan_distance' argument is required. Exemple : manhattan_distance=2")
            self.__whoIsReceiving__(distribution_method,
                                    manhattan_distance=kargs['manhattan_distance'],
                                    resolution=mask_parameters['resolution'])
        
        else :
            raise ValueError("distribution_method must be : {'everywhere' | 'manhattan'}")
        
    def classifyFisheries(self, strategy='sequential_iterrows', output_index='classification',
                          position_index=('sub_fishery_latitude','sub_fishery_longitude'),
                          **kargs) :
        """
        
        Classify each statement as inside the marine protected area, outside but
        eligible (in right position) to receive effort redistribution or outside
        and non eligible.
        Two strategy are usable :
            - Use pandas iterator and compute each statement. The subtlety here
            is that many statements have same position. For those statements a
            huge part of the computation can be skip by sorting data according
            to position and memoring last statement.
            - Use dask parallelism. The data is split between core then a function
            is applied to each sub partition. See Dask apply() documentation.

        Parameters
        ----------
        strategy : string, optional
            The strategy to use to iterate through statements in data. Must be
            'sequential_iterrows' or 'parallel_dask'.
            The default is 'sequential_iterrows'.
        output_index : string, optional
            The name of index which will contain classification of each statement.
            The default is 'classification'.
        position_index : tuple, optional
            The name of index which will contain position (latitude and longitude)
            of each statement.
            The default is ('sub_fishery_latitude','sub_fishery_longitude').
        **kargs : TYPE
            If the used strategy is 'parallel_dask', user can specify 4 additional
            arguments :
                - scheduler : string -> see Dask compute() documentation.
                - partition_number : int -> see Dask compute() documentation.
                - workers_number : int -> number of core to use.
                - sort : boolean -> see Dask dataframe documentation.
            If the used strategy is 'sequential_iterrows', use can specify one
            additionnal argument :
                - sort : boolean -> Sort the data according to the position.
                If not specify, it is set to True (that is the recommanded value).

        Returns
        -------
        None.

        """
        
        if self.mask is None :
            raise ValueError("The mask needs to be initialized first.")
        
        # PARALLEL using DASK ----------------------------------------------- #
        if strategy == 'parallel_dask' :
            
            scheduler = kargs['scheduler'] if ('scheduler' in kargs) else 'multiprocessing'
            partitions_number = kargs['partitions_number'] if ('partitions_number' in kargs) else None
            workers_number = kargs['workers_number'] if ('workers_number' in kargs) else None
            sort = kargs['sort'] if ('sort' in kargs) else False
                
            self.fisheries_dataframe[output_index] = self.__classifyFisheriesUsingDask__(
                position_index=position_index, output_index=output_index, scheduler=scheduler,
                partitions_number=partitions_number, workers_number=workers_number, sort=sort)
        
        # SEQUENTIAL using ITERROWS ----------------------------------------- #
        elif strategy == 'sequential_iterrows' :

            sort = kargs['sort'] if ('sort' in kargs) else True
                
            self.fisheries_dataframe[output_index] = self.__classifyFisheriesUsingIterrows__(
                position_index=position_index, output_index=output_index, sort=sort)
            
        else :
            raise ValueError("strategy must be : {'sequential_iterrows' | 'parallel_dask'}")
    
    def distributeCatchAndEffort(
            self, time_index='date', effort_index='sub_fishery_effort',
            catch_index="sub_fishery_catch", classification_index='classification',
            mask_index={'inside_zones':1., 'affected_by_redistribution':2.},
            ratio_distribution_effort=1, ratio_conservation_effort=0,
            ratio_distribution_catch=None, ratio_conservation_catch=None,
            date_start=None, date_end=None,
            catch_to_everybody=False
            ) :
        """
        Distribute catch and effort of each statement localised in the marine
        protected area to others fisheries (with the same fisheries identifier)
        according to the mask (and strategy used to create it). The ratio
        arguments are used here to fully or partially distribute catch and effort.
        If the catch_to_everybody argument is set to True, all fisheries will
        receive catch redistribution (whatever their number).
        

        Parameters
        ----------
        time_index : string, optional
            The name of index containing date of each statement.
            The default is 'date'.
        effort_index : string, optional
            The name of index containing effort of each statement. 
            The default is 'sub_fishery_effort'.
        catch_index : string, optional
            The name of index containing catch of each statement. 
            The default is 'sub_fishery_catch'.
        classification_index : string, optional
            The name of index containing classification of each statement.
            The default is 'classification'.
        mask_index : dict, optional
            The value used to classify a statement inside the marine protected area
            or affected by the effort redistribution (i.e. while receive a part of it). 
            The default is {'inside_zones':1., 'affected_by_redistribution':2.0}.
        ratio_distribution_effort : float, optional
            Proportion of the effort which will be redistribute to others fisheries.
            The default is 1.
        ratio_conservation_effort : float, optional
            Proportion of the effort which will be conserved by fisheries inside
            the marine protected area.
            The default is 0.
        ratio_distribution_catch : float, optional
            Proportion of the catch which will be redistribute to others fisheries.
            If equal to None then it take the ratio_distribution_effort value.
            The default is None.
        ratio_conservation_effort : float, optional
            Proportion of the catch which will be conserved by fisheries inside
            the marine protected area.
            If equal to None then it take the ratio_conservation_effort value.
            The default is None.
        date_start : string, optional
            The date from which the distribution is made. If the date does not 
            exist, it begin with the first date contained in the data.
            The default is None
        date_end : string, optional
            The date until which the distribution is made. If the date does not 
            exist, it end with the last date contained in the data.
            The default is None
        catch_to_everybody : boolean, optional.
            If catch_to_everybody is True, all fisheries will receive catch
            redistribution, whatever their number.
            The default is False.

        Returns
        -------
        None.

        """
        
        assert ratio_distribution_effort >= 0 and ratio_conservation_effort >= 0
        
        self.ratio_distribution_effort = ratio_distribution_effort
        self.ratio_conservation_effort = ratio_conservation_effort
        
        if ratio_distribution_catch is None :
           ratio_distribution_catch = ratio_distribution_effort
        if ratio_conservation_catch is None :
           ratio_conservation_catch = ratio_conservation_effort
        self.ratio_distribution_catch = ratio_distribution_catch
        self.ratio_conservation_catch = ratio_conservation_catch
        
        updated_efforts = pd.Series(dtype='float64', name=effort_index)
        updated_catchs = pd.Series(dtype='float64', name=catch_index)
        
        # Argument verification
        time_serie = []
        if (date_start is not None) and (date_end is not None) and (date_end != date_start):
            dates = np.unique(self.fisheries_dataframe['date'].copy().sort_values())
            
            start = np.where(dates == np.datetime64(date_start))
            # if date is out of bounds
            if start[0].size == 0 :
                print("Warning : Starting date does not exist, taking first time step.")
                start = 0
            else :
                start = start[0][0]
                
            end = np.where(dates == np.datetime64(date_end))
            # if date is out of bounds
            if end[0].size == 0 :
                print("Warning : Ending date does not exist, taking last time step.")
                end = dates.size - 1
            else :
                end = end[0][0]
                
            for time_step in dates[start:end] :
                time_serie.append(np.datetime64(time_step))
        
        # Time serie computing
        dates = np.unique(self.fisheries_dataframe['date'].copy().sort_values())
        start = None
        end = None
        
        if (date_start is not None) :
            start = np.where(dates == np.datetime64(date_start))
            # if date is out of bounds
            if start[0].size == 0 :
                print("Warning : Starting date does not exist, taking first time step.")
                start = 0
            else :
                start = start[0][0]
        
        if (date_end is not None) :
            end = np.where(dates == np.datetime64(date_end))
            # if date is out of bounds
            if end[0].size == 0 :
                print("Warning : Ending date does not exist, taking last time step.")
                end = dates.size - 1
            else :
                end = end[0][0]
        
        if (date_start is not None) and (date_end is not None) :
            if (start != end) :
                for time_step in dates[start:end] :
                        time_serie.append(np.datetime64(time_step))
            else :
                time_serie.append(np.datetime64(dates[start]))
            time_serie = np.unique(time_serie)
        else :
            time_serie = dates
            
        redistribution_function = (
                    lambda effort, sum_inside, sum_redist :
                        effort + (sum_inside * (effort / sum_redist)))
        
        for time_step in time_serie :
            
            time_condition = self.fisheries_dataframe[time_index] == time_step
            fisheries_in_time_step = self.fisheries_dataframe.loc[time_condition]
            
            # CATCH TO EVERYBODY -------------------------------------------- #
            
            if catch_to_everybody :
                # INSIDE
                condition_inside = fisheries_in_time_step[classification_index] == mask_index['inside_zones']
                fisheries_inside_zones_catch = fisheries_in_time_step.loc[condition_inside, catch_index]
                sum_catch_time_step_inside = fisheries_inside_zones_catch.sum() * ratio_distribution_catch
                # OUTSIDE
                #condition_outside = fisheries_in_time_step[classification_index] != mask_index['inside_zones']
                condition_outside = ~condition_inside
                affected_by_redistribution_catch = fisheries_in_time_step.loc[condition_outside, catch_index]
                sum_catch_time_step_redistributed = affected_by_redistribution_catch.sum()
                # UPDATE
                catch_redistribution_update = (
                    affected_by_redistribution_catch.apply(
                        redistribution_function, args=(sum_catch_time_step_inside,
                                                       sum_catch_time_step_redistributed)))
                catch_inside_update = fisheries_inside_zones_catch.apply(
                    lambda x : x * ratio_conservation_catch)
                updated_catchs = updated_catchs.append(
                    (catch_redistribution_update, catch_inside_update))
            
            for fishery_ID in np.unique(fisheries_in_time_step['fishery']) :

                fishery_in_ID_and_time_step = fisheries_in_time_step.loc[fisheries_in_time_step['fishery'] == fishery_ID]
                
                # Warning : ------------------------------------------------- # 
                #
                #### Some effort can be loose when all sub-fisheries are in zone 1.
                #
                # length = len(fishery_in_ID_and_time_step['classification'])
                # if (np.sum(fishery_in_ID_and_time_step['classification'] == 1) == length) and (np.sum(fishery_in_ID_and_time_step['sub_fishery_effort']) > 0) :
                #     print(fishery_in_ID_and_time_step[['fishery','resolution','sub_fishery','sub_fishery_effort','classification']])#.to_string(index=False))
                #     print(np.sum(fishery_in_ID_and_time_step['sub_fishery_effort']))
                #
                #### Some effort will be lost if sub-fisheries which that have 
                # the opportunity to  receive effort have an initial effort of 0.
                #
                # if sum_effort_time_step_redistributed == 0 and fishery_ID == 10:
                #     print(fishery_in_ID_and_time_step[['fishery','classification','sub_fishery_effort']])
                # ----------------------------------------------------------- #
                
                # EFFORT TO SAME FISHERIES ---------------------------------- #
                
                # FISHERIES INSIDE PROTECTED ZONE
                condition_inside = fishery_in_ID_and_time_step[classification_index] == mask_index['inside_zones']
                fisheries_inside_zones_effort = fishery_in_ID_and_time_step.loc[condition_inside, effort_index]
                sum_effort_time_step_inside = fisheries_inside_zones_effort.sum() * ratio_distribution_effort
                
                # FISHERIES AFFECTED BY REDISTRIBUTION
                condition_redistribution = fishery_in_ID_and_time_step[classification_index] == mask_index['affected_by_redistribution']
                affected_by_redistribution_effort = fishery_in_ID_and_time_step.loc[condition_redistribution, effort_index]     
                sum_effort_time_step_redistributed = affected_by_redistribution_effort.sum()
                
                # UPDATE FISHERIES
                effort_redistribution_update = (
                    affected_by_redistribution_effort.apply(
                        redistribution_function, args=(sum_effort_time_step_inside,
                                                       sum_effort_time_step_redistributed)))
                effort_inside_update = fisheries_inside_zones_effort.apply(
                    lambda x : x * ratio_conservation_effort)
                updated_efforts = updated_efforts.append(
                    (effort_redistribution_update, effort_inside_update))
                
                # CATCH TO SAME FISHERIES ----------------------------------- #
                
                if not catch_to_everybody : 
                    fisheries_inside_zones_catch = fishery_in_ID_and_time_step.loc[condition_inside, catch_index]
                    sum_catch_time_step_inside = fisheries_inside_zones_catch.sum() * ratio_distribution_catch
                    affected_by_redistribution_catch = fishery_in_ID_and_time_step.loc[condition_redistribution, catch_index]
                    sum_catch_time_step_redistributed = affected_by_redistribution_catch.sum()
                    catch_redistribution_update = (
                        affected_by_redistribution_catch.apply(
                            redistribution_function, args=(sum_catch_time_step_inside,
                                                           sum_catch_time_step_redistributed)))
                    catch_inside_update = fisheries_inside_zones_catch.apply(
                        lambda x : x * ratio_conservation_catch)
                    updated_catchs = updated_catchs.append(
                        (catch_redistribution_update, catch_inside_update))
                
        self.fisheries_dataframe_after_distribution = self.fisheries_dataframe.copy()
        self.fisheries_dataframe_after_distribution.update(updated_efforts)
        self.fisheries_dataframe_after_distribution.update(updated_catchs)
    
    def groupSubfisheries(self, effort_index="sub_fishery_effort",
                          catch_index="sub_fishery_catch") :
        """
        
        Regroup fisheries when they have been rescaled. Must execute
        rescaleFisheries() first.

        Parameters
        ----------
        effort_index : string, optional
            The name of index containing effort of each sub-fishery.
            The default is "sub_fishery_effort".

        Returns
        -------
        None.

        """
        
        if self.fisheries_dataframe_after_distribution is None :
            raise ValueError("Must execute rescaleFisheries() function before groupSubfisheries().")
        
        # TODO : verifier la modif
        self.grouped_fisheries_dataframe_after_distribution = self.fisheries_dataframe_after_distribution.groupby(
            by=['date','fishery','latitude','longitude','gear','resolution']).agg(
            effort=pd.NamedAgg(column=effort_index, aggfunc="sum"),
            catch=pd.NamedAgg(column=catch_index, aggfunc="sum"))
        
        self.grouped_fisheries_dataframe_after_distribution = self.grouped_fisheries_dataframe_after_distribution.reset_index([0,1,2,3,4,5])
                
    def saveInFile(self, filepath="./out.txt", remove_empty_statements=False, verbose=False, to_csv=False) :
        """
        
        Save all statements resulting to the effort redistribution in a CSV file.
        This function is mostly the last one of the typical execution.

        Parameters
        ----------
        filepath : string, optional
            Specify the location where to save the text file.
            The default is "./out.txt".
        to_csv : boolean, optinal
            Specify if you want to save the data in a CSV or text format.
            True to save in CSV, False to save in text.
            The default is False.
        remove_empty_statements : boolean, optional
            Specify if you want to remove empty statements (i.e. with catch=0
            and effort=0)
        verbose : boolean, optional
            Print the number of empty statements that have been removed.
            The default is False.

        Returns
        -------
        None.

        """
        if self.fisheries_dataframe_after_distribution is None :
            raise ValueError("The whole process must be run before calling this function : Load data - Load mask - Rescale Fisheries - Classify fisheries - Distribute effort")
        
        if self.grouped_fisheries_dataframe_after_distribution is None :
            df_to_save = self.fisheries_dataframe_after_distribution.copy()
        else :
            df_to_save = self.grouped_fisheries_dataframe_after_distribution.copy()
        
        
        size_before = df_to_save.index.size
        if remove_empty_statements :
            df_to_save = df_to_save.loc[(df_to_save['effort']!=0) | (df_to_save['catch']!=0)]
        size_after = df_to_save.index.size
        if verbose :
            print("%d lines have been removed."%(size_before-size_after))
        
        columns = ['fishery', 'year', 'month', 'day','gear', 'latitude', 'longitude',
                   'resolution', 'effort', 'catch']
        
        # Transforme date
        df_to_save['year'] = df_to_save.apply(lambda date : date['date'].year, axis=1)
        df_to_save['month'] = df_to_save.apply(lambda date : date['date'].month, axis=1)
        df_to_save['day'] = df_to_save.apply(lambda date : date['date'].day, axis=1)
        df_to_save = df_to_save[columns]
        df_to_save = df_to_save.sort_values(['fishery','year','month','latitude','longitude'])
        
        
        # Headers
        header_lines_each_fishery = df_to_save.groupby(['fishery']).count()['catch'].to_numpy().flatten()
        header_number_fisheries = header_lines_each_fishery.size
        header_columns = ['f','yr','mm','dd','gr','lat','lon','res','E','C']
        
        header = str(header_number_fisheries) + '\n'
        
        for i in header_lines_each_fishery :
            header += str(i)
            header += '\t' if i != header_lines_each_fishery[-1] else '\n'
        
        # Save
        if to_csv :
            df_to_save.to_csv(path_or_buf=filepath, sep=',', header=header_columns, index=False)
        else :
            buffer = df_to_save.to_string(columns=columns, index=False,
                                          header=header_columns, justify='left')
            f = open(filepath, 'w')
            f.write(header+buffer)
            f.close()

###############################################################################
# ---------------------- Informations & Plotting ---------------------------- #
###############################################################################

    def dataInformations(self, classification_index='classification',
                         effort_index='sub_fishery_effort',
                         catch_index='sub_fishery_catch') :
        """
        
        Print some informations about the data we are working on. Especially
        the columns name and the maximum and minimum of longitude and latitude
        which are particularly usefull to create a mask from rectangular figures.

        Parameters
        ----------
        classification_index : string, optional
            Name of classification index in the dataframe.
            The default is 'classification'.
        effort_index : string, optional
            Name of effort index in the dataframe after redistribution.
            The default is 'sub_fishery_effort'.
        catch_index : string, optional
            Name of catch index in the dataframe after redistribution.
            The default is 'sub_fishery_catch'.

        Returns
        -------
        None.

        """
        
        print("\n# ------------------------------------------------------------------------------- #\n")
        print("Columns are  :\n", list(self.fisheries_dataframe.columns.array),end="\n\n")
        
        print(self.fisheries_dataframe)
        
        lat = np.unique(self.fisheries_dataframe['latitude'])
        print("\nLatitude goes from %.2f to %.2f" % (lat[0], lat[-1]))
        lon = np.unique(self.fisheries_dataframe['longitude'])
        print("\nLongitude goes from %.2f to %.2f" % (lon[0], lon[-1]))
        
        min_r = np.min(np.unique(self.fisheries_dataframe['resolution']))
        max_r = np.max(np.unique(self.fisheries_dataframe['resolution']))
        print("\nMinimum resolution is %.2f, Maximum is %.2f" % (min_r,max_r))
        
        if 'classification' in self.fisheries_dataframe.columns.array :
            nb_distribute = np.sum(self.fisheries_dataframe[classification_index] == 1.0)
            nb_receive = np.sum(self.fisheries_dataframe[classification_index] == 2.0)
            print("\nDistribution : %d fisheries will distribute, %d fisheries will receive" % (nb_distribute, nb_receive))
        
        size = np.sum(self.fisheries_dataframe.memory_usage()) / (1000 * 1000)
        print("\nThe actual Dataframe is taking %.2f mo of memory usage" % size)
        
        if self.fisheries_dataframe_after_distribution is not None :
            effort = np.sum(self.fisheries_dataframe.loc[
                self.fisheries_dataframe[classification_index] == 1.0][effort_index])
            total_effort = np.sum(self.fisheries_dataframe[effort_index])
            catch = np.sum(self.fisheries_dataframe.loc[
                self.fisheries_dataframe[classification_index] == 1.0][catch_index])
            total_catch = np.sum(self.fisheries_dataframe[catch_index])
            print("\nRedistribution :\n\t%.2f effort has been redistributed and total is %.2f" % (effort, total_effort))
            print("\tThis is equivalent to %.2f %s" % ((effort / total_effort) * 100, '%'))
            print("\n\t%.2f catchs has been redistributed and total is %.2f" % (catch, total_catch))
            print("\tThis is equivalent to %.2f %s" % ((catch / total_catch) * 100, '%'))
        print("\n# ------------------------------------------------------------------------------- #\n")
    
    def checkSummary(self, before_effort_index='sub_fishery_effort', after_effort_index='sub_fishery_effort',
                     before_catch_index='sub_fishery_catch', after_catch_index='sub_fishery_catch',
                     classification_index='classification', inside_zones=1.0, verbose=True) :
        """
        
        Print information about each fishery effort and catch :
            - Sum of effort/catch for all statements before redistribution
            - Sum of effort/catch for statements inside marine protected area before redistribution
            - Sum of effort/catch for all statements after redistribution
            - Sum of effort/catch for statements inside marine protected area after redistribution

        Parameters
        ----------
        before_effort_index : string, optional
            The name of index containing effort of each statement.
            The default is 'subfishery_effort'.
        after_effort_index : string, optional
           The name of index containing effort of each statement.
           The default is 'subfishery_effort'.
        before_catch_index : string, optional
            The name of index containing catch of each statement.
            The default is 'sub_fishery_catch'.
        after_catch_index : string, optional
           The name of index containing catch of each statement.
           The default is 'sub_fishery_catch'.
        classification_index : string, optional
            The name of index containing classificaiton of each statement.
            The default is 'classification'.
        inside_zones : float, optional
            The value in the mask which significate that a location is inside
            the marine protected area.
            The default is 1.0.
        verbose : boolean, optional
            If True, informations are printed in the console.
            The default is True.

        Returns
        -------
        result : tuple of Pandas.Dataframe
            (result_effort, result_catch) where the index is the fisheries number.
            result_effort columns are :
                - Sum of effort for all statements before redistribution
                - Sum of effort for statements inside marine protected area before redistribution
                - Sum of effort for all statements after redistribution
                - Sum of effort for statements inside marine protected area after redistribution
            result_columns columns are :
                - Sum of catch for all statements before redistribution
                - Sum of catch for statements inside marine protected area before redistribution
                - Sum of catch for all statements after redistribution
                - Sum of catch for statements inside marine protected area after redistribution
        """
        if self.fisheries_dataframe is None :
            raise ValueError("The whole process must be run before calling this function : Load data - Load mask - Classify fisheries - Distribute effort")
        elif self.fisheries_dataframe_after_distribution is None :
            raise ValueError("The whole process must be run before calling this function : Load mask - Classify fisheries - Distribute effort")
        
        # ------------------------------------------------------------------- #
        # EFFORT ------------------------------------------------------------ #
        
        # BEFORE ------------------------------------------------------------ #
        sum_fisheries = self.fisheries_dataframe.groupby(
            by=['fishery']).agg(Effort=pd.NamedAgg(column=before_effort_index, aggfunc='sum'))
        before_sum_fisheries_inside = self.fisheries_dataframe.loc[
            self.fisheries_dataframe[classification_index] == inside_zones]
        before_sum_fisheries_inside = before_sum_fisheries_inside.groupby(
            by=['fishery']).agg(effort=pd.NamedAgg(column=before_effort_index, aggfunc='sum'))
        sum_fisheries['E_inside'] = before_sum_fisheries_inside
        
        # AFTER ------------------------------------------------------------- #
        after_sum_fisheries = self.fisheries_dataframe_after_distribution.groupby(
            by=['fishery']).agg(effort=pd.NamedAgg(column=after_effort_index, aggfunc='sum'))
        
        after_sum_fisheries_inside = self.fisheries_dataframe_after_distribution.loc[
            self.fisheries_dataframe_after_distribution[classification_index] == inside_zones]
        after_sum_fisheries_inside = after_sum_fisheries_inside.groupby(
            by=['fishery']).agg(effort=pd.NamedAgg(column=after_effort_index, aggfunc='sum'))
        sum_fisheries['E_After_Distribution'] = after_sum_fisheries
        sum_fisheries['E_A_D_Inside'] = after_sum_fisheries_inside
        
        # TOTAL ------------------------------------------------------------- #
        total_Effort = sum_fisheries['Effort'].sum()
        total_E_inside = sum_fisheries['E_inside'].sum()
        total_E_After_Distribution = sum_fisheries['E_After_Distribution'].sum()
        total_E_A_D_Inside = sum_fisheries['E_A_D_Inside'].sum()
        
        # FINAL PRINTING ---------------------------------------------------- #
        index = [["Before","After"], ["All","Inside MPA"]]
        index = pd.MultiIndex.from_product(index)
        result_effort = pd.DataFrame(sum_fisheries.to_numpy(), columns=index,
                              index=sum_fisheries.index)
        total = pd.DataFrame([[total_Effort,total_E_inside,total_E_After_Distribution,total_E_A_D_Inside]],
                               columns=index, index=['Total'])
        
        result_effort = pd.concat([result_effort, total]).fillna(0.0)
        
        if verbose :
            print('\n# -- EFFORT -- #')
            print("The percentage of effort distribution is :\t%.2f" % (self.ratio_distribution_effort * 100))
            print("The percentage of effort conservation is :\t%.2f" % (self.ratio_conservation_effort * 100), end="\n\n")
            print(result_effort.round(2))
        
        # ------------------------------------------------------------------- #
        # CATCH ------------------------------------------------------------- #
        
         # BEFORE ------------------------------------------------------------ #
        sum_fisheries = self.fisheries_dataframe.groupby(
            by=['fishery']).agg(Catch=pd.NamedAgg(column=before_catch_index, aggfunc='sum'))
        before_sum_fisheries_inside = self.fisheries_dataframe.loc[
            self.fisheries_dataframe[classification_index] == inside_zones]
        before_sum_fisheries_inside = before_sum_fisheries_inside.groupby(
            by=['fishery']).agg(catch=pd.NamedAgg(column=before_catch_index, aggfunc='sum'))
        sum_fisheries['C_inside'] = before_sum_fisheries_inside
        
        # AFTER ------------------------------------------------------------- #
        after_sum_fisheries = self.fisheries_dataframe_after_distribution.groupby(
            by=['fishery']).agg(catch=pd.NamedAgg(column=after_catch_index, aggfunc='sum'))
        
        after_sum_fisheries_inside = self.fisheries_dataframe_after_distribution.loc[
            self.fisheries_dataframe_after_distribution[classification_index] == inside_zones]
        after_sum_fisheries_inside = after_sum_fisheries_inside.groupby(
            by=['fishery']).agg(catch=pd.NamedAgg(column=after_catch_index, aggfunc='sum'))
        sum_fisheries['C_After_Distribution'] = after_sum_fisheries
        sum_fisheries['C_A_D_Inside'] = after_sum_fisheries_inside
        
        # TOTAL ------------------------------------------------------------- #
        total_Catch = sum_fisheries['Catch'].sum()
        total_C_inside = sum_fisheries['C_inside'].sum()
        total_C_After_Distribution = sum_fisheries['C_After_Distribution'].sum()
        total_C_A_D_Inside = sum_fisheries['C_A_D_Inside'].sum()
        
        # FINAL PRINTING ---------------------------------------------------- #
        index = [["Before","After"], ["All","Inside MPA"]]
        index = pd.MultiIndex.from_product(index)
        result_catch = pd.DataFrame(sum_fisheries.to_numpy(), columns=index,
                                    index=sum_fisheries.index)
        total = pd.DataFrame([[total_Catch,total_C_inside,total_C_After_Distribution,total_C_A_D_Inside]],
                               columns=index, index=['Total'])
        
        result_catch = pd.concat([result_catch, total]).fillna(0.0)
        
        if verbose :
            print('\n# -- CATCH -- #')
            print("The percentage of catch distribution is :\t%.2f" % (self.ratio_distribution_catch * 100))
            print("The percentage of catch conservation is :\t%.2f" % (self.ratio_conservation_catch * 100), end="\n\n")
            print(result_catch.round(2))
        
        return result_effort, result_catch
        
    def plotMask(self, plot_positions={'lat':(-80,80), 'lon':(0,360)}, external=False) :
        """
        
        Plot the Mask used to redistribut effort. This function use the pyGMT
        library to plot and the xarray library to store the mask.

        Parameters
        ----------
        plot_positions : dict, optional
            Contains the latitude and longitude minimum and maximum (as tuples)
            of the mask. This limits the mask to a specific area.
            The default is {'lat':(-80,80), 'lon':(0,360)}.
        external : boolean, optional
            True if you want to display the result in an external window. False
            otherwise.
            The default is False.

        Returns
        -------
        fig : pygmt.figure.Figure
            Usable if the IPython console doesn't allow pygmt plotting.
            Just call fig.show() function.

        """
        
        fig = pygmt.Figure()
        min_lat, max_lat = plot_positions['lat']
        min_lon, max_lon = plot_positions['lon']
        
        # Cannot include south/north poles with Mercator projection
        if min_lat < -89 :
            min_lat = -89
        if max_lat > 89 :
            max_lat = 89
        if max_lon > 360 :
            max_lon = 360
        if min_lon < 0 :
            min_lon = 0
            
        fig.basemap(region=[min_lon, max_lon, min_lat, max_lat], projection="M15c", frame="a")
        fig.coast(water="skyblue")
        fig.grdimage(grid=self.mask,cmap="viridis", interpolation="n")
        fig.coast(land="#666666",borders="1/1p")
        
        if external :
            fig.show(method='external')
        else :
            fig.show()
        
        return fig

    def plotFisheries(self, selected_fisheries=None, after_distribution=False,
                      date_start=None, date_end=None,
                      plot_positions={'lat':(-80,80), 'lon':(0,360)},
                      sort_ascending=False, external=False, transparency=50,
                      size_coef=1, minimal_size=0, plot_catch=False, color_map=False,
                      **kwargs) :
        """
        
        Plot effort or catch distribution among the fisheries on a particular area.

        Parameters
        ----------
        selected_fisheries : list of int, optional
            A list containing all fisheries selected for plotting.
            The default is None.
        after_distribution : boolean, optional
            Choose if you want to plot the effort/catch before or after the
            distribution. True is after False is before.
            The default is False.
        date_start : str or datetime64, optional
            Choose if you want to plot the effort/catch starting on a date or 
            during all the time serie. If date_end is not provide, it will only
            plot effort/catch on this date. If the starting date is not in the
            serie, the first date will be choosed.
            The default is None.
        date_end : str or datetime64, optional
            Choose if you want to plot the effort/catch until a specific date
            or until the end (None value). If the ending date is not in the
            serie, the last date will be choosed.
            The default is None.
        plot_positions : dict of tuples, optional
            Contains latitude and longitude limits as (min_limit, max_limit).
            The default is {'lat':(-80,80), 'lon':(0,360)}.
        sort_ascending : boolean, optional
            Descending sorting lets appear the smaller circles over the bigger.
            True is ascending, False is descending.
            The default is False.
        external : boolean, optional
            How the figure will be displayed.
            False : PNG preview. True : PDF preview in an external program.
            The default is False.
        transparency : int, optional
            Set the circles transparency. From 0 to 100 % transparent.
            The default is 50.
        size_coef : float, optional
            Circles size coefficient. Allow user to change the size of each 
            circle by multiplying the effort by this value.
            The default is 1.
        minimal_size : float, optional
            Minimal size for circle. Allow user to let smaller circle appear.
            The default is 0.
        plot_catch : boolean, optional
            Choose to plot catch or effort. True is Catch, False is Effort.
            The default is False.
        color_map : boolean, optional
            Activate the color bar option. Circles are colored according to their
            effort/catch value. A color bar representing the value scale is
            plotted just bellow. True to plot, False to not.
            The default is False.
        **kwargs : dict
            Additional arguments can be passed :
                - dpi : The resolution of the final plot. Default is 600.
                - width : The size of the final plot. Default is 1000.
                - color : Set the circles color. Default is 'Red'.

        Returns
        -------
        IPython.display.Image
            Only if 'external' == True

        """
        
        min_lat, max_lat = plot_positions['lat']
        min_lon, max_lon = plot_positions['lon']
        
        # Cannot include south/north poles with Mercator projection
        if min_lat < -89 :
            min_lat = -89
        if max_lat > 89 :
            max_lat = 89
        if max_lon > 360 :
            max_lon = 360
        if min_lon < 0 :
            min_lon = 0
        
        # Dataframe selection
        if not after_distribution :
            fisheries_effort = self.original_fisheries_dataframe
        else :
            fisheries_effort = self.grouped_fisheries_dataframe_after_distribution
        
        # Fisheries selection
        if selected_fisheries is not None :
            selected_fisheries = np.array(selected_fisheries)
            condition = fisheries_effort['fishery'] == selected_fisheries[0]
            for fishery in selected_fisheries[1:] :
                condition |= (fisheries_effort['fishery'] == fishery)
            fisheries_effort = fisheries_effort.loc[condition]
        
        # Date selection
        if date_start is not None :
            
            # If there is only the starting date 
            condition = (fisheries_effort['date'] == date_start)
            
            # Else, all other dates are added until it reachs the date_end
            if (date_end is not None) and (date_end != date_start):
                dates = np.unique(fisheries_effort['date'].sort_values())
                
                start = np.where(dates == np.datetime64(date_start))
                # if date is out of bounds
                if start[0].size == 0 :
                    print("Warning : Starting date does not exist, taking first time step.")
                    start = 0
                else :
                    start = start[0][0]
                    
                end = np.where(dates == np.datetime64(date_end))
                # if date is out of bounds
                if end[0].size == 0 :
                    print("Warning : Ending date does not exist, taking last time step.")
                    end = dates.size - 1
                else :
                    end = end[0][0]
                    
                for time_step in dates[start+1:end] :
                    condition |= (fisheries_effort['date'] == time_step)
            
            fisheries_effort = fisheries_effort.loc[condition]
        
        # Grouping then sorting
        index = ['latitude','longitude','effort']
        sort_by = 'effort'
        if plot_catch :
            index[2] = 'catch'
            sort_by= 'catch'
            
        fisheries_effort = fisheries_effort[index].groupby(by=['latitude','longitude']).sum()
        fisheries_effort = fisheries_effort.reset_index([0,1])
        fisheries_effort = fisheries_effort.sort_values(by=[sort_by],
                                                        ascending=sort_ascending)
        
        # GMT Plotting
        size_before = (fisheries_effort.effort if not plot_catch 
                       else fisheries_effort.catch)
        sizes = 0.00005 * size_before * size_coef + minimal_size
        title = "Catch" if plot_catch else "Effort"
        region = [min_lon,max_lon,min_lat,max_lat]
        fig = pygmt.Figure()
        fig.coast(region=region, water="skyblue", land="#666666",
                  projection="M20c", frame=["afg","+t"+title])
        color = kwargs['color'] if 'color' in kwargs else 'red'
        if color_map :
            min_map = np.min(size_before)
            max_map = np.max(size_before)
            pygmt.makecpt(cmap="viridis", series=[min_map, max_map])
            color = size_before
        fig.plot(x=fisheries_effort.longitude, y=fisheries_effort.latitude,
                 color=color,
                 pen="0.1p,black",
                 size=sizes,
                 cmap=color_map,
                 style="cc",
                 transparency=transparency)
        
        if color_map :
            fig.colorbar(frame='af+l'+title)
        
        # Output generation
        dpi = kwargs['dpi'] if 'dpi' in kwargs else 600
        width = kwargs['width'] if 'width' in kwargs else 1000
        if external :
            fig.show(dpi=dpi, width=width, method='external')
        else :
            return fig.show(dpi=dpi, width=width)
    
    def plotReceived(self, selected_fisheries=None, date_start=None, date_end=None,
                      plot_positions={'lat':(-80,80), 'lon':(0,360)},
                      sort_ascending=False, external=False, transparency=50,
                      size_coef=1, minimal_size=0, plot_catch=False, from_file=None,
                      color_map=False, **kwargs) :
        """
        
        Plot effort or catch received by fisheries on each coordinate.

        Parameters
        ----------
        selected_fisheries : list of int, optional
            A list containing all fisheries selected for plotting.
            The default is None.
        date_start : str or datetime64, optional
            Choose if you want to plot the effort/catch starting on a date or 
            during all the time serie. If date_end is not provide, it will only
            plot effort/catch on this date. If the starting date is not in the
            serie, the first date will be choosed.
            The default is None.
        date_end : str or datetime64, optional
            Choose if you want to plot the effort/catch until a specific date
            or until the end (None value). If the ending date is not in the
            serie, the last date will be choosed.
            The default is None.
        plot_positions : dict of tuples, optional
            Contains latitude and longitude limits as (min_limit, max_limit).
            The default is {'lat':(-80,80), 'lon':(0,360)}.
        sort_ascending : boolean, optional
            Descending sorting lets appear the smaller circles over the bigger.
            True is ascending, False is descending.
            The default is False.
        external : boolean, optional
            How the figure will be displayed.
            False : PNG preview. True : PDF preview in an external program.
            The default is False.
        transparency : int, optional
            Set the circles transparency. From 0 to 100 % transparent.
            The default is 50.
        size_coef : float, optional
            Circles size coefficient. Allow user to change the size of each 
            circle by multiplying the effort by this value.
            The default is 1.
        minimal_size : float, optional
            Minimal size for circle. Allow user to let smaller circle appear.
            The default is 0.
        plot_catch : boolean, optional
            Choose to plot catch or effort. True is Catch, False is Effort.
            The default is False.
        from_file : string, optional
            The filepath to the file you want to compare to. Corresponding to the
            output file computed with this effort and catch distribution module.
            Original file must be the one you read with this class instance.
            Default is None.
        color_map : boolean, optional
            Activate the color bar option. Circles are colored according to their
            effort/catch value. A color bar representing the value scale is
            plotted just bellow. True to plot, False to not.
            The default is False.
        **kwargs : dict
            Additional arguments can be passed :
                - dpi : The resolution of the final plot. Default is 600.
                - width : The size of the final plot. Default is 1000.
                - color : Set the circles color. Default is 'Red'.
            Those arguments are needed if you use the from_file argument :
                - lines_to_skip : int, number of lines to skip before read the
                                  header.
                - verbose : boolean, if you want to see informations contained 
                            in header.

        Returns
        -------
        IPython.display.Image
            Only if 'external' == True

        """
        
        min_lat, max_lat = plot_positions['lat']
        min_lon, max_lon = plot_positions['lon']
        
        # Cannot include south/north poles with Mercator projection
        if min_lat < -89 :
            min_lat = -89
        if max_lat > 89 :
            max_lat = 89
        if max_lon > 360 :
            max_lon = 360
        if min_lon < 0 :
            min_lon = 0
        
        # Subtraction
        index = 'effort'
        index_update = 'received_effort'
        if plot_catch :
            index = 'catch'
            index_update = 'received_catch'
        
        # In and Out files may be different in size.
        # Reason is that out file is grouped by [date, fishery, latitude, longitude, gear, resolution]
        # Some lines are sometimes divided in original file. That provoc a
        # reduction of the out file size.
        before = self.original_fisheries_dataframe
        
        # TODO : verifier la modif
        before = before.groupby(['date','fishery','latitude','longitude','gear','resolution']).sum()
        before = before.reset_index([0,1,2,3,4,5])
        before = before[['date', 'fishery', 'latitude','longitude', index]
                        ].copy().sort_values(
                            by=['date', 'fishery', 'latitude', 'longitude'],
                            ignore_index=True)
        
        after = None
        if from_file is not None :
            if (not 'lines_to_skip' in kwargs) :
                raise AttributeError("If you are using 'from_file' argument, you must specify 'lines_to_skip' argument too.")
            if (not 'verbose' in kwargs) :
                raise AttributeError("If you are using 'from_file' argument, you must specify 'verbose' argument too.")
            
            tmp_er = self.__readTextFile__(from_file, kwargs['lines_to_skip'], kwargs['verbose'])
            after = tmp_er[
            ['date', 'fishery', 'latitude','longitude', index]].copy().sort_values(
                by=['date', 'fishery', 'latitude', 'longitude'], ignore_index=True)
            
        else :
            after = self.grouped_fisheries_dataframe_after_distribution[
            ['date', 'fishery', 'latitude','longitude', index]].copy().sort_values(
                by=['date', 'fishery', 'latitude', 'longitude'], ignore_index=True)
        
        assert before.index.size == after.index.size, "Warning: dataframes have not the same size. You may have removed the rows that have neither effort nor capture"
        
        fisheries_effort = before[['date', 'fishery', 'latitude', 'longitude']]
        fisheries_effort[index_update] = (after[index] - before[index]).apply(lambda x : x if x > 0 else 0)
        
        # Fisheries selection
        if selected_fisheries is not None :
            selected_fisheries = np.array(selected_fisheries)
            condition = fisheries_effort['fishery'] == selected_fisheries[0]
            for fishery in selected_fisheries[1:] :
                condition |= (fisheries_effort['fishery'] == fishery)
            fisheries_effort = fisheries_effort.loc[condition]
        
        # Date selection
        if date_start is not None :
            
            # If there is only the starting date 
            condition = (fisheries_effort['date'] == date_start)
            
            # Else, all other dates are added until it reachs the date_end
            if (date_end is not None) and (date_end != date_start):
                dates = np.unique(fisheries_effort['date'].sort_values())
                
                start = np.where(dates == np.datetime64(date_start))
                # if date is out of bounds
                if start[0].size == 0 :
                    print("Warning : Starting date does not exist, taking first time step.")
                    start = 0
                else :
                    start = start[0][0]
                    
                end = np.where(dates == np.datetime64(date_end))
                # if date is out of bounds
                if end[0].size == 0 :
                    print("Warning : Ending date does not exist, taking last time step.")
                    end = dates.size - 1
                else :
                    end = end[0][0]
                    
                for time_step in dates[start+1:end] :
                    condition |= (fisheries_effort['date'] == time_step)
            
            fisheries_effort = fisheries_effort.loc[condition]
        
        # Grouping then sorting
        fisheries_effort = fisheries_effort[['latitude','longitude',index_update]
                                            ].groupby(by=['latitude','longitude']).sum()
        fisheries_effort = fisheries_effort.reset_index([0,1])
        fisheries_effort = fisheries_effort.sort_values(by=[index_update],
                                                        ascending=sort_ascending)
        
        # GMT Plotting
        size_before = (fisheries_effort.received_effort if not plot_catch 
                       else fisheries_effort.received_catch)
        sizes = 0.00005 * size_before * size_coef + minimal_size
        title = "Effort" if index=="effort" else "Catch"
        region = [min_lon,max_lon,min_lat,max_lat]
        fig = pygmt.Figure()
        fig.coast(region=region, water="skyblue", land="#666666",
                  projection="M20c", frame=["afg","+t"+title])
        color = kwargs['color'] if 'color' in kwargs else 'red'
        if color_map :
            min_map = np.min(size_before)
            max_map = np.max(size_before)
            pygmt.makecpt(cmap="viridis", series=[min_map, max_map])
            color = size_before
        fig.plot(x=fisheries_effort.longitude, y=fisheries_effort.latitude,
                 color=color,
                 pen="0.1p,black",
                 size=sizes,
                 cmap=color_map,
                 style="cc",
                 transparency=transparency)
        
        if color_map :
            fig.colorbar(frame='af+l'+title)
        
        # Output generation
        dpi = kwargs['dpi'] if 'dpi' in kwargs else 600
        width = kwargs['width'] if 'width' in kwargs else 1000
        if external :
            fig.show(dpi=dpi, width=width, method='external')
        else :
            return fig.show(dpi=dpi, width=width)
        
###############################################################################
# -------------------------------- Tools ------------------------------------ #
###############################################################################

    def slicePercent(self, percent, begin_percent=0) :
        """
        
        This function slice the Dataframe containing the data according to the
        percent parameter. Usable to test some functions on a reduced set of data.

        Parameters
        ----------
        percent : float
            The amount of data to select : 0 < percent <= 100.
        begin_percent : float, optional
            The amount of data to skip before slicing :
            0 <= begin_percent <= (100 - percent)
            The default is 0.

        Returns
        -------
        None.

        Warning
        -------
        If the data is sliced with this function the action is irreversible.
        You will need to read again the source text file to reload the data.
        Slicing a sliced dataframe will return a selection in the sliced data.
        slicePercent(50).slicePercent(50) is equivalent to : slicePercent(25)
        
        Examples
        --------
        >>> slicePercent(percent=25, begin_percent=50)
        # Select the third quarter of the data.

        """
        assert 0 < percent and percent <= 100, '0 < percent <= 100'
        assert 0 <= begin_percent and begin_percent <= begin_percent <= (100 - percent), '0 <= begin_percent <= 100'
        
        slice_begin = int(self.fisheries_dataframe.index.size * (begin_percent / 100))
        size = int(self.fisheries_dataframe.index.size * (percent / 100))
        slice_end = slice_begin + size
        
        assert slice_end < (self.fisheries_dataframe.index.size), 'Overflow : begin %f, size %f, end %f : But max is  %f' % (slice_begin, size, slice_end,self.fisheries_dataframe.index.size)
        
        self.fisheries_dataframe = self.fisheries_dataframe.iloc[slice_begin:slice_end]
        self.original_fisheries_dataframe = self.original_fisheries_dataframe.iloc[slice_begin:slice_end]

    def saveMaskToTextFile(self, replace_outside_zone=0, replace_inside_zone=1,
                           replace_distributed_zone=2, filepath='./mask.txt') :
        """
        
        Save the mask in a text file.

        Parameters
        ----------
        replace_outside_zone : int or float, optional
            Replace value 0 in mask. Corresponding to the positions outside the
            marine protected area.
            The default is 0.
        replace_inside_zone : int or float, optional
            Replace value 1 in mask. Corresponding to the positions inside the
            marine protected area.
            The default is 1.
        replace_distributed_zone : int or float, optional
            Replace value 2 in mask. Corresponding to the position where the
            effort while be distributed.
            The default is 2.
        filepath : string, optional
            The text file where to save the mask to. 
            The default is './mask.txt'.

        Returns
        -------
        None.

        """
        
        np_mask = self.mask.data.astype(int)
        
        np_mask = np.where(self.mask.data.astype(int)==0, replace_outside_zone, np_mask)
        np_mask = np.where(self.mask.data.astype(int)==1, replace_inside_zone, np_mask)
        np_mask = np.where(self.mask.data.astype(int)==2, replace_distributed_zone, np_mask)
        
        if (isinstance(replace_outside_zone, int) and
            isinstance(replace_inside_zone, int) and
            isinstance(replace_distributed_zone, int)) :
            fmt = '%d'
        else :
            fmt = '%.2f'

        np.savetxt(fname=filepath, X=np_mask, fmt=fmt)
