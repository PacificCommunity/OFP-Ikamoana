from functools import singledispatchmethod
import xarray as xr
import numpy as np
import parcels

## NOTE : This could be initialized only with DataSet/DataArray ?

class ForcingField :

    @singledispatchmethod
    def __init__(self, field : xr.Dataset) :
        self._field = field
        self.var_names = list(field.data_vars.keys())

    @__init__.register
    def _(self, field : xr.DataArray, attrs : dict = {}) :
        self._field = field.to_dataset()
        self._field.attrs.update(attrs)
        self.var_names = [field.name]

    @__init__.register
    def _(self, field : parcels.FieldSet, attrs : dict = None) :
        self._field = field
        self._attrs = attrs
        self.var_names = [i.name for i in field.get_fields()]
    
    def __getitem__(self, name) :
        if isinstance(self._field, parcels.FieldSet) :
            return self._field.__getattribute__(name)
        elif isinstance(self._field, xr.Dataset) :
            return self._field[name]
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")
    
    def __str__(self) -> str:
        return self.dataset.__str__()
    
    def __repr__(self) -> str:
        return self.dataset.__repr__()
    
    @property
    def attrs(self) :
        if isinstance(self._field, parcels.FieldSet) :
            return self._attrs
        elif isinstance(self._field, xr.Dataset) :
            return self._field.attrs
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")
    
    @property
    def lat(self) :

        if isinstance(self._field, parcels.FieldSet) :
            return self._field.gridset.grids[0].lat
        elif isinstance(self._field, xr.Dataset) :
            return self._field.coords['lat']
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")
    
    @property
    def lon(self) :

        if isinstance(self._field, parcels.FieldSet) :
            return self._field.gridset.grids[0].lon
        elif isinstance(self._field, xr.Dataset) :
            return self._field.coords['lon']
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")

## TODO : Add in description -> Time is the time serie of the FIRST Field
    @property
    def time(self) :
        if isinstance(self._field, parcels.FieldSet) :
            time_origin = self._field.gridset.grids[0].time_origin.time_origin
            time = [time_origin + np.timedelta64(int(time), 's')
                    for time in self._field.gridset.grids[0].time]
            return time
        elif isinstance(self._field, xr.Dataset) :
            return self._field.coords['time']
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")



## TODO : Add in description -> Converting to a FieldSet will replace NaN by 0.
# + Reverse only if needed
    @property
    def fieldset(self) -> parcels.FieldSet :
        
        if isinstance(self._field, parcels.FieldSet) :
            return self._field
        elif isinstance(self._field, xr.Dataset) :
            return parcels.FieldSet.from_xarray_dataset(
                self._field.reindex(lat=list(reversed(self._field.lat))),
                variables=dict([(i,i) for i in self._field.keys()]),
                dimensions=dict([(i,i) for i in self._field.dims.keys()]))
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")

    @property
    def dataset(self) -> xr.Dataset :

        if isinstance(self._field, xr.Dataset) :
            return self._field
        elif isinstance(self._field, parcels.FieldSet) :
            data = dict([(field.name,(('time','lat','lon'),field.data)) for field in self._field.get_fields()])
            return xr.Dataset(
                data_vars=data,
                coords={'time':self.time,'lat':self.lat,'lon':self.lon},
                attrs=self.attrs
            )
        else :
            raise AttributeError("Field attribut is neither a FieldSet nor a DataSet.")