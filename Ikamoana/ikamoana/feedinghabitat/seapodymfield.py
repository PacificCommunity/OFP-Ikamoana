## NOTE : 
# It is not recommanded to inherit from a DataArray. Another method is
# to create a SeapodymField constructor method which will return a
# Xarray DataArray using NetCDF or Dym file.
#
# Actual error is probably because of :
#   type(self)(variable, coords, name=name, fastpath=True, indexes=indexes)
#
# in DataArray._replace() which is calling SeapodymField.__init__() with
# DataArray.__init__() arguments.

# class SeapodymField(xr.DataArray) :

#     __slots__ = []

#     def __init__(self,
#                  filepath: str = None,
#                  data_array : xr.DataArray = None,
#                  dym_varname : str = None,
#                  dym_attributs : str = None):

#         if (filepath is None) and (data_array is None) :
#             raise ValueError(
#                 'filepath == None AND data_array == None.\n'
#                 + 'A SeapodymField must be initialized by NetCDF or Dym file'
#                 + ' using the filepath arg, or with a Xarray.DataArray'
#                 + ' using the data_array arg.')
#         elif (filepath is not None) and (data_array is None) :
            
#             #NetCDF
#             if filepath.lower().endswith(('.nc', '.cdf')) :
#                 super().__init__(xr.open_dataarray(filepath))
            
#             #DymFile
#             if filepath.lower().endswith('.dym') :
#                 if dym_varname is None :
#                     dym_varname = 'Default_Name'
#                 tmp_data_array = df.dym2ToDataArray(infile = filepath,
#                                                     varname = dym_varname,
#                                                     attributs = dym_attributs)
#                 super().__init__(tmp_data_array)

#         elif (filepath is None) and (data_array is not None) :
#             super().__init__(data_array)
            

#         else :
#             raise ValueError(
#                 'filepath != None AND data_array != None.\n'
#                 +'Initialization method must be chosen among File or DataArray.'
#                 +' Leave the other argument empty.')