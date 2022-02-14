import numpy as np
import xarray as xr

nan = np.NaN

# ----------------------------- LANDMASK ----------------------------- #

# -- HABITAT

array_3d_landmask_forage_lmeso = np.array(
    [[[nan,nan,nan,nan,nan],
      [0,0.,nan,nan,np.inf],
      [nan,nan,nan,nan,nan]]])

array_3d_landmask_habitat = np.array(
    [[[nan,nan,nan,nan,nan],
      [0,0.,0.1,nan,np.inf],
      [nan,nan,nan,nan,nan]]])

dataarray_3d_landmask_habitat = xr.DataArray(
    array_3d_landmask_habitat,
    coords={'time':np.array(['2000-01-15'],
                            dtype='datetime64[D]'),
            'lat':[0,1,2],'lon':[0,1,2,3,4]},
    attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':4})

res_shallow_habitat = np.array(
    [[0., 0., 0., 0., 0.],
     [1., 1., 0., 1., 0.],
     [0., 0., 0., 0., 0.]])

res_no_shallow_habitat = np.array(
    [[0., 0., 0., 0., 0.],
     [1., 1., 2., 1., 0.],
     [0., 0., 0., 0., 0.]])


# -- SEAPODYM GLOBAL MASK
# NOTE : False is land and True is ocean

landmask_mask_L1 = np.array(
    [[[0,0,0,0,0],
      [0,1,1,1,0],
      [0,0,0,0,0]]])

landmask_mask_L3 = np.array(
    [[[0,0,0,0,0],
      [0,0,1,0,0],
      [0,0,0,0,0]]])

landmask_coords = {'lat':[0,1,2], 'lon':[0,1,2,3,4]}

res_shallow_seapodym = np.array(
    [[[0,0,0,0,0],
      [1,0,0,0,1],
      [0,0,0,0,0]]])

res_no_shallow_seapodym = np.array(
    [[[0,0,0,0,0],
      [1,2,0,2,1],
      [0,0,0,0,0]]])