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
    coords={'time':np.array(['2000-01-15'], dtype='datetime64[D]'),
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
    [[0,0,0,0,0],
     [1,0,0,0,1],
     [0,0,0,0,0]])

res_no_shallow_seapodym = np.array(
    [[0,0,0,0,0],
     [1,2,0,2,1],
     [0,0,0,0,0]])

# ----------------------- _getCellEdgeSizes -------------------------- #

dataarray_3d_edge_size_1_degrees = xr.DataArray(
    np.zeros((1,3,2)),
    coords={'time':np.array(['2000-01-15'], dtype='datetime64[D]'),
            'lat':[-1,0,1],'lon':[0,1]},
    attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':1})

dataarray_3d_edge_size_2_degrees = xr.DataArray(
    np.zeros((1,3,2)),
    coords={'time':np.array(['2000-01-15'], dtype='datetime64[D]'),
            'lat':[-2,0,2],'lon':[0,2]},
    attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':1})

# ----------------------------- GRADIENT ----------------------------- #

# -- FULL ZERO

array_gradient_landmask_full_ocean = np.array(
    [[0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.]])

dataarray_3d_gradient_landmask_full_ocean = xr.DataArray(
    array_gradient_landmask_full_ocean,
    coords={'lat':[0,1,2],'lon':[0,1,2,3,4]},
    attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':4})

array_3d_gradient_empty_habitat = np.array(
    [[[0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0]]])

dataarray_3d_gradient_empty_habitat = xr.DataArray(
    array_3d_gradient_empty_habitat,
    coords={'time':np.array(['2000-01-15'], dtype='datetime64[D]'),
            'lat':[0,1,2],'lon':[0,1,2,3,4]},
    attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':4})

# # -- LEFT 0 TO RIGHT 4

# array_3d_gradient_0_to_4_habitat = np.array(
#     [[[0,1,2,3,4],
#       [0,1,2,3,4],
#       [0,1,2,3,4]]])

# dataarray_3d_gradient_0_to_4_habitat = xr.DataArray(
#     array_3d_gradient_0_to_4_habitat,
#     coords={'time':np.array(['2000-01-15'], dtype='datetime64[D]'),
#             'lat':[0,1,2],'lon':[0,1,2,3,4]},
#     attrs={'lat_min':0,'lat_max':2,'lon_min':0,'lon_max':4})

# ---------------------------- DIFFUSION ----------------------------- #

array_3d_diffusion_habitat = np.array(
    [[[0.0,1.0]],[[0.0,1.0]],[[0.0,1.0]],[[0.0,1.0]],[[0.0,1.0]]]
)
array_3d_diffusion_landmask = np.array(
    [[0,0]]
)

dataarray_3d_diffusion_habitat = xr.DataArray(
    array_3d_diffusion_habitat,
    dims=('time','lat','lon'),
    coords={'time':np.array(['2000-01-15','2000-02-15','2000-03-15', '2000-04-15',
                             '2000-05-15'], dtype='datetime64[D]'),
            'lat':[0], 'lon':[0,1], "cohorts":("time",[0,1,2,3,4])},
    attrs={'lat_min':0,'lat_max':0,'lon_min':0,'lon_max':11,
           'cohort_start':0, 'cohort_end':4})

dataarray_3d_diffusion_landmask = xr.DataArray(
    array_3d_diffusion_landmask,
    coords={'lat':[0],'lon':[0,1]})

diffusion_res_meters = np.array(
    [[[7776,544]],[[31104,2177]],[[124416,8709]],[[497664,34836]],[[1990656,139345]]]
)
diffusion_res_nauticmiles = np.array(
    [[[5876,411]],[[23505,1645]],[[94021,6581]],[[376087,26326]],[[1504351,105304]]]
)