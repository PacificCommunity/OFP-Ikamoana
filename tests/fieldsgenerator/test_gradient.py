import unittest

import numpy as np
import parcels
import xarray as xr
from ikamoana.ikamoanafields.core.fieldsgenerator import (
    getCellEdgeSizes,
    gradient
)

from .fieldsgenerator_data import (
    # dataarray_3d_gradient_0_to_4_habitat,
    dataarray_3d_gradient_empty_habitat,
    dataarray_3d_gradient_landmask_full_ocean,
    dataarray_3d_edge_size_1_degrees,
    dataarray_3d_edge_size_2_degrees
)

__parcels_edge_size_message__ = (
    "Parcels.Field.calc_cell_edge_sizes function has changed. This function "
    "should be corrected.")

class Test_getCellEdgeSizesMethod(unittest.TestCase):
    
    def test_one_degree(self):
        
        parcels_field = parcels.Field.from_xarray(
            dataarray_3d_edge_size_1_degrees, name='unnamed',
            dimensions={'time':'time','lat':'lat','lon':'lon'})
        parcels_field.calc_cell_edge_sizes()
        parcels_size_x = parcels_field.cell_edge_sizes['x']
        parcels_size_y = parcels_field.cell_edge_sizes['y']
        
        size_x, size_y = getCellEdgeSizes(dataarray_3d_edge_size_1_degrees)

        self.assertTrue(not False in (size_x == parcels_size_x),
                        __parcels_edge_size_message__)
        self.assertTrue(not False in (size_y == parcels_size_y),
                        __parcels_edge_size_message__)

    def test_two_degrees(self):
        
        parcels_field = parcels.Field.from_xarray(
            dataarray_3d_edge_size_2_degrees, name='unnamed',
            dimensions={'time':'time','lat':'lat','lon':'lon'})
        parcels_field.calc_cell_edge_sizes()
        parcels_size_x = parcels_field.cell_edge_sizes['x']
        parcels_size_y = parcels_field.cell_edge_sizes['y']
        
        size_x, size_y = getCellEdgeSizes(dataarray_3d_edge_size_2_degrees)
        
        self.assertTrue(not False in (size_x == parcels_size_x),
                        __parcels_edge_size_message__)
        self.assertTrue(not False in (size_y == parcels_size_y),
                        __parcels_edge_size_message__)

class TestGradientMethod(unittest.TestCase) :
    
    def test_empty(self):
        
        res_grad_lat, res_grad_lon = gradient(
            field=dataarray_3d_gradient_empty_habitat,
            landmask=dataarray_3d_gradient_landmask_full_ocean)
        
        self.assertTrue(not False in (
            res_grad_lat.data == dataarray_3d_gradient_empty_habitat.data))
        
        self.assertTrue(not False in (
            res_grad_lon.data == dataarray_3d_gradient_empty_habitat.data))
    
    def test_full_constant(self):
        
        full_one = dataarray_3d_gradient_empty_habitat + 1
        
        res_grad_lat, res_grad_lon = gradient(
            field=full_one,
            landmask=dataarray_3d_gradient_landmask_full_ocean)
        
        self.assertTrue(not False in (
            res_grad_lat.data == np.full_like(full_one, 0)))
        
        self.assertTrue(not False in (
            res_grad_lon.data == np.full_like(full_one, 0)))
    
    # def test_zero_to_four(self):
    #     print(dataarray_3d_gradient_0_to_4_habitat.data)
    #     print(dataarray_3d_gradient_landmask_full_ocean.data)
    #     res_grad_lat, res_grad_lon = gradient(
    #         field=dataarray_3d_gradient_0_to_4_habitat,
    #         landmask=dataarray_3d_gradient_landmask_full_ocean)
    #     print(res_grad_lat.data)
    #     print(res_grad_lon.data)
        
    