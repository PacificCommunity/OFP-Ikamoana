import unittest
from collections import namedtuple

from ikamoana.ikamoanafields.core.fieldsgenerator import landmask

from .fieldsgenerator_data import (
    array_3d_landmask_forage_lmeso,
    dataarray_3d_landmask_habitat,
    res_shallow_habitat, res_no_shallow_habitat,
    landmask_mask_L1, landmask_mask_L3,
    landmask_coords, 
    res_shallow_seapodym, res_no_shallow_seapodym)


class TestLandmaskMethod(unittest.TestCase) :
    
    def test_seapodym_global_mask_no_shallow(self):
        FHStructure = namedtuple('FHStructure', ['global_mask','coords'])
        global_mask = {'mask_L1':landmask_mask_L1,
                       'mask_L3':landmask_mask_L3}
        data_structure = FHStructure(global_mask, landmask_coords)
        res_landmask = landmask(
            data_structure, use_SEAPODYM_global_mask=True,
            shallow_sea_to_ocean=False, field_output=False)
        self.assertTrue(not False in (res_landmask.data == res_no_shallow_seapodym))
    
    
    def test_seapodym_global_mask_shallow(self):
        FHStructure = namedtuple('FHStructure', ['global_mask','coords'])
        global_mask = {'mask_L1':landmask_mask_L1,
                       'mask_L3':landmask_mask_L3}
        data_structure = FHStructure(global_mask, landmask_coords)
        res_landmask = landmask(
            data_structure, use_SEAPODYM_global_mask=True,
            shallow_sea_to_ocean=True, field_output=False)
        self.assertTrue(not False in (res_landmask.data == res_shallow_seapodym))

    
    def test_habitat_no_shallow(self):
        FHStructure = namedtuple('FHStructure', ['variables_dictionary'])
        variables_dictionary = {'forage_lmeso':array_3d_landmask_forage_lmeso}
        data_structure = FHStructure(variables_dictionary)
        res_landmask = landmask(
            data_structure, dataarray_3d_landmask_habitat, field_output=False,
            use_SEAPODYM_global_mask=False, shallow_sea_to_ocean=False)
        self.assertTrue(not False in (res_landmask.data == res_no_shallow_habitat))
    
    
    def test_habitat_shallow(self):
        FHStructure = namedtuple('FHStructure', ['variables_dictionary'])
        variables_dictionary = {'forage_lmeso':array_3d_landmask_forage_lmeso}
        data_structure = FHStructure(variables_dictionary)
        res_landmask = landmask(
            data_structure, dataarray_3d_landmask_habitat, field_output=False,
            use_SEAPODYM_global_mask=False, shallow_sea_to_ocean=True)
        self.assertTrue(not False in (res_landmask.data == res_shallow_habitat))

if __name__ == '__main__' :
    unittest.main()
