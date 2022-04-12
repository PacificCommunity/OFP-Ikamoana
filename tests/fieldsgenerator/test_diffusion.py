from collections import namedtuple
import unittest

import numpy as np
import parcels
import xarray as xr
from ikamoana.ikamoanafields.core.fieldsgenerator import (
    diffusion
)

from .fieldsgenerator_data import (
    diffusion_res_meters,
    diffusion_res_nauticmiles,
    dataarray_3d_diffusion_habitat,
    dataarray_3d_diffusion_landmask
)

class Test_Diffusion(unittest.TestCase):
    
    def findLengthByCohort(self,cohort) :
            length_cm_list = [5, 10, 20, 40, 80]
            return length_cm_list[cohort.data]
    
    FHStructure = namedtuple('FHStructure', ['findLengthByCohort'])
    IKAStructure = namedtuple(
        'IKAStructure',
        ["timestep", "units", "sigma_K","c_scale", "c", "P",
         "diffusion_scale", "diffusion_boost"])

    ## NOTE : These functions have changed, they need currents    
    # def test_diffusion_0_to_1(self):
    #     ika_s = self.IKAStructure(2592000,"m_per_s",4.8,1,1,0.93,3,1,0)
    #     fh_s = self.FHStructure(self.findLengthByCohort)
    #     diff_meter = diffusion(ika_s, fh_s, dataarray_3d_diffusion_habitat,
    #                            dataarray_3d_diffusion_landmask)
    #     self.assertTrue(False not in (
    #         diffusion_res_meters == diff_meter.data.astype(int)))
    
    # def test_diffusion_0_to_1(self):
    #     ika_s = self.IKAStructure(2592000,"nm_per_timestep",4.8,1,1,0.93,3,1,0)
    #     fh_s = self.FHStructure(self.findLengthByCohort)
    #     diff_nm = diffusion(ika_s, fh_s, dataarray_3d_diffusion_habitat,
    #                         dataarray_3d_diffusion_landmask)
    #     self.assertTrue(False not in (
    #         diffusion_res_nauticmiles == diff_nm.data.astype(int)))