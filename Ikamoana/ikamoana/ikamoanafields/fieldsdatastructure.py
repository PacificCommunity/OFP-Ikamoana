
class FieldsDataStructure :

## NOTE : This is a hardcoded structure. Only for test.

    def __init__(self) -> None :

        # TODO : This is from IKAMOANA config file
        self.diffusion_boost=0
        self.diffusion_scale=1
        self.sig_scale=1
        self.c_scale=1
        self.shallow_sea_to_ocean = False
        self.landmask_from_habitat = True   
        self.taxis_scale=1
        self.units='m_per_s',
        self.timestep=30*24*60*60
        self.start_file = ''

        # TODO : This is from the SEAPODYM config file
        ## TAXIS ####################################
        self.vmax_a=2.225841100458143
        """SEAPODYM name is MSS_species. Velocity at maximal habitat
        gradient and `A = 1, BL/s`."""
        self.vmax_b=0.8348850216641774
        """SEAPODYM name is MSS_size_slope. Slope coefficient in
        allometric function for tuna velocity."""
        ## CURRENTS #################################
        self.u_file = 'po_interim_historic_2x30d_u_L1_1979_2010.dym'
        self.v_file = 'po_interim_historic_2x30d_v_L1_1979_2010.dym'
        ## DIFFUSION ####################################
        self.sigma_K=0.1769952864978924
        """SEAPODYM name is sigma_species. Multiplier for the theoretical
        diffusion rate `((V̄**2) * ∆T)/4`"""
        self.c=0.662573993401526
        """SEAPODYM name is c_diff_fish. Coefficient of diffusion
        variability with habitat index."""
        self.P=3
        """The constant (`p=3`) is chosen to limit the reduction of `D0` in
        the lowest habitat indices `Ha < 0.5`."""
