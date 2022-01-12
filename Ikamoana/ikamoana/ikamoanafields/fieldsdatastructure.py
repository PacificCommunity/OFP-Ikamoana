
class FieldsDataStructure :

## NOTE : This is a hardcoded structure. Only for test.

    def __init__(self) -> None :

        ## CURRENTS #################################
        self.u_file = 'po_interim_historic_2x30d_u_L1_1979_2010.dym'
        self.v_file = 'po_interim_historic_2x30d_v_L1_1979_2010.dym'

        ## TAXIS ####################################
        self.taxis_scale=1

        # Take into account ? Normalize ?
        self.units='m_per_s',

        # Find these data name in the SEAPODYM config file
        self.vmax_a=2.225841100458143
        self.vmax_b=0.8348850216641774

        # Useless in Taxis ?
        # Can be deducte from SEAPODYM config file -> incorporate
        # directly from this data structure.
        self.timestep=30*24*60*60

        self.shallow_sea_to_ocean = False
        self.landmask_from_habitat = True

        ## DIFFUSION ####################################
        self.sigma_K=0.1769952864978924
        self.diffusion_boost=0
        self.diffusion_scale=1
        self.sig_scale=1
        self.c_scale=1
        self.c=0.662573993401526
        self.P=3

        ## FISHING MORTALITY ############################
        self.sigma_F = 8.614813906820441
        self.mu = 52.56103941719986
        self.q=0.001032652877899101
