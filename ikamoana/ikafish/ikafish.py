import numpy as np
from parcels.particle import JITParticle, ScipyParticle, Variable

# NOTE : Default parcels.particle.Variable values are :
# - dtype=np.float32
# - initial=0
# - to_write=True

class IkaFish(JITParticle):
        age          = Variable('age', to_write=False)
        age_class    = Variable('age_class', to_write=False)
        # TODO : active is not used in kernels ? Should it be removed ?
        active       = Variable("active", to_write=False, initial=1)
        prev_lon     = Variable('prev_lon', to_write=False)
        prev_lat     = Variable('prev_lat', to_write=False)
        loop_count   = Variable('loop_count', to_write=False)
        f_lat        = Variable('f_lat', to_write=False)
        f_lon        = Variable('f_lon', to_write=False)
        Dx           = Variable('Dx', to_write=False)
        Dy           = Variable('Dy', to_write=False)
        Cx           = Variable('Cx', to_write=False)
        Cy           = Variable('Cy', to_write=False)
        Tx           = Variable('Tx', to_write=False, dtype=np.float64)
        Ty           = Variable('Ty', to_write=False, dtype=np.float64)
        Ax           = Variable('Ax', to_write=False)
        Ay           = Variable('Ay', to_write=False)
        Rx_component = Variable('Rx_component', to_write=False)
        Ry_component = Variable('Ry_component', to_write=False)


        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super().__init__(*args, **kwargs)

class IkaFADFish(ScipyParticle):
    ptype        = Variable('ptype', to_write=True)
    FADkap       = Variable('FADkap', to_write=True)
    age          = Variable('age', to_write=False)
    age_class    = Variable('age_class', to_write=False)
    prev_lon     = Variable('prev_lon', to_write=False)
    prev_lat     = Variable('prev_lat', to_write=False)
    loop_count   = Variable('loop_count', to_write=False)
    f_lat        = Variable('f_lat', to_write=False)
    f_lon        = Variable('f_lon', to_write=False)
    Dx           = Variable('Dx', to_write=True)
    Dy           = Variable('Dy', to_write=True)
    Cx           = Variable('Cx', to_write=True)
    Cy           = Variable('Cy', to_write=True)
    Tx           = Variable('Tx', to_write=True, dtype=np.float64)
    Ty           = Variable('Ty', to_write=True, dtype=np.float64)
    Ax           = Variable('Ax', to_write=True)
    Ay           = Variable('Ay', to_write=True)
    Fx           = Variable('Fx', to_write=True)
    Fy           = Variable('Fy', to_write=True)
    Rx_component = Variable('Rx_component', to_write=False)
    Ry_component = Variable('Ry_component', to_write=False)

    CapProb = Variable('CapProb', to_write=False)
    SurvProb = Variable('SurvProb', initial=1, to_write=True)
    depletionF = Variable('depletionF', to_write=False)
    depletionN = Variable('depletionN', to_write=False)
    Fmor = Variable('Fmor', to_write=False)
    Nmor = Variable('Nmor', to_write=False)
    Zint = Variable('Zint', to_write=False)

    def __init__(self, *args, **kwargs):
        """Custom initialisation function which calls the base
        initialisation and adds the instance variable p"""
        super().__init__(*args, **kwargs)

class IkaFishDebug(ScipyParticle):
    # Internal
    active       = Variable("active", to_write=False, initial=1)
    prev_lon     = Variable('prev_lon', to_write=False)
    prev_lat     = Variable('prev_lat', to_write=False)
    # Debug
    age          = Variable('age')
    age_class    = Variable('age_class')
    loop_count   = Variable('loop_count')
    f_lat        = Variable('f_lat')
    f_lon        = Variable('f_lon')
    Dx           = Variable('Dx')
    Dy           = Variable('Dy')
    Cx           = Variable('Cx')
    Cy           = Variable('Cy')
    Ax           = Variable('Ax')
    Ay           = Variable('Ay')
    Tx           = Variable('Tx', dtype=np.float64)
    Ty           = Variable('Ty', dtype=np.float64)
    Rx_component = Variable('Rx_component')
    Ry_component = Variable('Ry_component')

    def __init__(self, *args, **kwargs):
        """Custom initialisation function which calls the base
        initialisation and adds the instance variable p"""
        super().__init__(*args, **kwargs)


class IkaTag(IkaFish):
        region = Variable('region')
        CapProb = Variable('CapProb')
        SurvProb = Variable('SurvProb', initial=1)
        depletionF = Variable('depletionF', to_write=False)
        depletionN = Variable('depletionN', to_write=False)
        Fmor = Variable('Fmor', to_write=False)
        Nmor = Variable('Nmor', to_write=False)
        Zint = Variable('Zint', to_write=False)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super().__init__(*args, **kwargs)


class IkaMix(IkaTag):
        TAL = Variable('TAL', to_write=False)
        Mix3CapProb = Variable('Mix3CapProb')
        Mix6CapProb = Variable('Mix6CapProb')
        Mix9CapProb = Variable('Mix9CapProb')
        Mix3SurvProb = Variable('Mix3SurvProb', to_write=True, initial=1)
        Mix6SurvProb = Variable('Mix6SurvProb', to_write=True, initial=1)
        Mix9SurvProb = Variable('Mix9SurvProb', to_write=True, initial=1)


        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super().__init__(*args, **kwargs)
