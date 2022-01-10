import numpy as np
from parcels.particle import JITParticle, ScipyParticle, Variable

class IkaFish(JITParticle):
        active = Variable("active", to_write=True)
        age = Variable('age',dtype=np.float32, to_write=False)
        age_class = Variable('age_class',dtype=np.float32)
        Dx = Variable('Dx', to_write=False, dtype=np.float32)
        Dy = Variable('Dy', to_write=False, dtype=np.float32)
        Cx = Variable('Cx', to_write=False, dtype=np.float32)
        Cy = Variable('Cy', to_write=False, dtype=np.float32)
        Tx = Variable('Tx', to_write=False, dtype=np.float64)
        Ty = Variable('Ty', to_write=False, dtype=np.float64)
        Ax = Variable('Ax', to_write=False, dtype=np.float32)
        Ay = Variable('Ay', to_write=False, dtype=np.float32)
        prev_lon = Variable('prev_lon', to_write=False)
        prev_lat = Variable('prev_lat', to_write=False)
        In_Loop = Variable('In_Loop', to_write=False, dtype=np.float32)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(IkaFish, self).__init__(*args, **kwargs)
            self.Dx = self.Dy = self.Cx = self.Cy = self.Tx = self.Ty = self.Ax = self.Ay = 0.
            self.active = 1

class IkaTag(IkaFish):
        CapProb = Variable('CapProb', to_write=True, dtype=np.float32)
        SurvProb = Variable('SurvProb', to_write=True, dtype=np.float32)
        region = Variable('region', to_write=True, dtype=np.float32)
        depletionF = Variable('depletionF', to_write=False, dtype=np.float32)
        depletionN = Variable('depletionN', to_write=False, dtype=np.float32)
        Fmor = Variable('Fmor', to_write=True, dtype=np.float32)
        Nmor = Variable('Nmor', to_write=False, dtype=np.float32)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(IkaTag, self).__init__(*args, **kwargs)
            self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0.
            self.active = 1
            self.CapProb = 0
            self.SurvProb = 1
