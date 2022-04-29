import parcels.rng as ParcelsRandom
import math

# NOTE : No more SEAPODYM_dt ? Can we use particle.dt instead ?

################## Moving and cleanup kernels ####################

def MoveSouth(particle, fieldset, time):
    particle.prev_lat = particle.lat
    particle.prev_lon = particle.lon
    particle.lat = particle.lat - 0.3

def LandBlock(particle, fieldset, time):
    onland = fieldset.landmask[0, particle.depth, particle.lat, particle.lon]
    if onland == 1:
        particle.lat = particle.prev_lat
        particle.lon = particle.prev_lon

def IkaDymMove(particle, fieldset, time):
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
    adv_x = particle.Ax + particle.Tx
    adv_y = particle.Ay + particle.Ty
    
    if adv_x > 2:
        adv_x = 2
    if adv_y > 2:
        adv_y = 2
    
    particle.lon = particle.lon + adv_x + particle.Dx + particle.Cx
    particle.lat = particle.lat + adv_y + particle.Dy + particle.Cy

def IkaDimMoveWithDiffusionReroll(particle, fieldset, time):
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
    adv_x = particle.Ax + particle.Tx
    adv_y = particle.Ay + particle.Ty
    
    if adv_x > 2:
        adv_x = 2
    if adv_y > 2:
        adv_y = 2
        
    particle.loop_count = 0
    jump_loop = 0
    sections = 8
    #Check along the trajectory to make sure we're not jumping over small landmasses
    #mainly for sub 1deg forcing fields
    while jump_loop < sections:
        move_x = adv_x + particle.Dx + particle.Cx
        move_y = adv_y + particle.Dy + particle.Cy
        # Look along a transect of the potential move for land
        newlon = particle.lon + (jump_loop + 1) * (move_x/sections) # one section of the potential movement
        newlat = particle.lat + (jump_loop + 1) * (move_y/sections)
        onland = fieldset.landmask[0, particle.depth, newlat, newlon]
        jump_loop += 1
        if onland == 1:
            Rx = ParcelsRandom.uniform(-1., 1.)
            Ry = ParcelsRandom.uniform(-1., 1.)
            particle.Dx = Rx * particle.Rx_component * particle.f_lon
            particle.Dy = Ry * particle.Ry_component * particle.f_lat
            particle.loop_count += 1
            jump_loop = 0 # restart the transect
            if particle.loop_count > 500: # Give up trying to find legal moves
                move_x = 0
                move_y = 0
                jump_loop = sections # Exit the loop
        # else:
        #     if particle.prev_lat < -8.5 and particle.lat > -8.5:
        #         if particle.prev_lon < 146.5 and particle.lon > 146.5:
        #             onland = 1 # Hardcoded check for illegal Coral to Solomon Sea moves
        #     elif particle.lat < -8.5 and particle.prev_lat > -8.5:
        #         if particle.lon < 146.5 and particle.prev_lon > 146.5:
        #             onland = 1 # Hardcoded check for illegal Coral to Solomon Sea moves
        #     if particle.prev_lat > -5.5 and particle.lat < -5.5:
        #         if particle.prev_lon < 150.5 and particle.lon > 150.5:
        #             onland = 1 # Hardcoded check for illegal Bismarck to Solomon Sea moves
        #     elif particle.lat > -5.5 and particle.prev_lat < -5.5:
        #         if particle.lon < 150.5 and particle.prev_lon > 150.5:
        #             onland = 1 # Hardcoded check for illegal Bismarck to Solomon Sea moves
    particle.lon += move_x
    particle.lat += move_y

def KillFish(particle, fieldset, time):
    particle.delete()

############### Advection Kernels ####################

def CalcLonLatScalers(particle, fieldset, time):
    """See also parcels.tools.converters.GeographicPolar converter."""
    
    # Geographic _      source to target : m -> degree
    particle.f_lat = 1 / 1000. / 1.852 / 60.
    # GeographicPolar _ source to target : m -> degree
    particle.f_lon = particle.f_lat / math.cos(particle.lat*math.pi/180)

def IkAdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
    Function needs to be converted to Kernel object before execution"""
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    uv_lon1 = particle.lon + u1*.5*particle.dt
    uv_lat1 = particle.lat + v1*.5*particle.dt
    
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, uv_lat1, uv_lon1]
    uv_lon2 = particle.lon + u2*.5*particle.dt
    uv_lat2 = particle.lat + v2*.5*particle.dt
    
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, uv_lat2, uv_lon2]
    uv_lon3 = particle.lon + u3*particle.dt
    uv_lat3 = particle.lat + v3*particle.dt
    
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, uv_lat3, uv_lon3]
    particle.Ax = (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.Ay = (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

def TaxisRK4(particle, fieldset, time):
    """Method inspired by IkAdvectionRK4."""
    tx1 = fieldset.Tx[time, particle.depth, particle.lat, particle.lon]
    ty1 = fieldset.Ty[time, particle.depth, particle.lat, particle.lon]
    t_lon1 = particle.lon + tx1 * .5
    t_lat1 = particle.lat + ty1 * .5

    tx2 = fieldset.Tx[time + .5 * particle.dt, particle.depth, t_lat1, t_lon1]
    ty2 = fieldset.Ty[time + .5 * particle.dt, particle.depth, t_lat1, t_lon1]
    t_lon2 = particle.lon + tx2 * .5
    t_lat2 = particle.lat + ty2 * .5

    tx3 = fieldset.Tx[time + .5 * particle.dt, particle.depth, t_lat2, t_lon2]
    ty3 = fieldset.Ty[time + .5 * particle.dt, particle.depth, t_lat2, t_lon2]
    t_lon3 = particle.lon + tx3
    t_lat3 = particle.lat + ty3

    tx4 = fieldset.Tx[time + particle.dt, particle.depth, t_lat3, t_lon3]
    ty4 = fieldset.Ty[time + particle.dt, particle.depth, t_lat3, t_lon3]
    
    particle.Tx = (tx1 + 2*tx2 + 2*tx3 + tx4) / 6. * particle.dt * particle.f_lon
    particle.Ty = (ty1 + 2*ty2 + 2*ty3 + ty4) / 6. * particle.dt * particle.f_lat

def RandomWalkNonUniformDiffusion(particle, fieldset, time):
    # Init
    r_var = 1./3.
    Rx = ParcelsRandom.uniform(-1., 1.)
    Ry = ParcelsRandom.uniform(-1., 1.)
    # Read
    dKxdx = fieldset.dKx_dx[time, particle.depth, particle.lat, particle.lon]
    dKydy = fieldset.dKy_dy[time, particle.depth, particle.lat, particle.lon]
    kx = fieldset.Kx[time, particle.depth, particle.lat, particle.lon]
    ky = fieldset.Ky[time, particle.depth, particle.lat, particle.lon]
    # Convert
    dKxdx = dKxdx * particle.dt
    dKydy = dKydy * particle.dt
    kx = kx * particle.dt 
    ky = ky * particle.dt 
    # Compute
    particle.Rx_component = math.sqrt(2 * kx / r_var)
    particle.Ry_component = math.sqrt(2 * ky / r_var)
    particle.Dx = Rx * particle.Rx_component * particle.f_lon
    particle.Dy = Ry * particle.Ry_component * particle.f_lat
    particle.Cx = dKxdx * particle.f_lon
    particle.Cy = dKydy * particle.f_lat


######################## Mortality Kernels #########################

def FishingMortality(particle, fieldset, time):
    # particle.Fmor = fieldset.F[time, particle.depth, particle.lat, particle.lon]/fieldset.SEAPODYM_dt
    particle.Fmor = fieldset.F[time, particle.depth, particle.lat, particle.lon]/particle.dt

def NaturalMortality(particle, fieldset, time):
    Mnat = fieldset.MPmax * math.exp(-fieldset.MPexp*particle.age_class) + fieldset.MSmax*math.pow(particle.age_class, fieldset.MSslope)
    Mvar = Mnat * math.pow(1 - fieldset.Mrange,
                           1 - fieldset.H[time, particle.depth, particle.lat, particle.lon] / 2)
    particle.Nmor = Mvar/fieldset.cohort_dt


def UpdateSurvivalProbNOnly(particle, fieldset, time):
    depletion = particle.SurvProb - particle.SurvProb * math.exp(-particle.Nmor)
    particle.depletionN = depletion
    particle.SurvProb -= depletion

def UpdateSurvivalProb(particle, fieldset, time):
    particle.Zint = math.exp(-(particle.Fmor + particle.Nmor)*particle.dt)
    depletion = particle.SurvProb - particle.SurvProb * particle.Zint
    particle.depletionF = depletion*particle.Fmor/(particle.Fmor+particle.Nmor)
    particle.depletionN = depletion*particle.Nmor/(particle.Fmor+particle.Nmor)
    particle.SurvProb -= depletion
    particle.CapProb += particle.depletionF

def UpdateMixingPeriod(particle, fieldset, time):
    particle.TAL += particle.dt
    if particle.TAL > 90*86400 :
        depletion = particle.Mix3SurvProb - particle.Mix3SurvProb * particle.Zint
        depF = depletion*particle.Fmor/(particle.Fmor+particle.Nmor)
        particle.Mix3SurvProb -= depletion
        particle.Mix3CapProb += depF
    if particle.TAL > 180*86400 :
        depletion = particle.Mix6SurvProb - particle.Mix6SurvProb * particle.Zint
        depF = depletion*particle.Fmor/(particle.Fmor+particle.Nmor)
        particle.Mix6SurvProb -= depletion
        particle.Mix6CapProb += depF
    if particle.TAL > 270*86400 :
        depletion = particle.Mix9SurvProb - particle.Mix9SurvProb * particle.Zint
        depF = depletion*particle.Fmor/(particle.Fmor+particle.Nmor)
        particle.Mix9SurvProb -= depletion
        particle.Mix9CapProb += depF

###################### Field sampling kernels ########################
def getRegion(particle, fieldset, time):
    particle.region = fieldset.region[time, particle.depth, particle.lat, particle.lon]

###################### Internal state kernels ########################

def Age(particle, fieldset, time):
    particle.age += particle.dt
    if (particle.age - (particle.age_class*fieldset.cohort_dt)) > (fieldset.cohort_dt):
        particle.age_class += 1

# All Kernel dict, needed for dynamic kernel compilation

AllKernels = {'IkaDymMove':IkaDymMove,
              'IkaDimMoveWithDiffusionReroll': IkaDimMoveWithDiffusionReroll,
              'KillFish':KillFish,
              'CalcLonLatScalers':CalcLonLatScalers,
              'IkAdvectionRK4':IkAdvectionRK4,
              'TaxisRK4':TaxisRK4,
              'RandomWalkNonUniformDiffusion':RandomWalkNonUniformDiffusion,
              'FishingMortality':FishingMortality,
              'NaturalMortality':NaturalMortality,
              'UpdateSurvivalProbNOnly':UpdateSurvivalProbNOnly,
              'UpdateSurvivalProb':UpdateSurvivalProb,
              'UpdateMixingPeriod':UpdateMixingPeriod,
              'getRegion':getRegion,
              'Age':Age,
              'MoveSouth':MoveSouth,
              'LandBlock':LandBlock}
