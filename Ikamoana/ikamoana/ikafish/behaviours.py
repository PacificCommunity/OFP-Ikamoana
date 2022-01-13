import math

################## Moving and cleanup kernels ####################

def IkaDymMove(particle, fieldset, time):
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
    adv_x = Ax + tx
    adv_y = Ay + ty
    if adv_x > 2:
        adv_x = 2
    if adv_y > 2:
        adv_y = 2
    move_x = adv_x + Dx + Cx
    move_y = adv_y + Dy + Cy
    particle.lon = particle.lon + move_x
    particle.lat = particle.lat + move_y

def IkaDimMoveWithDiffusionReroll(particle, fieldset, time):
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
    adv_x = Ax + Tx
    adv_y = Ay + Ty
    if adv_x > 2:
        adv_x = 2
    if adv_y > 2:
        adv_y = 2
    loop_count = 0
    jump_loop = 0
    sections = 8
    #Check along the trajector to make sure we're not jumping over small landmasses
    #mainly for sub 1deg forcing fields
    while jump_loop < sections:
        move_x = adv_x + Dx + Cx
        move_y = adv_y + Dy + Cy
        # Look along a transect of the potential move for land
        newlon = particle.lon + (jump_loop + 1) * (move_x/sections) # one section of the potential movement
        newlat = particle.lat + (jump_loop + 1) * (move_y/sections)
        onland = fieldset.landMask[0, particle.depth, newlat, newlon]
        jump_loop += 1
        if onland == 1:
            Rx = random.uniform(-1., 1.)
            Ry = random.uniform(-1., 1.)
            Dx = Rx * Rx_component * f_lon
            Dy = Ry * Ry_component * f_lat
            loop_count += 1
            jump_loop = 0 # restart the transect
            if loop_count > 500: # Give up trying to find legal moves
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

def SaveMoveComponents(particle, fieldset, time):
    particle.Ax = Ax
    particle.Ay = Ay
    particle.Tx = Tx
    particle.Ty = Ty
    particle.Dx = Dx
    particle.Dy = Dy
    particle.Cx = Cx
    particle.Cy = Cy
    particle.In_Loop = loop_count

def KillFish(particle, fieldset, time):
    particle.delete()

############### Advection Kernels ####################
def CalcLonLatScalers(particle, fieldset, time):
    f_lat = particle.dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(particle.lat*math.pi/180)

def IkAdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
    Function needs to be converted to Kernel object before execution"""
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
    particle.Ax = (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.Ay = (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    Ax = particle.Ax
    Ay = particle.Ay

def TaxisRK4(particle, fieldset, time):
    u1 = fieldset.Tx[time, particle.depth, particle.lat, particle.lon]
    v1 = fieldset.Ty[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1*.5, particle.lat + v1*.5)
    u2, v2 = (fieldset.Tx[time + .5 * particle.dt, particle.depth, lat1, lon1], fieldset.Ty[time + .5 * particle.dt, particle.depth, lat1, lon1])
    lon2, lat2 = (particle.lon + u2*.5, particle.lat + v2*.5)
    u3, v3 = (fieldset.Tx[time + .5 * particle.dt, particle.depth, lat2, lon2], fieldset.Ty[time + .5 * particle.dt, particle.depth, lat2, lon2])
    lon3, lat3 = (particle.lon + u3, particle.lat + v3)
    u4, v4 = (fieldset.Tx[time + particle.dt, particle.depth, lat3, lon3], fieldset.Ty[time + particle.dt, particle.depth, lat3, lon3])
    particle.Tx = (u1 + 2*u2 + 2*u3 + u4) / 6. * f_lon
    particle.Ty = (v1 + 2*v2 + 2*v3 + v4) / 6. * f_lat
    tx = particle.Tx
    ty = particle.Ty

def RandomWalkNonUniformDiffusion(particle, fieldset, time):
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    dKdx, dKdy = (fieldset.dK_dx[time, particle.depth, particle.lat, particle.lon], fieldset.dK_dy[time, particle.depth, particle.lat, particle.lon])
    k = fieldset.K[time, particle.depth, particle.lat, particle.lon]
    Rx_component = math.sqrt(2 * k * particle.dt / r_var)
    Ry_component = math.sqrt(2 * k * particle.dt / r_var)
    CorrectionX = dKdx * f_lon
    CorrectionY = dKdy * f_lat
    Dx = Rx * Rx_component * f_lon /particle.dt
    Dy = Ry * Ry_component * f_lat /particle.dt
    Cx = CorrectionX
    Cy = CorrectionY
    particle.Cx = Cx
    particle.Cy = Cy
    particle.Dx = Dx
    particle.Dy = Dy


######################## Mortality Kernels #########################

def FishingMortality(particle, fieldset, time):
    Fmor = fieldset.F[time, particle.depth, particle.lat, particle.lon]
    particle.Fmor = Fmor

def NaturalMortality(particle, fieldset, time):
    Mnat = fieldset.MPmax*math.exp(-fieldset.MPexp*particle.age_class) + fieldset.MSmax*math.pow(particle.age_class, fieldset.MSslope)
    Mvar = Mnat * math.pow(1 - fieldset.Mrange, 1-fieldset.H[time, particle.depth, particle.lat, particle.lon]/2)
    Nmor = (Mvar * (particle.dt / (7 * 24 * 60 * 60))) #The division by the seapodym dt needs to be generalised
    particle.Nmor = Nmor

def UpdateSurvivalProb(particle, fieldset, time):
    depletion = particle.SurvProb - particle.SurvProb * math.exp(-(Fmor + Nmor))
    particle.depletionF = depletion*Fmor/(Fmor+Nmor)
    particle.depletionN = depletion*Nmor/(Fmor+Nmor)
    particle.SurvProb -= depletion
    particle.CapProb += particle.depletionF


###################### Internal state kernels ########################

def Age(particle, fieldset, time):
    particle.age += particle.dt
    if (particle.age - (particle.age_class*fieldset.cohort_dt)) > (fieldset.cohort_dt):
        particle.age_class += 1

# All Kernel dict, needed for dynamic kernel compilation

AllKernels = {'IkaDymMove':IkaDymMove,
              'IkaDimMoveWithDiffusionReroll': IkaDimMoveWithDiffusionReroll,
              'SaveMoveComponents':SaveMoveComponents,
              'KillFish':KillFish,
              'CalcLonLatScalers':CalcLonLatScalers,
              'IkAdvectionRK4':IkAdvectionRK4,
              'TaxisRK4':TaxisRK4,
              'RandomWalkNonUniformDiffusion':RandomWalkNonUniformDiffusion,
              'FishingMortality':FishingMortality,
              'NaturalMortality':NaturalMortality,
              'UpdateSurvivalProb':UpdateSurvivalProb,
              'Age':Age}
