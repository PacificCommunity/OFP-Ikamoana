import parcels.rng as ParcelsRandom
import math
import scipy
from scipy.stats import vonmises
import parcels.rng as ParcelsRandom
import numpy as np
from copy import copy

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

def Faugeras(particle, fieldset, time):
    # Tuna swimming based on Faugeras and Maury (2007)
    if(particle.ptype == 0):
        def LogisticCurve(x, L=fieldset.pL, k=15, x0=0.7):
            # x is the stomach fullness
            x = 1 - x # make it stomach emptiness
            res = L / (1+math.e**(-k*(x-x0)))
            return res

        gradx = (fieldset.P[time,0,particle.lat, particle.lon+fieldset.gres] -
                 fieldset.P[time,0,particle.lat, particle.lon-fieldset.gres])
        grady = (fieldset.P[time,0,particle.lat+fieldset.gres, particle.lon] -
                 fieldset.P[time,0,particle.lat-fieldset.gres, particle.lon])

        mu = np.arctan2(grady, gradx) # mean angle based on prey field gradient

        kappaM = fieldset.alpha * np.linalg.norm([gradx*fieldset.gres, grady*fieldset.gres]) # standard deviation angle
        angle = ParcelsRandom.vonmisesvariate(mu, kappaM)

        # the particle displacement
        particle.Tx = fieldset.kappaP * np.cos(angle) * LogisticCurve(particle.St)
        particle.Ty = fieldset.kappaP * np.sin(angle) * LogisticCurve(particle.St)

######################## Mortality Kernels #########################

def dyingFish(particle, fieldset, time):
    if(particle.ptype==0):
        rnumber = np.random.beta(fieldset.betaa, fieldset.betab)
        if(False):#particle.SurvProb < 0):#rnumber):
            particle.delete()


def FishingMortalityFADsim(particle, fieldset, time):
    def geometric(x, p=fieldset.p, nfad=fieldset.nfad):
        # determine geometric probability
        if(p==0): # return uniform distribution
            return 1 / nfad
        else: # geometric distribution
            xa = np.arange(nfad+1)
            norm = ((1-p)**(xa-1)*p).sum()
            return (1-p)**(x-1)*p / norm

    if(particle.ptype==0 and particle.FADkap>=0): # if tuna particle associated to a FAD
        # if tuna is associated to FAD with many associated tuna, it
        # experiences more mortality, depending on parameter p of the geometric distribution
        fr = fieldset.FADorders.data[0][0].tolist()
        nl = np.array([fr.index(x) for x in sorted(fr, reverse=True)[:fieldset.nfad]])
        diff = len(nl) - len(np.unique(nl))
        bo = True
        for diff in range(len(nl)):
            if(bo):
                if(particle.FADkap-diff in nl):
                    geoP = geometric(np.where(nl==particle.FADkap-diff)[0][0])
                    bo = False

        # Final fishing mortality:
        particle.Fmor = geoP * fieldset.FADfishingP * fieldset.F[time, particle.depth, particle.lat, particle.lon]/particle.dt

    elif(particle.ptype==0 and particle.FADkap==-1): # if tuna particle not associated to FAD
        particle.Fmor = (1-fieldset.FADfishingP) * fieldset.F[time, particle.depth, particle.lat, particle.lon]/particle.dt
    
def FishingMortality(particle, fieldset, time):
    # particle.Fmor = fieldset.F[time, particle.depth, particle.lat, particle.lon]/fieldset.SEAPODYM_dt
    particle.Fmor = fieldset.F[time, particle.depth, particle.lat, particle.lon]/particle.dt

def NaturalMortality(particle, fieldset, time):
    Mnat = fieldset.MPmax * math.exp(-fieldset.MPexp*particle.age_class) + fieldset.MSmax*math.pow(particle.age_class, fieldset.MSslope)
    Mvar = Mnat * math.pow(1 - fieldset.Mrange,
                           1 - fieldset.H[time, particle.depth, particle.lat, particle.lon] / 2)
    particle.Nmor = Mvar/fieldset.cohort_dt


def UpdateSurvivalProbNOnly(particle, fieldset, time):
    depletion = particle.SurvProb * math.exp(-particle.Nmor)
    particle.depletionN = depletion
    particle.SurvProb -= depletion

def UpdateSurvivalProb(particle, fieldset, time):
    particle.Zint = math.exp(-(particle.Fmor + particle.Nmor)*particle.dt)
    depletion = particle.SurvProb -  particle.SurvProb * particle.Zint
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


###################### Particle-Particle Interaction kernels #####################

def Iattraction(particle, fieldset, time, neighbors, mutator):
    """Kernel determines the attraction strength of FADs,
       determined by Logistic function"""
    def f(particle, nom):  # define mutation function for mutator
        particle.FADkap = nom

    # if the FAD attraction strength is determined
    # by the number of associated tuna
    if particle.ptype==1:
        nom = 0 # keeps track of number of associated tuna
        for n in neighbors:
            if n.ptype==0:
                dist = ((particle.lat-n.lat)**2+(particle.lon-n.lon)**2)**0.5
            else:
                dist = np.inf
            if dist <= fieldset.RtF:
                nom += 1
        mutator[particle.id].append((f, [nom]))  # add mutation to the mutator

        fieldset.FADorders.data[0,0,particle.id] = nom
        fieldset.Forders.grid.time[0] = time # updating Field prey time

    # flag to which dFAD a tuna particle is associated
    elif particle.ptype==0:
        particle.FADkap = 0
        anya = True
        for n in neighbors:
            if n.ptype==1:
                dist = ((particle.lat-n.lat)**2+(particle.lon-n.lon)**2)**0.5
            else:
                dist = np.inf
            if dist <= fieldset.RtF:
                particle.FADkap = n.id
                anya=False
        if(anya):
            particle.FADkap = -1
    return StateCode.Success

def ItunaFAD(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel that "pulls" all neighbor tuna particles of FADs
    toward the FAD'''
    distances = []
    na_neighbors = []

     # the swimming
    if particle.ptype==0: # if tuna swims towards FAD
        # Define the Logistic curve
        def LogisticCurve(x, L=fieldset.lL, k=fieldset.lk, x0=fieldset.lx0):
         # x is the number of associated tuna
            res = 1 + L / (1+math.e**(-k*(x-x0)))
            return res

        #Reset the particle FAD vector
        DS = [0,0]
        for n in neighbors:
            #if neighbour is a FAD, determine the normalised FAD and add to Fx/Fy
            if n.ptype==1:
                pPos = np.array([particle.lat, particle.lon, particle.depth]) # n location
                fPos = np.array([n.lat, n.lon, n.depth]) # FAD location
                assert particle.depth==n.depth, 'this kernel is only supported in two dimensions for now'
                Fvec = [f-p for f,p in zip(fPos, pPos)]
                Fmag = math.sqrt(math.pow(Fvec[0],2)+math.pow(Fvec[1],2))
                if Fmag == 0:
                    Fnorm = [0,0]
                else:
                    Fnorm = [Fvec[0]/Fmag,
                            Fvec[1]/Fmag]
                #if Fnorm>0 :
                DS[0] += Fnorm[0] * LogisticCurve(n.FADkap)
                DS[1] += Fnorm[1] * LogisticCurve(n.FADkap)

        if DS!=[0,0]:
            #if FAD vector is non-zero, add mutator to update Fx/Fy
            VP = [0,0,0]
            VP[0] = DS[0] * fieldset.kappaF
            VP[1] = DS[1] * fieldset.kappaF
            d_vec = VP
            particle.Fy = VP[0]
            particle.Fx = VP[1]
        else:
            particle.Fy = 0
            particle.Fx = 0

    return StateCode.Success

def Imovement(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel resolves all displacment vectors following
    interactive and non-interactive kernel execution'''
    def A(drifter): #Advection mutator
        drifter.lon += drifter.Ax
        drifter.lat += drifter.Ay
    def S(p): #Swimming mutator
        S = np.array([p.Tx+p.Dx+p.Cx,p.Ty+p.Dy+p.Cy])
        Smag = math.sqrt(math.pow(S[0],2)+math.pow(S[1],2))
        if Smag == 0:
            Snorm = [0,0]
        else:
            Snorm = [S[0]/Smag,
                    S[1]/Smag]
        #add normalised FAD vector to normalised swimming vector
        dlon = Snorm[0]+p.Fx
        dlat = Snorm[1]+p.Fy
        norm = (dlon**2+dlat**2)**0.5
        if(norm>0):
            dlon /= norm
            dlat /= norm
        Smag = (S[0]**2+S[1]**2)**0.5
        p.lon += dlon*Smag
        p.lat += dlat*Smag
    #all particles are advected by flow
    mutator[particle.id].append((A,[]))
    if particle.ptype == 0: #only fish swim
        mutator[particle.id].append((S,[]))
    return StateCode.Success

def ImovementFaugeras(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel resolves all displacment vectors following
    interactive and non-interactive kernel execution'''

    def A(drifter): #Advection mutator
        drifter.lon += drifter.Ax
        drifter.lat += drifter.Ay
    def S(p): #Swimming mutator
        S = np.array([p.Fx+p.Tx+p.Dx+p.Cx,p.Fy+p.Ty+p.Dy+p.Cy])
        Smag = math.sqrt(math.pow(S[0],2)+math.pow(S[1],2))
        if Smag == 0:
            Snorm = [0,0]
        else:
            Snorm = [S[0]/Smag,
                    S[1]/Smag]
        #add normalised FAD vector to normalised swimming vector
        dlon = S[0]
        dlat = S[1]

        # zonal and meriodional converters (meter to latitude)
        converter = np.array([1852*60, 1852*60*np.cos(p.lat*math.pi/180)])
        Vmax = fieldset.Vmax / converter
        Vmag = Vmax*(1-fieldset.P[0, 0, p.lat, p.lon])
        p.lon += dlon*Vmag[0]*particle.dt
        p.lat += dlat*Vmag[1]*particle.dt
    #all particles are advected by flow
    mutator[particle.id].append((A,[]))
    if particle.ptype == 0: #only fish swim
        mutator[particle.id].append((S,[]))
    return StateCode.Success

###################### Particle-Field Interaction kernels #####################

def PreyDepletion(particle, fieldset, time):
    # Determine location index of particle in Prey field
    xi = (np.abs(np.array(fieldset.P.lon)-particle.lon)).argmin()
    yi = (np.abs(np.array(fieldset.P.lat)-particle.lat)).argmin()

    # field depletion
    deplete = min(fieldset.P[particle], fieldset.deplete/(86400)*particle.dt)
    if(deplete<0):
        print('DEPLETE SMALLER THAN 0')

    fieldset.F.data[0, yi, xi] -= deplete
    fieldset.F.grid.time[0] = time # updating Field F time

    # These functions are age dependent, so will have to be implemented
    # in the kernel that displaces the tuna particle.
    def l(a, linf=87.96, Agemin=3., k=2.564600347374714):
        # skipjack:
        #     k = 2.564600347374714
        #     linf = 87.96 (cm?)
        #     a0 = 3 (years)
        return linf * (1-math.e**(-k * (a-Agemin)))

    def weight(age, p=3, q=0.2):
        # skipjack:
        #     p = 
        #     q = 
        return q*l(age)**p

    def _sigmaStar(age, sigma_0=2.388505133695817,
                   sigma_K=3.255065398014956,
                   Agemax=11):
        """Return sigmaStar (the termal tolerance intervals, i.e. standard
        deviation) for each cohorts."""
        w = weight(age)
        mw = weight(Agemax)

        return sigma_0 + ((sigma_K - sigma_0)
                          * (w / mw))

    def _tStar(age, T_star_1=31.13068425631411,
               T_star_K=27.54999999968445,
               bT=2.564600347374714,
               Agemax=11) :
        """Return T_star (optimal temperature, i.e. mean) for each cohorts"""
        length = l(age)
        mlength = l(Agemax)
        return T_star_1 - ((T_star_1 - T_star_K)
                           * ((length / mlength)**bT))

    def f4(age, T):
        res = math.e**(-(T-_tStar(age))**2 / (2*_sigmaStar(age)**2))
        return res

    def f5(O2, O2s=3.76, gamma=10**-4):
        # bigeye: O2s=1.49, gamma=0.001
        # skipjack: O2s=3.76, gamma=10**-4
        return 1 / (1+gamma**(O2-O2s))

    def theta(age, O2, T):
        # oxygen O2 and temperature T should
        # be obtained from their fields.
        return f4(age, T)*f5(O2)

    def scaleP(data):
        phi = 22.5 * np.pi/180.
        a = 0.07
        e = 1. / np.cos(phi)
        b = a * np.sqrt(e*e - 1)

        # coordinate center
        # shift is to have all y>=0
        x0 = 1.0-0.00101482322788
        y0 = 1.0
        # equation for hyperbola
        sinsq = np.sin(phi) * np.sin(phi)
        cossq = 1.0-sinsq
        rasq  = 1.0 / (a*a)
        rbsq  = 1.0 / (b*b)
        A = sinsq*rasq - cossq*rbsq
        B = -2.0 * (data-x0) * np.cos(phi) * np.sin(phi) * (rasq+rbsq)
        C = 1.0 - (data-x0) * (data-x0) * (sinsq*rbsq - cossq*rasq)

        return (y0+(B+np.sqrt(B*B-4.0*A*C))/(2*A))
    fieldset.P.data[0, yi, xi] = scaleP(theta(particle.age,
        fieldset.O2[time, particle.depth, particle.lat, particle.lon],
        fieldset.T[time, particle.depth, particle.lat, particle.lon]) * fieldset.F.data[0, yi, xi])
    fieldset.P.grid.time[0] = time # updating Field P time

def PreyAdvectionMICRestore(particle, fieldset, time):
    if(particle.id==0):# and (fieldset.Pdiff.data[0]!=0).any()): # only once per time step
        # advect the difference according to the advection scheme
        from scipy import interpolate
        def remove_land(fieldset, glon, glat, vals):
            ocean = np.where(np.logical_and(fieldset.U.data[0].flatten()!=0,
                                            fieldset.V.data[0].flatten()!=0)
                            )
            glon = glon[ocean]
            glat = glat[ocean]
            vals = vals[ocean]
            return vals, glon, glat

        def interpolator(fieldset, field, tracer=True, op2=False):
            # time interpolation of H field
            ti = field.time_index(time)
            tint_field = field.temporal_interpolate_fullfield(ti[0], time)

            values = tint_field.flatten()
            gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
            gridlon, gridlat = (gridlon.flatten(), gridlat.flatten())

            grid_x, grid_y = np.meshgrid(fieldset.P.lon,fieldset.P.lat)
            if(True): # remove land data before interpolation
                values, gridlon, gridlat = remove_land(fieldset,
                                                       gridlon,
                                                       gridlat,
                                                       values.flatten())
            points = np.swapaxes(np.vstack((gridlat, gridlon)), 0, 1)

            if(op2):
                dataI = interpolate.griddata(points, values, (grid_y, grid_x), method='nearest')
            else:
                dataI = interpolate.griddata(points, values, (grid_y, grid_x), method='linear')

            return dataI


        def Adv_2D(T0, nt, dt, dx, dz, Vx, Vz, Lx1, Lx2, Lz1, Lz2, x, z, X, Z):
            """
            Computes and returns the temperature distribution
            after a given number of time steps for the 2D advection 
            problem. A marker-in-cell approach with Dirichlet conditions 
            on all boundaries is used in order to mitigate the effect of 
            numerical diffusion.
            see https://nbviewer.org/github/daniel-koehn/Differential-equations-earth-system/blob/master/08_Convection_2D/02_2D_Linear_Advection.ipynb

            Parameters
            ----------
            T0 : np.ndarray
                The initial temperature distribution as a 2D array of floats.
            nt : integer
                Maximum number of time steps to compute.
            dt : float
                Time-step size.
            dx : float
                Grid spacing in the x direction.
            dz : float
                Grid spacing in the z direction.
            Vx : float
                x-component of the velocity field.
            Vz : float
                y-component of the velocity field.        
            Lx1, Lx2 : float
                Model extension from Lx1 - Lx2.
            Lz1, Lz2 : float
                Model extension from Lz1 - Lz2.
            x, z : float
                Model coordinates as 1D arrays.
            X, Z : float
                Model coordinates as 2D arrays.    
                
            
            Returns
            -------
            T : np.ndarray
                The temperature distribution as a 2D array of floats.
            """
        
            # Integrate in time.
            T = T0.copy()
            
            # Estimate number of grid points in x- and z-direction
            nz, nx = T.shape
            
            # Define number of markers and initial marker positions
            nx_mark = 4 * nx  # number of markers in x-direction
            nz_mark = 4 * nz  # number of markers in z-direction    
            
            # Time loop
            for n in range(nt):
                # initial marker positions
                x_mark = np.linspace(Lx1, Lx2, num=nx_mark)
                z_mark = np.linspace(Lz1, Lz2, num=nz_mark)
                X_mark, Z_mark = np.meshgrid(x_mark,z_mark)
                
                # Interpolate velocities from grid to marker position at timestep n        
                f = interpolate.interp2d(x, z, Vx, kind='linear')
                vx_mark_n = f(x_mark, z_mark)
                
                f = interpolate.interp2d(x, z, Vz, kind='linear')
                vz_mark_n = f(x_mark, z_mark)
                
                # Interpolate temperature from grid to marker position at timestep n
                f = interpolate.interp2d(x, z, T, kind='cubic')
                T_mark = f(x_mark, z_mark)
                
                # Save current marker positions
                X0 = X_mark
                Z0 = Z_mark
                
                # Update marker position
                X_mark = X_mark + vx_mark_n * dt
                Z_mark = Z_mark + vz_mark_n * dt
                
                # Interpolate velocities from grid to marker position at timestep n+1 
                vx_mark_n1 = interpolate.griddata((X.flatten(), Z.flatten()), Vx.flatten(), (X_mark, Z_mark), method='linear')
                vz_mark_n1 = interpolate.griddata((X.flatten(), Z.flatten()), Vz.flatten(), (X_mark, Z_mark), method='linear')
                
                # Replace Nan values 
                mask = np.where(np.isnan(vx_mark_n1))
                vx_mark_n1[mask] = 0
                mask = np.where(np.isnan(vz_mark_n1))
                vz_mark_n1[mask] = 0

                # Update marker position with midpoint velocity
                X_mark = X0 + dt * (vx_mark_n + vx_mark_n1) / 2.
                Z_mark = Z0 + dt * (vz_mark_n + vz_mark_n1) / 2.
        
                # Interpolate temperature field from marker to grid positions
                T = interpolate.griddata((X_mark.flatten(), Z_mark.flatten()), T_mark.flatten(), (X, Z), method='cubic')
                
                # Replace Nan-values by old temperature field 
                mask = np.where(np.isnan(T))
                T[mask] = T0[mask] 
            return T

        def set_land(fieldset, VV, i=0):
            if(i==0):
                VV[np.where(fieldset.Land.data[0].astype(bool))] = 0
            else:
                VV[i:][np.where(fieldset.Land.data[0,:-i].astype(bool))] = 0
                VV[:-i][np.where(fieldset.Land.data[0, i:].astype(bool))] = 0
                VV[:,i:][np.where(fieldset.Land.data[0, :,:-i].astype(bool))] = 0
                VV[:,:-i][np.where(fieldset.Land.data[0, :, i:].astype(bool))] = 0
            return VV

        def AO(T):
            Tr = T.copy()
            Dx = 0.1
            Dy = 0.1
            Uv = interpolator(fieldset, fieldset.U, tracer=False)
            Vv = interpolator(fieldset, fieldset.V, tracer=False)

            lons, lats = (fieldset.P.lon[:], fieldset.P.lat[:])
            X, Y = np.meshgrid(lons, lats)
            Uv /= (1852*60*np.cos(Y*np.pi/180))
            Vv /= (1852*60)
            T = Adv_2D(Tr, 1, particle.dt, Dx, Dy, Uv, Vv,
                       lons[0], lons[-1], lats[0], lats[-1],
                       lons, lats, X, Y)
            return T


        def interpolatorBC(fieldset, field, tracer=True, op2=False):
            # time interpolation of F field
            ti = field.time_index(time)
            tint_field = field.temporal_interpolate_fullfield(ti[0], time)

            values = tint_field.flatten()
            gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
            gridlon, gridlat = (gridlon.flatten(), gridlat.flatten())

            grid_x_west, grid_y_west = np.meshgrid(fieldset.P.lon[0],fieldset.P.lat)
            grid_x_east, grid_y_east = np.meshgrid(fieldset.P.lon[-1],fieldset.P.lat)
            grid_x_south, grid_y_south = np.meshgrid(fieldset.P.lon,fieldset.P.lat[0])
            grid_x_north, grid_y_north = np.meshgrid(fieldset.P.lon,fieldset.P.lat[-1])
            if(op2):
                values[np.where(values<0)] = 0
                values[np.where(np.isnan(values)==0)] = 0
            elif(False):#tracer): # remove land data before interpolation
                values, gridlon, gridlat = remove_land(fieldset,
                                                       gridlon,
                                                       gridlat,
                                                       values.flatten())


            points = np.swapaxes(np.vstack((gridlat, gridlon)), 0, 1)
            dataEast = interpolate.griddata(points, values, (grid_y_west, grid_x_west), method='linear')
            dataWest = interpolate.griddata(points, values, (grid_y_east, grid_x_east), method='linear')
            dataSouth = interpolate.griddata(points, values, (grid_y_south, grid_x_south), method='linear')
            dataNorth = interpolate.griddata(points, values, (grid_y_north, grid_x_north), method='linear')

            return dataEast, dataWest, dataSouth, dataNorth

        # Update domain boundaries
        west, east, south, north = interpolatorBC(fieldset, fieldset.epi_mnk_pb)
        fieldset.F.data[0,:,0] = west[:,0]
        fieldset.F.data[0,:,-1] = east[:,0]
        fieldset.F.data[0,0] = south[0]
        fieldset.F.data[0,-1] = north[0]

        # Advect F field by flow
        fieldset.F.data[0,1:-1,1:-1] = AO(fieldset.F.data[0])[1:-1,1:-1]

        fieldset.F.grid.time[0] = time # updating Field P time


def PreyAdvectionFournierSibertRestore(particle, fieldset, time):
    if(particle.id==0):# and (fieldset.Pdiff.data[0]!=0).any()): # only once per time step
        # advect the difference according to the advection scheme
        from scipy import interpolate
        from scipy.linalg import lu_factor, lu_solve
        def remove_land(fieldset, glon, glat, vals):
            ocean = np.where(np.logical_and(fieldset.U.data[0].flatten()!=0,
                                            fieldset.V.data[0].flatten()!=0)
                            )
            glon = glon[ocean]
            glat = glat[ocean]
            vals = vals[ocean]
            return vals, glon, glat

        def BCs_fromland(vel, direction='x'):
            land = (vel==0)
            if(direction=='x'):
                # left closed:
                LC = np.zeros(land.shape)
                LC[:,:1] = 1
                LC[:,1:][np.where(land[:,:-1])] = 1
                # right closed:
                RC = np.zeros(land.shape)
                RC[:,-1:] = 1
                RC[:,:-1][np.where(land[:,1:])] = 1
                return land, LC, RC
            elif(direction=='y'):
                # bottom closed:
                BC = np.zeros(land.shape)
                BC[:1] = 1
                BC[1:][np.where(land[:,:-1])] = 1
                # upper closed:
                UC = np.zeros(land.shape)
                UC[-1:] = 1
                UC[:-1][np.where(land[:,1:])] = 1
                return land, BC, UC

        def interpolator(fieldset, field, tracer=True, op2=False):
            # time interpolation of H field
            ti = field.time_index(time)
            tint_field = field.temporal_interpolate_fullfield(ti[0], time)

            values = tint_field.flatten()
            gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
            gridlon, gridlat = (gridlon.flatten(), gridlat.flatten())

            grid_x, grid_y = np.meshgrid(fieldset.P.lon,fieldset.P.lat)
            if(op2):
                values[np.where(values<0)] = 0
                values[np.where(np.isnan(values)==0)] = 0
            elif(True):#tracer): # remove land data before interpolation
                values, gridlon, gridlat = remove_land(fieldset,
                                                       gridlon,
                                                       gridlat,
                                                       values.flatten())


            points = np.swapaxes(np.vstack((gridlat, gridlon)), 0, 1)
            dataI = interpolate.griddata(points, values, (grid_y, grid_x), method='linear')

            return dataI

        def compA(sig, u, d):
            land, LC, RC = BCs_fromland(u, direction='x')
            sh = u.shape[1]
            A = np.zeros((u.shape[0], sh, sh))
            dt = particle.dt
            for j in range(u.shape[0]):
                idxrc1 = np.logical_and(u[j]>0,
                                        RC[j]==1)
                idxrc2 = np.logical_and(u[j]<0,
                                        RC[j]==1)
                idxlc1 = np.logical_and(u[j]>0,
                                        LC[j]==1)
                idxlc2 = np.logical_and(u[j]<0,
                                        LC[j]==1)
                idxopen1 = (u[j]>0)
                idxopen2 = (u[j]<0)
                a = np.zeros(sh-1)
                b = np.zeros(sh)
                c = np.zeros(sh-1)
                for i in range(u.shape[1]):
                    if(land[j,i]):
                        b[i] = 2/dt
                    elif(idxrc1[i]):
                        a[i-1] = -(u[j, i-1]/d - (sig[j,i]+sig[j,i-1])/(2*d**2))
                        b[i] = 2/dt + (sig[j,i]+sig[j,i-1])/(2*d**2)
                    elif(idxrc2[i]):
                        a[i-1] = -(sig[j,i]+sig[j, i-1]) / (2*d**2)
                        b[i] = 2/dt - u[j,i]/d + (sig[j,i]+sig[j,i-1])/(2*d**2)
                    elif(idxlc1[i]):
                        b[i] = 2/dt + u[j,i]/d + (sig[j,i+1]+sig[j,i])/(2*d**2)
                        c[i] = -(sig[j,i]+sig[j,i+1])/(2*d**2)
                    elif(idxlc2[i]):
                        b[i] = 2/dt + (sig[j,i+1]+sig[j,i])/(2*d**2)
                        c[i] = (u[j, i+1]/d - (sig[j,i]+sig[j,i+1])/(2*d**2))
                    elif(idxopen1[i]):
                        a[i-1] = (-u[j, i-1]/d - (sig[j,i]+sig[j,i-1])/(2*d**2))
                        b[i] = 2/dt + u[j,i]/d + (sig[j, i+1]+2*sig[j,i]+sig[j,i-1])/2*d**2
                        c[i] = -(sig[j,i]+sig[j,i+1])/(2*d**2)
                    elif(idxopen2[i]):
                        a[i-1] = -(sig[j,i]+sig[j, i-1]) / (2*d**2)
                        b[i] = 2/dt - u[j,i]/d + (sig[j, i+1]+2*sig[j,i]+sig[j,i-1])/2*d**2
                        c[i] = (u[j, i+1]/d - (sig[j,i]+sig[j,i+1])/(2*d**2))
                    else:
                        print('A no bc applies')
                A[j] += np.diag(a,-1)
                A[j] += np.diag(b)
                A[j] += np.diag(c,1)
        
            return A
        
        def compB(sig, N, v, dd):
            land, BC, UC = BCs_fromland(v, direction='y')
            sh = land.shape[0]

            B = np.zeros((land.shape[1], sh, sh))
            g = np.zeros((sh, land.shape[1]))
            dt = particle.dt
            for i in range(v.shape[1]):
                idxrc1 = np.logical_and(v[:,i]>0,
                                        BC[:,i]==1)
                idxrc2 = np.logical_and(v[:,i]<0,
                                        BC[:,i]==1)
                idxlc1 = np.logical_and(v[:,i]>0,
                                        UC[:,i]==1)
                idxlc2 = np.logical_and(v[:,i]<0,
                                        UC[:,i]==1)
                idxopen1 = (v[:, i]>0)
                idxopen2 = (v[:, i]<0)
                d = np.zeros(sh-1)
                e = np.zeros(sh)
                f = np.zeros(sh-1)
                for j in range(v.shape[0]):
                    if(land[j,i]):
                        e[j] = 2/dt
                    elif(idxrc1[j]):
                        d[j-1] = -(v[j+1, i]/dd + (sig[j,i]+sig[j+1,i])/(2*dd**2))
                        e[j] = 2/dt + (sig[j,i]+sig[j+1,i])/(2*dd**2)
                    elif(idxrc2[j]):
                        d[j-1] = -(sig[j,i]+sig[j+1, i]) / (2*dd**2)
                        e[j] = 2/dt - v[j,i]/dd + (sig[j,i]+sig[j+1,i])/(2*dd**2)
                    elif(idxlc1[j]):
                        e[j] = 2/dt + v[j,i]/dd + (sig[j-1,i]+sig[j,i])/(2*dd**2)
                        f[j-1] = -(sig[j,i]+sig[j-1,i])/(2*dd**2)
                    elif(idxlc2[j]):
                        e[j] = 2/dt + (sig[j-1,i]+sig[j,i])/(2*dd**2)
                        f[j-1] = (v[j-1, i]/dd - (sig[j,i]+sig[j-1,i])/(2*dd**2))
                    elif(idxopen1[j]):
                        d[j-1] = (-v[j-1, i]/dd - (sig[j,i]+sig[j-1,i])/(2*dd**2))
                        e[j] = 2/dt + v[j,i]/dd + (sig[j+1, i]+2*sig[j,i]+sig[j,i-1])/2*dd**2
                        f[j] = -(sig[j,i]+sig[j+1,i])/(2*dd**2)
                    elif(idxopen2[j]):
                        d[j-1] = -(sig[j,i]+sig[j-1, i]) / (2*dd**2)
                        e[j] = 2/dt + -v[j,i]/dd + (sig[j+1, i]+2*sig[j,i]+sig[j-1,i])/2*dd**2
                        f[j] = (v[j+1, i]/dd - (sig[j,i]+sig[j+1,i])/(2*dd**2))
                    else:
                        print('B no bc applies')

                B[i] += np.diag(d,-1)
                B[i] += np.diag(e)
                B[i] += np.diag(f,1)
                g[:,i] = np.append(-d*N[:-1,i],0) + (4/dt-e)*N[:,i] - np.append(0,f*N[1:,i])                
            return B, g


        def GaussElim(A,b):
            fac = lu_factor(A)
            res = lu_solve(fac, b)
            return res

        def prepGaussElim(A, b, direction='x'):
            if(direction=='x'):
                N = np.zeros((A.shape[0], A.shape[1]))
                for j in range(A.shape[0]):
                    N[j] = GaussElim(A[j], b[j])
            if(direction=='y'):
                N = np.zeros((A.shape[1], A.shape[0]))
                for i in range(A.shape[0]):
                    N[:,i] = GaussElim(A[i], b[:,i])
            return N

        def calch(N, A):
            dt = particle.dt
            h = np.zeros((A.shape[0], A.shape[1]))
            for j in range(A.shape[0]):
                a = np.diagonal(A[j], 1)
                b = np.diagonal(A[j])
                c = np.diagonal(A[j], -1)
                h[j] = np.append(-a*N[j,:-1],0) + (4/dt - b)*N[j,:] + np.append(0,-c*N[j,1:])            
            return h

        def Adv_2D(T0, nt, dt, dx, dz, Vx, Vz, Lx1, Lx2, Lz1, Lz2, x, z, X, Z):
            sig = np.ones(Vx.shape) * 0 # use zero diffusion

            B, g = compB(sig, T0, Vz, dz)
            assert not (B==0).all(), 'B only consists of zeros'
            A = compA(sig, Vx, dx)
            assert not (A==0).all(), 'A only consists of zeros'
            N = prepGaussElim(A, g)
            h = calch(N, A)
            N = prepGaussElim(B, h, direction='y')
            return N

        def AO(T):
            Tr = T.copy()
            Dx = 0.1
            Dy = 0.1

            Uv = interpolator(fieldset, fieldset.U, tracer=False)
            Vv = interpolator(fieldset, fieldset.V, tracer=False)

            #Uv = set_land(fieldset, Uv, 0)
            #Vv = set_land(fieldset, Vv, 0)
            if(len(fieldset.P.lat[:].shape)==1):
                lons, lats = np.meshgrid(fieldset.P.lon[:], fieldset.P.lat[:])
            else:
                lons, lats = (fieldset.P.lon[:], fieldset.P.lat[:])
            X, Y = (lons, lats)
            Uv /= (1852*60*np.cos(lats*np.pi/180))
            Vv /= (1852*60)
            T = np.clip(Adv_2D(Tr, 1, particle.dt, Dx, Dy, Uv, Vv,
                       lons[0], lons[-1], lats[0], lats[-1],
                       lons, lats, X, Y), 0, 1)
            return T 
        # Advection
        fieldset.P.data[0, 1:-1, 1:-1] = AO(fieldset.P.data[0])[1:-1,1:-1]
        # change domain boundaries


def UpdatePfield(particle, fieldset, time):
    if(particle.id==0):# and (fieldset.Pdiff.data[0]!=0).any()): # only once per time step
        def interpolator(fieldset, field, tracer=True, op2=False, tint=True):
            if(tint):
            # time interpolation of H field
                ti = field.time_index(time)
                tint_field = field.temporal_interpolate_fullfield(ti[0], time)
            else:
                tint_field = field.data

            values = tint_field.flatten()
            gridlon, gridlat = np.meshgrid(field.lon[:], field.lat[:])
            gridlon, gridlat = (gridlon.flatten(), gridlat.flatten())

            grid_x, grid_y = np.meshgrid(fieldset.P.lon,fieldset.P.lat)
            if(op2):
                values[np.where(values<0)] = 0
                #values[np.where(np.isnan(values)==0)] = 0
            elif(True):#tracer): # remove land data before interpolation
                values, gridlon, gridlat = remove_land(fieldset,
                                                       gridlon,
                                                       gridlat,
                                                       values.flatten())


            points = np.swapaxes(np.vstack((gridlat, gridlon)), 0, 1)
            dataI = interpolate.griddata(points, values, (grid_y, grid_x), method='linear')
            return dataI

        def lamb(T, alpha=0.0002, tmax=3.5*10**2*particle.dt):
            res = (math.e**(alpha*T)) / tmax
            return res

        # These functions are age dependent, so will have to be implemented
        # in the kernel that displaces the tuna particle.
        def l(a, linf=87.96, Agemin=3., k=2.564600347374714):
            # skipjack:
            #     k = 2.564600347374714
            #     linf = 87.96 (cm?)
            #     a0 = 3 (years)
            return linf * (1-math.e**(-k * (a-Agemin)))

        def weight(age, p=3, q=0.2):
            # skipjack:
            #     p =
            #     q =
            return q*l(age)**p

        def _sigmaStar(age, sigma_0=2.388505133695817,
                       sigma_K=3.255065398014956,
                       Agemax=11):
            """Return sigmaStar (the termal tolerance intervals, i.e. standard
            deviation) for each cohorts."""
            w = weight(age)
            mw = weight(Agemax)

            return sigma_0 + ((sigma_K - sigma_0)
                              * (w / mw))

        def _tStar(age, T_star_1=31.13068425631411,
                   T_star_K=27.54999999968445,
                   bT=2.564600347374714,
                   Agemax=11) :
            """Return T_star (optimal temperature, i.e. mean) for each cohorts"""
            length = l(age)
            mlength = l(Agemax)
            return T_star_1 - ((T_star_1 - T_star_K)
                               * ((length / mlength)**bT))

        def f4(age, T):
            res = math.e**(-(T-_tStar(age))**2 / (2*_sigmaStar(age)**2))
            return res

        def f5(O2, O2s=3.76, gamma=10**-4):
            # bigeye: O2s=1.49, gamma=0.001
            # skipjack: O2s=3.76, gamma=10**-4
            res =  1. / (1+gamma**(np.array(O2)-O2s))
            return res

        def theta(age, O2, T):
            # oxygen O2 and temperature T should
            # be obtained from their fields.
            return f4(age, T)*f5(O2)

        def scaleP(data):
            phi = 22.5 * np.pi/180.
            a = 0.07
            e = 1. / np.cos(phi)
            b = a * np.sqrt(e*e - 1)

            # coordinate center
            # shift is to have all y>=0
            x0 = 1.0-0.00101482322788
            y0 = 1.0
            # equation for hyperbola
            sinsq = np.sin(phi) * np.sin(phi)
            cossq = 1.0-sinsq
            rasq  = 1.0 / (a*a)
            rbsq  = 1.0 / (b*b)
            A = sinsq*rasq - cossq*rbsq
            B = -2.0 * (data-x0) * np.cos(phi) * np.sin(phi) * (rasq+rbsq)
            C = 1.0 - (data-x0) * (data-x0) * (sinsq*rbsq - cossq*rasq)

            return (y0+(B+np.sqrt(B*B-4.0*A*C))/(2*A))

        # Add P field according to source field
        Ps = interpolator(fieldset, fieldset.epi_mnk_pp, op2=True)
        daytosec = 86400
        Ps /= daytosec
        fieldset.F.data[0] += Ps * particle.dt

        # Reduce F according to the T field
        T = interpolator(fieldset, fieldset.T, tint=False)
        fieldset.F.data[0] -= fieldset.F.data[0] * np.clip(lamb(T), a_min=0, a_max=1) * particle.dt

        # Set land values to 0
        fieldset.F.data[0][np.where(fieldset.Land.data[0].astype(bool))] = 0
         # Update P field according to the changed F field
        O2 = interpolator(fieldset, fieldset.O2, tint=False)
        fieldset.P.data[0] = scaleP(theta(particle.age,
                                           O2.data,
                                           T.data) * fieldset.F.data[0])


        fieldset.F.grid.time[0] = time # updating Field P time
        fieldset.P.grid.time[0] = time # updating Field P time




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
              'LandBlock':LandBlock,
#              'PreyInteraction':PreyInteraction}
              'PreyDepletion':PreyDepletion,
              'PreyAdvectionFournierSibertRestore':PreyAdvectionFournierSibertRestore,
              'PreyAdvectionMICRestore':PreyAdvectionMICRestore,
              'dyingFish':dyingFish,
              'FishingMortalityFADsim':FishingMortalityFADsim,
              'UpdatePfield':UpdatePfield,
              'Faugeras':Faugeras,
              }

AllInteractions = {'Iattraction': Iattraction,
                   'ItunaFAD': ItunaFAD,
                   'Imovement': Imovement,
                   'ImovementFaugeras': ImovementFaugeras}
