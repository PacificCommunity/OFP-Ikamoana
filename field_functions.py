import numpy as np

def calcK(H, sigma_scale, sigmaD, c, Pexp=3, Kscaler=1, Kbooster=1):
    return sigma_scale * sigmaD * (1 - c * np.power(H, Pexp)) * Kscaler + Kbooster

def getMaxDiffusion(age_class, length_classes, timestep, units='km_per_timestep'):
    if units == 'nm_per_timestep':
        Dmax = (np.power(GetLengthFromAge(age_class, length_classes)*((timestep)/1852), 2) / 4 ) * timestep/(timestep) #vmax = L for diffusion
    else:
        Dmax = (np.power(GetLengthFromAge(age_class, length_classes), 2) / 4) * timestep

def GetLengthFromAge(age, lengths=[3.00, 4.51, 6.02, 11.65, 16.91, 21.83, 26.43, 30.72, 34.73, 38.49, 41.99, 45.27,
                                  48.33, 51.19, 53.86, 56.36, 58.70, 60.88, 62.92, 64.83, 66.61, 68.27, 69.83, 71.28,
                                  72.64, 73.91, 75.10, 76.21, 77.25, 78.22, 79.12, 79.97, 80.76, 81.50, 82.19, 82.83,
                                  83.44, 84.00, 84.53, 85.02, 85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96,
                                  88.23, 88.48, 88.71, 88.93, 89.14, 89.33, 89.51, 89.67, 89.83, 89.97, 90.11, 90.24,
                                  90.36, 90.47, 90.57, 90.67, 91.16]):
    age -= 1
    if age >= len(lengths):
        age = len(lengths)-1
    return lengths[age]/100 # Convert to meters
