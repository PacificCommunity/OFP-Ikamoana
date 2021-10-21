# -*- coding: utf-8 -*-

import numpy as np
import copy


########################################################################
# spatial interpolation
########################################################################

def spatial(inveclon, inveclat, indata, outveclon, outveclat, mask, gridlib='scipy',\
            fill_value=np.float32(1e34),mis_value=np.float32(-999) ):

    if not (gridlib in ['scipy', 'mlab']):
        logging.error('unsupported gridlib %s. Expecting "scipy" or "mlab")' %(gridlib))
        raise RuntimeError
        
    # output coordinates
    out_nblon = len(outveclon)
    out_nblat = len(outveclat)
    #out_mat_lon = (np.tile(outveclon, (out_nblat, 1))).reshape((out_nblon*out_nblat,1))
    #out_mat_lat = np.transpose(np.tile(outveclat, (out_nblon,1))).reshape((out_nblon*out_nblat,1))

    # input coordinales
    vec  = copy.copy(inveclon)
    data = copy.copy(indata)

    # minlon
    k = 1
    while (True):
        if min(outveclon)<min(inveclon):
            inveclon = np.hstack([vec-k*360, inveclon])
            indata = np.hstack([data, indata])
            k += 1
        else:
            break
    # maxlon
    k = 1
    while (True):
        if max(outveclon)>max(inveclon):
            inveclon = np.hstack([inveclon, vec+k*360])
            indata = np.hstack([data, indata])
            k += 1
        else:
            break

    ind_min_lon=np.where(inveclon<min(outveclon))

    if np.size(ind_min_lon) == 0:
        ind_min_lon = 0
    else:
        ind_min_lon = max(ind_min_lon[0])

    ind_max_lon = np.where(inveclon>max(outveclon))

    if np.size(ind_max_lon) == 0:
        ind_max_lon = len(outveclon)
    else:
        ind_max_lon = min(ind_max_lon[0])

    inveclon = inveclon[ind_min_lon:ind_max_lon+1]
    indata   = indata[:,ind_min_lon:ind_max_lon+1]

    in_nblon   = len(inveclon)    
    in_nblat   = len(inveclat)
    in_mat_lon = np.tile(inveclon, (in_nblat, 1))
    in_mat_lat = np.transpose(np.tile(inveclat, (in_nblon, 1)))

    points = np.zeros((in_nblon*in_nblat,2))
    points[:,0] = in_mat_lon.reshape(in_nblon*in_nblat,1)[:,0]
    points[:,1] = in_mat_lat.reshape(in_nblon*in_nblat,1)[:,0]


    values = copy.copy(indata)
    values = values.reshape(in_nblon*in_nblat,1)[:,0]

    ind = np.where((values!=fill_value) & (values!=mis_value))[0]
    # print(ind)
    # print(np.shape(values))
    # print(np.shape(points))
    points = points[ind, :]
    values = values[ind]

    # default : griddata from scipy
    if gridlib == 'scipy':
        from scipy.interpolate import griddata
        xi, yi = np.meshgrid(outveclon, outveclat)

        outdata = griddata((points[:,0], points[:,1]), values, (xi, yi),method='linear')
    # gridddata from matplotlib.mlab
    else:
        from matplotlib.mlab import griddata
        outdata = griddata(points[:,0], points[:,1], values, outveclon[:],outveclat[:],  interp='nn')

    outdata = np.float32(outdata.reshape(out_nblat, out_nblon))
    outdata[mask==0] = fill_value
    outdata[np.isnan(outdata)]=fill_value


    return outdata