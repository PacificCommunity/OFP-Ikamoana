# -*- coding: utf-8 -*-

import os
import numpy as np
import datetime
import calendar


filval = float(1e34)
misval = float(-999)


def isRegular(vector,eps=1e-3):

    step = np.float32(vector[1]-vector[0])
    diff = np.float32(vector[1:] - vector[0:-1])

    
    if np.abs(np.max(diff-step)) < eps:
        return True
    else:
        return False
    
    return

def gen_find(filepat,top):
    import fnmatch
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist,filepat):
            yield os.path.join(path,name)

def date_dym2tostr(dym2date):

    # Dym2date format is a float
    year = int(dym2date)

    if calendar.isleap(year):
        daysinyear = 366
    else:
        daysinyear = 365
    days = int(round(((dym2date-year)*daysinyear)))
   
    try:
        d=datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
        return d.strftime("%Y%m%d")
    except:
        return "xxxxxxxx"

########################################################################
# spatial interpolation
########################################################################

# def spatial_interp(inveclon, inveclat, indata, outveclon, outveclat, mask,gridlib='scipy'):

#     if not (gridlib in ['scipy', 'mlab']):
#         logging.error('unsupported gridlib %s. Expecting "scipy" or "mlab")' %(gridlib))
#         raise RuntimeError
        
#     # output coordinates
#     out_nblon = len(outveclon)
#     out_nblat = len(outveclat)
#     #out_mat_lon = (np.tile(outveclon, (out_nblat, 1))).reshape((out_nblon*out_nblat,1))
#     #out_mat_lat = np.transpose(np.tile(outveclat, (out_nblon,1))).reshape((out_nblon*out_nblat,1))
    
#     # input coordinales
#     vec = copy.copy(inveclon)
#     data = copy.copy(indata)
    
#     # minlon
#     k = 1
#     while (True):
#         if min(outveclon)<min(inveclon):
#             inveclon = np.hstack([vec-k*360, inveclon])
#             indata = np.hstack([data, indata])
#             k += 1
#         else:
#             break
#     # maxlon
#     k = 1
#     while (True):
#         if max(outveclon)>max(inveclon):
#             inveclon = np.hstack([inveclon, vec+k*360])
#             indata = np.hstack([data, indata])
#             k += 1
#         else:
#             break
    
#     ind_min_lon=np.where(inveclon<min(outveclon))
    
#     if np.size(ind_min_lon)==0:
#         ind_min_lon=0
#     else:
#         ind_min_lon=max(ind_min_lon[0])
    
#     ind_max_lon=np.where(inveclon>max(outveclon))

#     if np.size(ind_max_lon)==0:
#        ind_max_lon=len(outveclon)
#     else:
#        ind_max_lon = min(ind_max_lon[0])


#     inveclon=inveclon[ind_min_lon:ind_max_lon+1]
#     indata=indata[:,ind_min_lon:ind_max_lon+1]

#     in_nblon = len(inveclon)    
#     in_nblat = len(inveclat)
#     in_mat_lon = np.tile(inveclon, (in_nblat, 1))
#     in_mat_lat = np.transpose(np.tile(inveclat, (in_nblon, 1)))
    
#     points = np.zeros((in_nblon*in_nblat,2))
#     points[:,0] = in_mat_lon.reshape(in_nblon*in_nblat,1)[:,0]
#     points[:,1] = in_mat_lat.reshape(in_nblon*in_nblat,1)[:,0]
    

#     values = copy.copy(indata)
#     values = values.reshape(in_nblon*in_nblat,1)[:,0]
    
#     ind = np.where((values!=filval) &(values!=misval))[0]
#     points = points[ind, :]
#     values = values[ind, :]

#     # default : griddata from scipy
#     if gridlib == 'scipy':
#         from scipy.interpolate import griddata
#         xi, yi = np.meshgrid(outveclon, outveclat)

#         outdata = griddata((points[:,0], points[:,1]), values, (xi, yi),method='linear')
#     # gridddata from matplotlib.mlab
#     else:
#         from matplotlib.mlab import griddata
#         outdata = griddata(points[:,0], points[:,1], values, outveclon[:],outveclat[:],  interp='nn')

#     outdata = np.float32(outdata.reshape(out_nblat, out_nblon))
#     outdata[mask==0]=filval
#     outdata[np.isnan(outdata)]=filval
    

#     return outdata


# def date_dym2tostr(dym2date):

#     # Dym2date format is a float
#     year = int(dym2date)

#     if calendar.isleap(year):
#         daysinyear = 366
#     else:
#         daysinyear = 365
#     days = int(round(((dym2date-year)*daysinyear)))
   
#     try:
#         d=datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
#         return d.strftime("%Y%m%d")
#     except:
#         return "xxxxxxxx"


# def date_str2dym(datestr):

#     try:
#         date = datetime.datetime.strptime(datestr, '%Y%m%d')
#     except:
#         print('Wrong date format: ' + datestr)
#         raise RuntimeError

#     # Dym2date format is a float
#     year=date.timetuple().tm_year
#     if calendar.isleap(year):
#         daysinyear = 366
#     else:
#         daysinyear = 365
    
#     yday=date.timetuple().tm_yday
    
#     return  np.float32(year)+np.float32(yday)/np.float32(daysinyear)

# # Convert date given in float dym2 format to unix time stamp
# def date_float2time_t(date):
#     fracday,intpart = math.modf(date)
#     year=np.int32(intpart)

#     if calendar.isleap(year):
#         daysinyear = 366
#     else:
#         daysinyear = 365
#     doy = np.round(fracday * daysinyear)

#     if doy == 0:
#         year  = year - 1
#         month = 12
#         mday  = 31
    
#     else:
#         year,month,mday=ymd(year,doy)
        
#     d=datetime.datetime(np.int32(year), np.int32(month), np.int32(mday))
#     # time.mktime() assumes that the passed tuple is in local time, calendar.timegm() assumes it's in GMT/UTC. Depending on the interpretation the tuple represents a different time, so the functions return different values (seconds since the epoch are UTC based).

#     # unixtimestamp = time.mktime(d.timetuple())
#     unixtimestamp = calendar.timegm(d.timetuple())

#     return unixtimestamp
    

