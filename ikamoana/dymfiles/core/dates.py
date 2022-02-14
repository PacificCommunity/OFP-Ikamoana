# -*- coding: utf-8 -*-

######################################################################
# dym files date handling
######################################################################

__author__ = "O. Titaud"
__date__ = "2015-11-10"

import numpy as np
import datetime
import calendar
import math
from dateutil.parser import parse

"""
Given year = Y and day of year = N, return year, month, day
Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 
"""
def ymd(Y,N):
    if calendar.isleap(Y):
        K = 1
    else:
        K = 2
    M = int((9 * (K + N)) / 275.0 + 0.98)
    if N < 32:
        M = 1
    D = N - int((275 * M) / 9.0) + K * int((M + 9) / 12.0) + 30
    return Y, M, D

def doy(floatDate):
    """
    Return day of year 
    """

    year = int(floatDate)
    
    if calendar.isleap(year):
        daysinyear = 366
    else:
        daysinyear = 365
    d = int(round(((floatDate-year)*daysinyear)))

    return doy


def timeSteps(zlevels):
    """
    Compute time steps resolution in time level vector
    """
    if len(zlevels) == 1 :
        return None, None

    zlevUnix = np.array([ floatToUnix(x) for x in zlevels ])

    steps = np.array( [ datetime.timedelta(seconds=int(x)).days \
                        for x in np.array(zlevUnix[1:] - zlevUnix[:-1])] )

    return steps, np.mean(steps)


"""
Format a float date to string following dateFormat
"""
def floatToStr(floatDate,dateFormat="%Y%m%d"):
    # human format: "%a, %b %d %Y"

    year = int(floatDate)

    if calendar.isleap(year):
        daysinyear = 366
    else:
        daysinyear = 365
    days = int(round(((floatDate-year)*daysinyear)))

    try:
        d=datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
        return d.strftime(dateFormat)
    except:
        return str(floatDate)


"""
Parse string formated date to float
"""
def strToFloat(datestr,dateFormat="%Y%m%d"):

    try:
        date = datetime.datetime.strptime(datestr,dateFormat)
    except:
        logging.error('Wrong date format: ' + datestr)
        raise RuntimeError

    year=date.timetuple().tm_year
    if calendar.isleap(year):
        daysinyear = 366
    else:
        daysinyear = 365
    
    yday=date.timetuple().tm_yday
    
    return  np.float32(year)+np.float32(yday)/np.float32(daysinyear)


"""
Convert a float date to unix time stamp
"""
def floatToUnix(floatDate):

    fracday,intpart = math.modf(floatDate)
    year=np.int32(intpart)

    if calendar.isleap(year):
        daysinyear = 366
    else:
        daysinyear = 365
    doy = np.round(fracday * daysinyear)

    if doy == 0:
        year  = year - 1
        month = 12
        mday  = 31
    
    else:
        year,month,mday=ymd(year,doy)
        
    d=datetime.datetime(np.int32(year), np.int32(month), np.int32(mday))
    # time.mktime() assumes that the passed tuple is in local time, calendar.
    # timegm()      assumes it's in GMT/UTC. 
    # Depending on the interpretation the tuple represents a different time, 
    # so the functions return different values (seconds since the epoch are UTC based).

    # unixtimestamp = time.mktime(d.timetuple())
    unixtimestamp = calendar.timegm(d.timetuple())

    return unixtimestamp


"""
Convert a float date to unix time stamp
"""
def UnixToFloat(unixtimestamp,originStr):

    origin   = parse(originStr)
    unix     = origin + datetime.timedelta(seconds=unixtimestamp)
    datestr  = datetime.datetime.strftime(unix,"%Y%m%d")

    return strToFloat(datestr,dateFormat="%Y%m%d")
    

def findDateStr(zlevels,datestr,dateFormat="%Y%m%d",EPS=1.0/(2*365)):
    """
    Find the index in a levels vector of a given float date 
    :param zlevels: vector of float dates
    :param datestr: string date
    :return index of date corresponding of datestr in zlevel or None if not found
    """

    floatDate = strToFloat(datestr,dateFormat=dateFormat)

    err = np.abs(zlevels-floatDate)

    index = np.where(err < EPS )

    if index[0].size == 0:
        return None

    return index[0][0]

