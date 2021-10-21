# -*- coding: utf-8 -*-

# This file is part of dym-python library

__author__ = "O. Titaud"
__date__ = "2014-02-14"

"""
class for NetCDF4 files
"""

import numpy as np
import logging
import netCDF4 as nc
from dateutil.parser import parse
import datetime
import dateutil

from . import dym


class NcFile(object):

    def __init__(self,filename):

        self.fileName_ = filename

        self.latName_    = None # latitude  dimension name in NetCDF file
        self.lonName_    = None # longitude dimensiosn name in NetCDF file
        self.depName_    = None # depth dimension name in NetCDF file if exists
        self.recName_    = None # time (record) dimension name in NetCDF file if exists
        self.nLon_       = None # number of points in longitude axis
        self.nLat_       = None # number of points in latitude  axis
        self.nDep_       = None # number of depth layers
        self.nRec_       = None # number of records
        self.xLon_       = None # longitude values
        self.yLat_       = None # latitude  values
        self.zDep_       = None # depth values
        self.tRec_       = None # record values
        self.originRec_  = None # time origin if specified in NetCDF file
        self.unitsRec_   = None # time units  if specified in NetCDF file
        self.variables_  = None # non-dimensional variables in NetCDF file
        self.dimensions_ = None # dimensions in NetCDF file
        self.globalAtts_ = {}   # global attributes



    def create(self, datestr, veclon, veclat, globalAtts, 
            timeUnits='seconds since 1970-1-1 00:00:00',
            ncFormat='NETCDF4'):

        self.latName_ =  'lat'  
        self.lonName_ =   'lon'
        self.depName_ =   None
        self.recName_ =   'time'
        self.nLon_    =   len(veclon)
        self.nLat_    =   len(veclat)
        self.nDep_    =   0
        self.nRec_    =   0
        self.xLon_    =   veclon
        self.yLat_    =   veclat
        self.zDep_    =   []
        self.tRec_    =   []
        self.globalAtts_ = globalAtts

        # Parsing time unit
        units  = timeUnits.split("since")[0].strip()
        origin = timeUnits.split("since")[1].strip()

        try:
            self.originRec_ = dateutil.parser.parse(origin)
        except: 
            logging.warning("unable to recognize time origin \"%s\" " \
                "in file %s " %(origin,self.fileName_))
            self.originRec_ = None

        if units != "seconds" and units != "days" and units != "hours":
            logging.warning("unable to recognise time units \"%s\" " \
                "in file %s " %(units,self.fileName_))
            self.unitsRec_ = None
        else:
            self.unitsRec_ = units


        fid = nc.Dataset(self.fileName_, 'w', format=ncFormat)

        # global attributes

        for att in globalAtts:
            setattr(fid, att, globalAtts[att])

        # dimensions: latitude, longitude and time

        # latitude
        fid.createDimension('lat', len(veclat))
        dim = fid.createVariable('lat', 'f', ('lat',))
        dim.standard_name='latitude'
        dim.units = 'degrees_north'
        dim[:] = np.float32(veclat)

        # longitude 
        fid.createDimension('lon', len(veclon))
        dim = fid.createVariable('lon', 'f', ('lon',))
        dim.standard_name='longitude'
        dim.units = 'degrees_east'
        dim[:] = np.float32(veclon)

        # time
        fid.createDimension('time', 0)
        dim_time = fid.createVariable('time', 'i', ('time',))
        dim_time.standard_name='time'
        dim_time.units = timeUnits
        t1 = self.originRec_ 
        datestr = datestr + " 12:00:00"
        t2 = dateutil.parser.parse(datestr)
        # original version
        # t1 = datetime.datetime(2007,1,3,12,0,0)
        # t2 = datetime.datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]),12,0,0)
        dim_time[:]=np.int32([(t2-t1).total_seconds()])

        fid.close()

        # update dimensions
        self.getDimensions()

        # update records
        self.getRecords()

        # Check
        # self.dump()


        return 0

    def getDimensions(self):
        """
        Get dimensions in NetCDF fileName_
        """

        fid = nc.Dataset(self.fileName_)

        self.dimensions_ = []

        for key, value in fid.dimensions.items():
            #logging.info("Found dimension %s", key)

            self.dimensions_.append(key)

            if key.lower() in ['lat','latitude','nblatitudes']:
                self.latName_ = key
                self.nLat_    = len(value)
            elif key.lower() in ['lon','longitude','nblongitudes']:
                self.lonName_ = key
                self.nLon_    = len(value)
            elif key.lower() in ['depth']:
                self.depName_ = key
                self.nDep_    = len(value)
            elif key.lower() in ['time','time_counter']:
                self.recName_ = key
                self.nRec_     = len(value)

        fid.close()

        return self.dimensions_


    def getLongitudes(self):
        """
        Get longitude values from file
        """

        if self.lonName_ is None:
            self.getDimensions()

        fid = nc.Dataset(self.fileName_)
        self.xLon_ = fid.variables[self.lonName_][:]
        fid.close()

        return np.float32(self.xLon_)

    def getLatitudes(self):
        """
        Get latitude values from file
        """

        if self.latName_ is None:
            self.getDimensions()

        fid = nc.Dataset(self.fileName_)
        self.yLat_ = fid.variables[self.latName_][:]
        fid.close()

        return self.yLat_

    def getRecords(self):
        """
        Get records values from file
        """
        #print(self.nRec_)
        if self.nRec_ != 0 and self.recName_ is not None:

            fid = nc.Dataset(self.fileName_)
            self.tRec_ = fid.variables[self.recName_][:]
            fid.close()
        else:
            self.tRec_ = None

        return np.float32(self.tRec_)

    def getRecordsUnits(self):

        if self.recName_ is None:
            self.getDimensions()

        if self.recName_ is None:
            return None
        
        fid = nc.Dataset(self.fileName_)

        var_units = fid.variables[self.recName_].getncattr("units")

        units  = var_units.split("since")[0].strip()
        origin = var_units.split("since")[1].strip()

        try:
            self.originRec_ = dateutil.parser.parse(origin)
        except: 
            logging.warning("unable to recognize time origin \"%s\" " \
                "in file %s " %(origin,self.fileName_))
            self.originRec_ = None

        if units != "seconds" and units != "days" and units != "hours":
            logging.warning("unable to recognise time units \"%s\" " \
                "in file %s " %(units,self.fileName_))
            self.unitsRec_ = None
        else:
            self.unitsRec_ = units
            
        rawunits = fid.variables[self.recName_].getncattr("units")

        fid.close()

        return rawunits


    def getDepths(self):

        fid = nc.Dataset(self.fileName_)
        if self.depName_ is not None:
            self.zDep_ = fid.variables[self.depName_][:] 
        else:
            self.zDep_ = None
        fid.close()

        return self.zDep_


    def getVariables(self):

        fid = nc.Dataset(self.fileName_)
        if self.dimensions_ is None:
            self.getDimensions()
        
        self.variables_ = []

        for var in fid.variables:

            if var not in self.dimensions_:

                # logging.info("Found variable %s with dimension %s" \
                #     %(var,fid.variables[var].dimensions))
                self.variables_.append(var)

        fid.close()
        return self.variables_


    def read(self):

        self.getDimensions()
        self.getLongitudes()
        self.getLatitudes()
        self.getDepths()
        self.getRecords()
        self.getRecordsUnits()
        self.getVariables()

        return None

    def dump(self):

        print("%s: %d" % (self.latName_, self.nLat_))
        print("%s: %d" % (self.lonName_, self.nLon_))
        print("%s: %d" % (self.depName_, self.nDep_))
        print("%s: %d" % (self.recName_, self.nRec_))

        return None

    
    def getMaskVariable(self,varName):

        fid = nc.Dataset(self.fileName_)

        if self.variables_ is None:
            self.getVariables()

        if varName not in self.variables_:
            logging.error("%s: unable to get mask of this variable" %varName)
            raise RuntimeError

            #def set_auto_mask(  self,mask)
            #def set_auto_maskandscale(  self,maskandscale)
        
        varDim = fid.variables[varName].dimensions

        # Version that not uses masked arrays
        indata = np.array(fid.variables[varName][:])
        fillVal = fid.variables[varName]._FillValue

        indata[indata!=fillVal] = int(1)
        indata[indata==fillVal] = int(0)
        if self.recName_ in varDim:    
            out_mask = indata[0]
        else:
            out_mask = indata[:]
        
        fid.close()
        return out_mask, varDim

        # # Version using masked arrays
        # indata = np.ma.array(fid.variables[varName][:])
        
        # if self.recName_ in varDim:
        #     #varDim = [ t for t in varDim if t !=self.recName_ ]    
        #     out_mask = np.ma.make_mask(indata.mask[0])
        # else:
        #     out_mask = np.ma.make_mask(indata.mask[:])
        
        # # FillValue = fid.variables[varName]._FillValue
        # # logging.info("FillValue of variable %s: %12e" %(varName,FillValue))
        
        # fid.close()
        # return np.logical_not(out_mask).astype(int), varDim


    def getVarDimensions(self,varName):

        fid    = nc.Dataset(self.fileName_)
        #fid.set_auto_maskandscale = False
        varDim = fid.variables[varName].dimensions
        fid.close()
        
        return varDim

    def getData(self,varName):

        if self.variables_ is None:
            self.getVariables()

        if varName not in self.variables_:
            logging.error("%s: unable to get value of this variable" % varName)
            raise RuntimeError

        fid     = nc.Dataset(self.fileName_)

        fillVal  = fid.variables[varName]._FillValue
        
        # Take scale_factor and add_offset if necessary 
        try:
            factor = np.float32(fid.variables[varName].scale_factor)
        except:
            factor = np.float32(1.0)
        try:
            offset = np.float32(fid.variables[varName].add_offset)
        except:
            offset = np.float32(0.0)
        
        outData  = np.array(fid.variables[varName]) * factor + offset
        
        outData[outData==fillVal]=0

        # Version with masked array
        # outData = np.ma.array(fid.variables[varName]).data[:]
        fid.close()

        return outData


    def convertRecToFloat(self,units=None,origin=None):

        rawunit = self.getRecordsUnits()

        #logging.info("raw time units: {}".format(rawunit))
        
        if units is None:
            units = self.unitsRec_
        if origin is None:
            origin = self.originRec_ 
        
        if origin is None or units is None:

            logging.warning("unable to convert record time value to string")
            return None

        tRec = []

        for rec in self.tRec_:

            if units == "seconds":
                unix = origin + datetime.timedelta(seconds=float(rec))

            elif units == "days":
                unix = origin + datetime.timedelta(days=float(rec))

            elif units == "hours":
                unix = origin + datetime.timedelta(hours=float(rec))

            else:
                logging.warning("%s: time unit not recognized" %units)
                return None

            datestr = datetime.datetime.strftime(unix,"%Y%m%d")

            tRec.append(dym.dates.strToFloat(datestr,dateFormat="%Y%m%d"))

        return tRec

