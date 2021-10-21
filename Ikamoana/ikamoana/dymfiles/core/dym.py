# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import zipfile
import xml.etree.ElementTree as ET

from .dates import *

# This file is part of dym-python library

__author__ = "O. Titaud"
__date__   = "2014-02-14"

"""
classes for generic DYM files 
"""


def getFileFormat(filename):
    """
    Get DYM file format
    :param filename: file name
    :return: file format in ["DYM2","DYM3","DYMZ","UNKNOWN"]
    """
    unknown = "UNKNOWN"
    fo = open(filename, "rb")
    IdFormat = fo.read(4).decode('utf-8')
    
    fo.close()
    
    if IdFormat in ['DYM2','DYM3']:
       return IdFormat
    else:
        try: 
            # Try to unzip file
            zfile=zipfile.ZipFile(filename)
        except:
            logging.warning("%s: not a dym[2,3,Z] file" %(filename))
            return unknown

        file_list=zfile.namelist()
        if len(file_list)!=2:
            logging.warning("%s: expecting 2 files in archive, %d found "\
                                %(filename,len(file_list)) )
            return unknown
        try:
            meta=[a for a in file_list if a[-3:]=='xml'][0]
        except:
            logging.warning("%s: no meta file found in zip archive" %(filename))
            return unknown
        try:
            data=[a for a in file_list if '.dym' in a][0]
        except:
            logging.warning("%s: no data file found in zip archive" %(filename))
            return unknown       
               
        fo = zfile.open(data, "r")
        IdFormat= fo.read(4).decode('utf-8')
        fo.close()
        if  IdFormat != "DYM3":
            logging.warning("%s: data file in dymz archive: unknown format " %(filename))
            return unknown
 
        fo = zfile.open(meta, "r")
        
        try:
            ET.parse(fo)
            fo.close()
        except:
            fo.close()
            logging.warning(sys.exc_info())
            logging.warning("%s: meta file in dymz archive: unknown format " %(filename))
            return unknown

        return "DYMZ"


class DymFileHeader:

    ##########################################################################
    ## header variable initialisation
    ########################################################################## 

    def __init__(self,filename,idformat):
        self.sizeofChar_    = 1
        self.sizeofInt_     = 4
        self.sizeofFloat_   = 4
        self.sizeofLongInt_ = 8

        self.fileName_ = filename
        self.idFormat_ = idformat
        self.nLon_     = np.int32(0)
        self.nLat_     = np.int32(0)
        self.nLev_     = np.int32(0)

    ##########################################################################
    ## Compute header size: useful to jump to data 
    ########################################################################## 

    def headerSize(self):
        return 0

    def dataBlockSize(self):
        return self.nLon_ * self.nLat_ * self.sizeofFloat_

    def dataSize(self):
        return self.nLev_ * self.dataBlockSize()

    def fileSize(self):
        return self.headerSize()+self.dataSize()


class DymFile:

    def __init__(self, filename,dymformat=None):
        """
        Constructor
        :param filename : DYM file name
        :param dymformat: DYM file format (to be set for creating new file)
        """

        self.fileName_ = filename

        if os.path.exists(filename):
            idformat = getFileFormat(filename)

            if dymformat is not  None and dymformat != idformat:
                logging.error("%s: dymformat is not consistent with file format:" %(self.fileName_))
                logging.error("expected format : %s" %dymformat)
                logging.error("found    format : %s" %idformat )
                raise RuntimeError
            self.format_ = idformat
        else:
            if dymformat is None:
                logging.error("no file format selected")
                raise RuntimeError
            else:
                self.format_ = dymformat

        
        self.dataFile_ = None
        self.metaFile_ = None
        self.header_   = DymFileHeader(self.dataFile_,self.format_)

        return

    def headerSize(self):
        return self.header_.headerSize()
        
    def dataBlockSize(self):
        return self.header_.dataBlockSize()

    def dataSize(self):
        return self.header_.dataSize()

    def fileSize(self):
        return self.header_.fileSize()
 
    #########################################################################
    ## Read the header of the data file
    #########################################################################

    def readHeader(self):
        self.header_.readHeader()

    #########################################################################
    ## Read one data matrix
    #########################################################################

    def readData(self, level=1):

        if not os.path.exists(self.dataFile_):
            logging.error("%s: no such file" % self.dataFile_)
            raise IOError

        blocsize = self.dataBlockSize()

        skip  = np.int64(self.headerSize())
        skip += np.int64(blocsize) * ( level - 1)

        fo = open(self.dataFile_, "rb")
        fo.seek(skip)
        data = np.fromfile(fo, "f", blocsize, "")
        data.resize(self.header_.nLat_, self.header_.nLon_)
        fo.close()
        return data

    #########################################################################
    ## Read all data matrix of the DYM file
    #########################################################################
    
    
    def readAllData(self):
    
        data = np.zeros((self.header_.nLev_, self.header_.nLat_,\
                         self.header_.nLon_), np.float32)

        for i in range(0,self.header_.nLev_):
            data[i,:,:]= self.readData(i+1)
            
        return data

    #########################################################################
    ## Write one data matrix at level k
    #########################################################################
    
    def writeData(self,data,level):

        blocsize = self.dataBlockSize()

        skip  = np.int64(self.headerSize())
        skip += np.int64(blocsize) * ( level - 1)

        fo = open(self.dataFile_, "rb+")
        fo.seek(skip,0)
        fo.write(np.float32(data))
        fo.close

    #########################################################################
    ## append data matrix 
    #########################################################################

    def appendData(self,data):

        fo = open(self.dataFile_, "rb+")
        fo.seek(0,2)
        fo.write(np.ascontiguousarray(data,dtype=np.float32))
        fo.close

    #########################################################################
    ## write all data matrix of the DYM file
    #########################################################################

    def writeAllData(self,data):

        skip  = np.int64(self.headerSize())

        fo = open(self.dataFile_, "rb+")
        fo.seek(skip,0)
        fo.write(np.float32(data))
        fo.close

    #########################################################################
    ## Compute mean value over a selected zone 
    #########################################################################

    def mean(self,step=None,domain=None,filename=None,header=None):

        # Get boundary zone indexes

        xlon = self.header_.xLon_[:,0]
        ylat = np.flipud(self.header_.yLat_[0,:])
        if domain is None:
            lonmin = np.min(xlon)
            lonmax = np.max(xlon)
            latmin = np.min(ylat)
            latmax = np.max(ylat)        

        else:
            if len(domain)!=4:
                logging.error("domain specification needs 4 coordinates (lonmin,lonmax,latmin,latmax)")
                raise RuntimeError
            
            lonmin = np.float(domain[0])
            lonmax = np.float(domain[1])
            latmin = np.float(domain[2])
            latmax = np.float(domain[3])
        
        if lonmin >= lonmax or latmin >=latmax:
            logging.error("domain definition is not correct")
            raise RuntimeError

        imin = np.argmin(np.abs(xlon-lonmin))
        imax = np.argmin(np.abs(xlon-lonmax))
        jmin = np.argmin(np.abs(ylat-latmin))
        jmax = np.argmin(np.abs(ylat-latmax))

        if filename is not None:
            f = open(filename,"w")

        if header is not None:
            if len(header) != 3:
                logging.error("header information is not correct")
                raise RuntimeError
        else:
            header=("step","date","mean")
            logging.info("header:  %s\t%s\t%s" % header)
        
        f.write("%s\t%s\t%s\n" % header)

        inmask     = np.flipud(np.transpose(self.header_.mask_))
        mask_layer = inmask[jmin:jmax+1,imin:imax+1]
        mask       = mask_layer
        mask[mask_layer>0]=1
        n = np.sum(mask)
        if n==0:
            logging.warning("no point selected in mask, taking first data!")
            indata = np.flipud(self.readData(1))
            data = indata[jmin:jmax+1,imin:imax+1]         
            mask = data
            mask[data>0]=1
            mask[data==0]=0
            n = np.sum(mask)
            #raise RuntimeError

        

        if step is None:
            for k in range(self.header_.nLev_):
                indata = np.flipud(self.readData(k+1))
                data = indata[jmin:jmax+1,imin:imax+1]
                data[mask==0]=0
                mean = np.sum(data)/n
                line="%3d\t%s\t%f" \
                %(k+1, floatToStr(self.header_.zLev_[k],dateFormat="%d-%m-%Y"), mean)
                if filename is not None:
                    f.write(line+'\n')
                else:
                    print(line)
        else:

            stepmean = 0
            for k in range(self.header_.nLev_):
                indata = np.flipud(self.readData(k+1))
                data = indata[jmin:jmax+1,imin:imax+1]
                data[mask==0]=0
                mean = np.sum(data)/n
                if (int(k+1)%step != 0):
                    stepmean += mean
                else:
                    line="%3d\t%s\t%f" \
                    %(k+1, floatToStr(self.header_.zLev_[k-step/2],\
                        dateFormat="%d-%m-%Y"), stepmean/step)
                    stepmean = 0
                    if filename is not None:
                        f.write(line+'\n')
                    else:
                        print((line))
            
        if filename is not None:
            f.close()
            logging.info("wrote %s" %filename)

        return 0

    #########################################################################
    ## Compute sum value over a selected zone 
    #########################################################################

    def sum(self,step=None,domain=None,filename=None,header=None):

        # Get boundary zone indexes

        xlon = self.header_.xLon_[:,0]
        ylat = np.flipud(self.header_.yLat_[0,:])
        if domain is None:
            lonmin = np.min(xlon)
            lonmax = np.max(xlon)
            latmin = np.min(ylat)
            latmax = np.max(ylat)        

        else:
            if len(domain)!=4:
                logging.error("domain specification needs 4 coordinates (lonmin,lonmax,latmin,latmax)")
                raise RuntimeError
            
            lonmin = np.float(domain[0])
            lonmax = np.float(domain[1])
            latmin = np.float(domain[2])
            latmax = np.float(domain[3])
        
        if lonmin >= lonmax or latmin >=latmax:
            logging.error("domain definition is not correct")
            raise RuntimeError

        imin = np.argmin(np.abs(xlon-lonmin))
        imax = np.argmin(np.abs(xlon-lonmax))
        jmin = np.argmin(np.abs(ylat-latmin))
        jmax = np.argmin(np.abs(ylat-latmax))

        if filename is not None:
            f = open(filename,"w")

        if header is not None:
            if len(header) != 3:
                logging.error("header information is not correct")
                raise RuntimeError
        else:
            header=("step","date","sum")
            logging.info("header:  %s\t%s\t%s" % header)
            
        f.write("%s\t%s\t%s\n" % header )

        inmask     = np.flipud(np.transpose(self.header_.mask_))
        mask_layer = inmask[jmin:jmax+1,imin:imax+1]
        mask       = mask_layer
        mask[mask_layer>0]=1

        if step is None:
            for k in range(self.header_.nLev_):
                indata = np.flipud(self.readData(k+1))
                data = indata[jmin:jmax+1,imin:imax+1]
                data[mask==0]=0
                sum = np.sum(data)
                line="%3d\t%s\t%f" \
                    %(k+1, floatToStr(self.header_.zLev_[k],dateFormat="%d-%m-%Y"),sum)
                if filename is not None:
                    f.write(line+'\n')
                else:
                    print(line)
        else:
            stepsum = 0
            for k in range(self.header_.nLev_):
                indata = np.flipud(self.readData(k+1))
                data = indata[jmin:jmax+1,imin:imax+1]
                data[mask==0]=0
                sum = np.sum(data)
                if (int(k+1)%step != 0):
                    stepsum += sum
                else:
                    line="%3d\t%s\t%f" \
                    %(k+1, floatToStr(self.header_.zLev_[k-step/2],dateFormat="%d-%m-%Y"), stepsum)
                    stepsum = 0
                    if filename is not None:
                        f.write(line+'\n')
                    else:
                        print(line)
            
        if filename is not None:
            f.close()
            logging.info("wrote %s" %filename)

        return 0

    def getTimeSeries(self,statOp,startStr=None, endStr=None,dateFormat="%Y%m%d"):
        """
        Get time series statistics
        statOp: statistical operator (e.g. np.nanmean)
        startStr: first date to compute 
        endStr: first date to compute 
        dateFormat: date format
        return dates, date in string format and statistics time series as arrays
        """
        from dateutil import parser
        
        nLev = self.header_.nLev_
        zLev = self.header_.zLev_
        mask = np.transpose(self.header_.mask_)
        values  = []
        dates   = []
        datestr = []
        startIndex = 0
        endIndex   = nLev - 1

        if startStr is not None:
            startFloat = strToFloat(startStr,dateFormat=dateFormat)
            diff = np.abs(np.array(zLev) - startFloat)
            idx = np.where(diff<1e-6)
            if len(idx[0])!=1:
                startIndex = idx[0][0]
            else:
                startIndex = 0

        if endStr is not None:
            endFloat = strToFloat(endStr,dateFormat=dateFormat)
            diff = np.abs(np.array(zLev) - endFloat)
            idx = np.where(diff<1e-6)
            if len(idx[0])!=1:
                endIndex = nLev - 1
            else:
                endIndex   = idx[0][0]
        
            
        for k in range(startIndex, endIndex+1):

            data          = self.readData(k+1)
            data[mask==0] = np.nan
            data[data<0] = np.nan
            
            theValue = statOp(data)
            theDate  = parser.parse(floatToStr(zLev[k],dateFormat="%Y-%m-%d"))
            theDateStr = floatToStr(zLev[k],dateFormat="%Y-%m-%d")
            values.append(theValue)
            dates.append(theDate)
            datestr.append(theDateStr)

        return dates, datestr, values
