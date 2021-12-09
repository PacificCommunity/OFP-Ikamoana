# -*- coding: utf-8 -*-

# This file is part of dym-python library

__author__ = "O. Titaud"
__date__ = "2014-02-14"

"""
classes for DYM2 files 
"""

import os
import sys
import numpy as np
import logging
import calendar
#import netCDF4 as nc
import copy

from . import interpolation, netcdf, dym

#########################################################################
## Header class
#########################################################################

class DymFileHeader(dym.DymFileHeader):

    # header variable initialisation
    def __init__(self,filename):

        dym.DymFileHeader.__init__(self,filename,"DYM2")

        self.idFunction_ = np.int32(0)
        self.minVal_     = np.float32(0)
        self.maxVal_     = np.float32(0)
        self.firstDate_  = np.float32(0)
        self.lastDate_   = np.float32(0)
        self.xLon_ = []
        self.yLat_ = []
        self.zLev_ = []
        self.mask_ = []

        return
    
    def headerSize(self):

        header_size  =  4 * self.sizeofChar_
        header_size +=  4 * self.sizeofInt_
        header_size +=  4 * self.sizeofFloat_
        header_size +=  2 * self.nLon_ * self.nLat_ * self.sizeofFloat_ # xLon, yLat (float)
        header_size +=  1 * self.nLon_ * self.nLat_ * self.sizeofInt_   # mask (int)
        header_size +=  self.nLev_ * self.sizeofFloat_                  # zLev (float)

        return header_size

    def readHeader(self):

        try:
            fo = open(self.fileName_, "rb")
            self.idFormat_   = fo.read(4).decode('utf-8')
            self.idFunction_ = np.fromfile(fo, "i", 1,"")[0]
            self.minVal_     = np.fromfile(fo, "f", 1, "")[0]
            self.maxVal_     = np.fromfile(fo, "f", 1, "")[0]
            self.nLon_       = np.fromfile(fo, "i", 1, "")[0]
            self.nLat_       = np.fromfile(fo, "i", 1, "")[0]
            self.nLev_       = np.fromfile(fo, "i", 1, "")[0]
            self.firstDate_  = np.fromfile(fo, "f", 1, "")[0]
            self.lastDate_   = np.fromfile(fo, "f", 1, "")[0]

            blocsize = self.nLon_ * self.nLat_

            ## Xlon
            data = np.fromfile(fo, "f", blocsize, "")
            data.resize(self.nLat_, self.nLon_)
            self.xLon_ = np.transpose(data)

            ## Ylat
            data = np.fromfile(fo, "f", blocsize, "")
            data.resize(self.nLat_, self.nLon_)
            self.yLat_ = np.transpose(data)

            self.zLev_ = np.fromfile(fo, "f",self.nLev_, "" )

            data = (np.fromfile(fo, "i", blocsize, ""))
            data.resize(self.nLat_, self.nLon_)
            self.mask_ = np.transpose(data)
            fo.close()

        except IOError as e:
            logging.error("readHeader fails to read %s" %(e.fileName_))
            logging.error(sys.exc_info())
            raise RuntimeError
        except:
            logging.error("Header fails")
            logging.error(sys.exc_info())
            raise RuntimeError

        return

    def writeHeader(self):

        try:
            fo = open(self.fileName_, 'wb+')

            fo.write(self.idFormat_.encode('utf-8'))
            fo.write(np.float32(self.idFunction_))
            fo.write(np.float32(self.minVal_))
            fo.write(np.float32(self.maxVal_))
            fo.write(np.int32(self.nLon_))
            fo.write(np.int32(self.nLat_))
            fo.write(np.int32(self.nLev_))
            fo.write(np.float32(self.firstDate_))
            fo.write(np.float32(self.lastDate_))
            
            X = np.float32(self.xLon_)
            X = np.transpose(X)
            
            try:
                X.tofile(fo)
            except:
                data = np.array(X.filled(0).data)
                data.tofile(fo)
            Y = np.float32(self.yLat_)
            Y = np.transpose(Y)
            try:
                Y.tofile(fo)
            except:
                data = np.array(Y.filled(0).data)
                data.tofile(fo)   
            fo.write(np.float32(self.zLev_))
            M = np.int32(self.mask_)
            M = np.transpose(M)
            try:
                M.tofile(fo)
            except:
                data = np.array(M.filled(0).data)
                data.tofile(fo) 
            fo.close()

        except IOError as e:
            logging.error("writeHeader fails to write file %s" %(self.fileName_))
            logging.error(sys.exc_info())
            raise RuntimeError

        except Exception as e:
            logging.error("writeHeader fails")
            #logging.error(e.strerror)
            logging.error(sys.exc_info())
            raise RuntimeError

        return

    def updateMinMaxValues(self, minVal, maxVal):

        fo = open(self.fileName_, 'rb+')
        fo.seek(2*4)
        fo.write(np.float32(minVal))
        fo.write(np.float32(maxVal))
        fo.close()

        return

    def updateMask(self, newmask):

        try:
            newshape=np.shape(newmask)
            self.readHeader()
            oldshape = (self.nLon_,self.nLat_)
            if newshape != oldshape:
                raise IndexError
            fo = open(filename, 'rb+')
            skip = (9+2*self.nLon_*self.nLat_+self.nLev_)*4
            fo.seek(skip)
            fo.write(np.int32(newmask))
            fo.close()
        except IndexError:
            logging.error("in UpdateMask: input mask size mismatch")
            logging.error("old shape : %d %d" %(oldshape))
            logging.error("new shape : %d %d" %(newshape))
            raise RuntimeError

        return

    def updateLevelValues (self, levelValues):
        try:
            self.readHeader()
            oldnlev = self.nLev_
            newnlev = len(levelValues)
            if oldnlev != newnlev:
                raise IndexError
            firstdate = levelValues[0]
            lastdate = levelValues[-1]
            skip1 = 7*4
            skip2 = (9+2*self.nLon_*self.nLat_)*4
            fo = open(self.fileName_, 'rb+')
            fo.seek(skip1)
            fo.write(np.float32(firstdate))
            fo.write(np.float32(lastdate))
            fo.seek(skip2)
            fo.write(np.float32(levelValues))
            fo.close()
        except IndexError:
            logging.error("new levels values size mismatch")
            raise RuntimeError

        return

    def write_mask(self,outfile):

        fo = open(outfile, 'w')
       
        for j in range(self.nLat_):
            for i in range(self.nLon_):
               fo.write("%2d" %self.mask_[i][j])
            fo.write("\n")
        fo.close()

        logging.info("wrote %s" %outfile)

        return 

    def dump(self,show_levs=False):

        if self.idFunction_ == 0:
            idfunction_str = "Regular grid"
        else:
            idfunction_str = "Non regular grid"

        print('\ndata file: ' + os.path.abspath(self.fileName_) +'\n')
        print('\tIdFormat   : %s' %self.idFormat_)
        print("\tIdFunc     : %d (%s)" %(self.idFunction_,idfunction_str))
        print('\tMIN_VAL    : %.9e' %self.minVal_)
        print('\tMAX_VAL    : %.9e' %self.maxVal_)
        print('\tNLONG      : %d' %self.nLon_)
        print('\tNLAT       : %d' %self.nLat_)
        print('\tNLEVEL     : %d' %self.nLev_)
        print('\tSTART DATE : %.6f %s' % (self.firstDate_,dym.dates.floatToStr(self.firstDate_)))
        print('\tEND DATE   : %.6f %s' % (self.lastDate_,dym.dates.floatToStr(self.lastDate_)))
        print('\t=================================')
        nlat = self.nLat_
        nlon = self.nLon_
        latmax = self.yLat_[0,0]
        lonmin = self.xLon_[0,0]
        latmin = self.yLat_[0,nlat-1]
        lonmax = self.xLon_[nlon-1,0]
        zlevels = self.zLev_

        resx = self.xLon_[1,0] - self.xLon_[0,0]
        resy = self.yLat_[0,0] - self.yLat_[0,1]


        print('\tLatMin     : %+.6f' % latmin)
        print('\tLatMax     : %+.6f' % latmax)
        print('\tLonMin     : %+.6f' % lonmin)
        print('\tLonMax     : %+.6f' % lonmax)
        print('\tRes Y      : %.6f ' % resy)
        print('\tRes X      : %.6f ' % resx)

        try:
            year=int(dym.dates.floatToStr(zlevels[0])[0:4])
        
            if calendar.isleap(year):
                    daysinyear = 366
            else:
                    daysinyear = 365

            resz = (zlevels[1]-zlevels[0]) * daysinyear

            print('\tRes T      : %d day(s)' % int(resz+0.5))
        except:
            print('\tRes T      : xxxx' )

        # Dates
        if show_levs:
                print('\t=================================')
                print('\tDates (k,dym2,datestr,date)')
                for k in range(self.nLev_):
                        print("\t%03d : %f : %s : %s" 
                        %(k+1,zlevels[k],
                            dym.dates.floatToStr(zlevels[k]),\
                            dym.dates.floatToStr(zlevels[k],dateFormat="%a, %b %d %Y")))
  
        print()


#########################################################################
## DymFile class
#########################################################################


class DymFile(dym.DymFile):

    def __init__(self, filename):

        # call parent class constructor
        dym.DymFile.__init__(self,filename,"DYM2")

        self.dataFile_ = self.fileName_ 
 
        self.header_ = DymFileHeader(self.dataFile_)

        if os.path.isfile(self.dataFile_):
            self.readHeader()

    def readHeader(self):
        self.header_.readHeader()

    def headerSize(self):
        return self.header_.headerSize()

    def writeHeader(self):
        self.header_.writeHeader()

    def getTimeStep(self):

        return dym.dates.timeSteps(self.header_.zLev_)

    def dump(self,show_levs=False,show_minmax=False):   
  
        self.header_.dump(show_levs)

        # Compute and show actual min and max values
        if show_minmax:
            data = self.readAllData()
            minval=np.min(data[:,np.transpose(self.header_.mask_)!=0])
            maxval=np.max(data[:,np.transpose(self.header_.mask_)!=0])

            print('\tActual MIN_VAL  : %f ' % minval)
            print('\tActual MAX_VAL  : %f ' % maxval)

    
    def minMaskedData(self,data):
        """
        compute minimum value of masked data 
        :param data: numpy array
        return minimum value of masked data or np.nan is shapes mismatch
        """
        if np.shape(data) != np.shape(np.transpose(self.header_.mask_)):
            logging.warning("shape mismatch")
            return np.nan

        return np.min(data[np.transpose(self.header_.mask_)!=0])

    def maxMaskedData(self,data):
        """
        compute maximum value of masked data 
        :param data: numpy array
        return maximum value of masked data or np.nan is shapes mismatch
        """
        if np.shape(data) != np.shape(np.transpose(self.header_.mask_)):
            logging.warning("shape mismatch")
            return np.nan
        
        return np.max(data[np.transpose(self.header_.mask_)!=0])

    ###############################################################################################
    # SCALE                                                                                       #
    ###############################################################################################

    def scale(self,factor,outFile):

        if os.path.exists(outFile):
            os.remove(outFile)

        # Writing Header
        outDym = DymFile(outFile)
    
        outDym.header_.minVal_    = self.header_.minVal_ * factor
        outDym.header_.maxVal_    = self.header_.maxVal_ * factor
        outDym.header_.firstDate_ = self.header_.firstDate_
        outDym.header_.lastDate_  = self.header_.lastDate_
        outDym.header_.zLev_      = self.header_.zLev_
        outDym.header_.nLev_      = self.header_.nLev_
        outDym.header_.nLon_      = self.header_.nLon_
        outDym.header_.nLat_      = self.header_.nLat_
        outDym.header_.mask_      = self.header_.mask_
        outDym.header_.xLon_      = self.header_.xLon_
        outDym.header_.yLat_      = self.header_.yLat_

        nlon = outDym.header_.nLon_
        nlat = outDym.header_.nLat_

        outDym.header_.writeHeader()
     
        fid_out = open(outDym.fileName_, "rb+")
     
        for t in range(outDym.header_.nLev_+1):
      
            data = np.float32(self.readData(t+1))
            data = factor * data
            fid_out.seek(0,2)
            fid_out.write(data)

            datamin = outDym.minMaskedData(data)
            datamax = outDym.maxMaskedData(data)
            
            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                if datamin < minval :
                    minval = datamin

        fid_out.close()
        
        outDym.header_.updateMinMaxValues(minval, maxval)

        logging.info("wrote %s" %outFile)

        return outDym
        

    def unary_op(self,operator,outFile):

        if os.path.exists(outFile):
            os.remove(outFile)

        # Writing Header
        outDym = DymFile(outFile)
    
        outDym.header_.minVal_    = self.header_.minVal_ * factor
        outDym.header_.maxVal_    = self.header_.maxVal_ * factor
        outDym.header_.firstDate_ = self.header_.firstDate_
        outDym.header_.lastDate_  = self.header_.lastDate_
        outDym.header_.zLev_      = self.header_.zLev_
        outDym.header_.nLev_      = self.header_.nLev_
        outDym.header_.nLon_      = self.header_.nLon_
        outDym.header_.nLat_      = self.header_.nLat_
        outDym.header_.mask_      = self.header_.mask_
        outDym.header_.xLon_      = self.header_.xLon_
        outDym.header_.yLat_      = self.header_.yLat_

        nlon = outDym.header_.nLon_
        nlat = outDym.header_.nLat_

        outDym.header_.writeHeader()
     
        fid_out = open(outDym.fileName_, "rb+")
     
        for t in range(outDym.header_.nLev_+1):
      
            indata = np.float32(self.readData(t+1))
            data   = operator(indata)
            fid_out.seek(0,2)
            fid_out.write(data)

            datamin = outDym.minMaskedData(data)
            datamax = outDym.maxMaskedData(data)
            
            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                if datamin < minval :
                    minval = datamin

        fid_out.close()
        
        outDym.header_.updateMinMaxValues(minval, maxval)

        logging.info("wrote %s" %outFile)

        return outDym


    def binary_op(self,operator,inFile,outFile,force=True):

        if os.path.exists(inFile) == False:
            logging.error("%s: no such file" % inFile)
            raise IOError

        if os.path.exists(outFile):
            os.remove(outFile)
        
        inDym = DymFile(inFile)

        # Format should be the same
        if self.header_.idFormat_ != inDym.header_.idFormat_:
            logging.error("input files format mismatch")
            raise RuntimeError

        # Header information should be the same
        if self.header_.nLon_ != inDym.header_.nLon_ and \
           self.header_.nLat_ != inDym.header_.nLat_ and \
           self.header_.nLev_ != inDym.header_.nLev_:
            logging.error("dimensions mismatch")
            raise RuntimeError

        # Check header difference

        if np.sum(self.header_.xLon_-inDym.header_.xLon_) >= 1e-6:
            logging.warning("longitudes mismatch")
            if force is False:
                raise RuntimeError  
        if np.sum(self.header_.yLat_-inDym.header_.yLat_) >= 1e-6:
            logging.warning("latitudes mismatch")
            if force is False:
                raise RuntimeError
        if np.max(np.abs(self.header_.zLev_-inDym.header_.zLev_)) > 1.0/365:
            logging.warning("input file levels mismatch")
            logging.warning("error = %f" % np.sum(self.header_.zLev_-inDym.header_.zLev_))
            print(inDym.header_.zLev_)
            print(self.header_.zLev_)
            print(inDym.header_.zLev_ - self.header_.zLev_ )
            if force is False:
                raise RuntimeError
        if np.sum(self.header_.mask_-inDym.header_.mask_) != 0:
            logging.error("masks mismatch")
            np.savetxt("mask1.txt",self.header_.mask_,"%2d"," ","\n")
            logging.info("wrote mask1.txt")
            np.savetxt("mask2.txt",inDym.header_.mask_,"%2d"," ","\n")
            logging.info("wrote mask2.txt")
            logging.info("taking first file mask into account")
            np.savetxt("maskdiff.txt",self.header_.mask_-inDym.header_.mask_,"%2d"," ","\n")
            if force is False:
                raise RuntimeError

        outDym = DymFile(outFile)
         
        outDym.header_.idFunction_   = self.header_.idFunction_
        outDym.header_.minVal_       = np.float32(0)
        outDym.header_.maxVal_       = np.float32(1)
        outDym.header_.nLon_         = self.header_.nLon_
        outDym.header_.nLat_         = self.header_.nLat_
        outDym.header_.nLev_         = self.header_.nLev_
        outDym.header_.firstDate_    = self.header_.firstDate_
        outDym.header_.lastDate_     = self.header_.lastDate_  
        outDym.header_.xLon_         = self.header_.xLon_
        outDym.header_.yLat_         = self.header_.yLat_
        outDym.header_.zLev_         = self.header_.zLev_
        outDym.header_.mask_         = self.header_.mask_        

        mask  = np.transpose(outDym.header_.mask_)
        data1 = self.readAllData()
        data2 = inDym.readAllData()

        data = operator(data1,data2)

        data[:,mask==0]=0
        minval = np.min(data)
        maxval = np.max(data)

        outDym.header_.minVal_ = minval
        outDym.header_.maxVal_ = maxval

        outDym.writeHeader()

        outDym.writeAllData(data)

        logging.info("wrote %s" %(outFile))

        return outDym


    def multiple_op(self,operator,fileList,outFile,force=True):

        zlevels = self.header_.zLev_
        nlevels = len(zlevels)

        outDym = DymFile(outFile)
        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      
        outDym.header_.nLev_      = nlevels
        outDym.header_.firstDate_ = np.float32(zlevels[0])
        outDym.header_.lastDate_  = np.float32(zlevels[-1])
        outDym.header_.zLev_      = zlevels
    
        outDym.header_.writeHeader()

        # First check spatial dimension consistency

        
        for fileToProcess in fileList:

            logging.info("checking spatial and time dimensions consistency with %s" %fileToProcess)

            inDym2 = DymFile(fileToProcess)
            inDym2.readHeader()
        
            error=False

            if ( self.header_.nLon_ != inDym2.header_.nLon_ ):
                logging.error("input files has different longitude number")
                error=True

            if ( self.header_.nLat_ != inDym2.header_.nLat_ ):
                logging.error("input files has different latitude number")
                error=True

            if np.sum(self.header_.mask_ - inDym2.header_.mask_) != 0:
                logging.warning("input files has different mask")
                logging.warning("taking first file mask into account")
                #error=True
        
            if np.sum(self.header_.xLon_ - inDym2.header_.xLon_) != 0:
                logging.error("input files has different longitudes")
                error=True
        
            if np.sum(self.header_.yLat_ - inDym2.header_.yLat_) != 0:
                logging.error("input files has different latitude")
                error=True

            # Check levels consistency
            zlevels2 = inDym2.header_.zLev_
            nlevels2 = len(zlevels2)

            if nlevels != nlevels2:
                logging.error("input files has different number of records")
                error=True
        
            else:
                for k in range(nlevels):

                    if zlevels[k]!=zlevels2[k]:
                        logging.error("input files has different record dates")
                        error=True

            if error:
                raise RuntimeError
        
        # apply operator to each grid point of two files

        fid_out = open(outDym.fileName_, "rb+")    
            
        for t in range(self.header_.nLev_):

            #mask  = np.transpose(outDym.header_.mask_)
            data = np.float32(self.readData(t+1))

            datamin = self.minMaskedData(data)
            datamax = self.maxMaskedData(data)


            for fileToProcess in fileList:

                inDym = DymFile(fileToProcess)
                data2 = np.float32(inDym.readData(t+1))

                data  = operator(data,data2)


            fid_out.seek(0,2)
            fid_out.write(data)

            minval = np.min(data)
            maxval = np.max(data)

            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                        
                if datamin < minval :
                    minval = datamin  

        outDym.header_.minVal_ = minval
        outDym.header_.maxVal_ = maxval

        outDym.header_.updateMinMaxValues(minval, maxval)    
            
        fid_out.close()
        
        logging.info("wrote %s" % outFile)

        return outDym

    #########################################################################
    ## Extract some levels in file and write it in another file
    #########################################################################

    def extractLevelsIndex(self,lmin,lmax,outfile):
        """
        Extract time levels in file 
        :param lmin: lower index 
        :param max:  upper index 
        :param outfile: file name to write extracted data 
        :return: outfile name
        """

        # Prevent opening a bad dym file
        if os.path.exists(outfile):
            os.remove(outfile)

        outDym = DymFile(outfile)

        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      

        lev  = self.header_.zLev_[lmin:lmax+1]
        nlev = len(lev)

        if nlev == 0:
            logging.error("Unable to extract levels. Checks index range")
            raise RuntimeError
        
        self.header_.zLev_ 
        outDym.header_.nLev_      = nlev
        outDym.header_.firstDate_ = np.float32(lev[0])
        outDym.header_.lastDate_  = np.float32(lev[nlev-1])
        outDym.header_.zLev_      = lev
        
        outDym.header_.writeHeader()
        
        fid_out = open(outDym.fileName_, "rb+")
        
        for t in range(nlev):
            in_data = np.float32(self.readData(t+lmin+1))
            fid_out.seek(0,2)
            fid_out.write(in_data)
            
            if t == 0:
                minval = np.min(in_data[np.transpose(self.header_.mask_)!=0])
                maxval = np.max(in_data[np.transpose(self.header_.mask_)!=0])
            else:
                if np.max(in_data[np.transpose(self.header_.mask_)!=0])> maxval :
                    maxval = np.max(in_data[np.transpose(self.header_.mask_)!=0])
                    
                if np.min(in_data[np.transpose(self.header_.mask_)!=0]) < minval :
                    minval = np.min(in_data[np.transpose(self.header_.mask_)!=0])

        outDym.header_.updateMinMaxValues(minval, maxval)

        fid_out.close()
        logging.info("Wrote %s" %(outfile))

        return outDym


    def extractLevelsDateStr(self,firstDateStr,lastDateStr,outfile,dateFormat="%Y%m%d"):
        """
        Extract time levels in file 
        :param firstDateStr: lower date bound to extract (string expected)
        :param endDateStr:   upper date bound to extract (string expected)
        :param outfile: file name to write extracted data 
        :param format: date string format
        :return: outfile name
        """

        # Read levels in file
        
        zlevels = self.header_.zLev_
        


        minidx  = dym.dates.findDateStr(zlevels,firstDateStr,dateFormat=dateFormat)
        maxidx  = dym.dates.findDateStr(zlevels,lastDateStr,dateFormat=dateFormat)

        if minidx is not None and maxidx is not None:
            logging.info("START DATE (index): %.6f %s (%d)" \
                        % (zlevels[minidx],\
                            dym.dates.floatToStr(zlevels[minidx],dateFormat=dateFormat),minidx+1) )
            logging.info("END   DATE (index): %.6f %s (%d)" \
                        % (zlevels[maxidx],\
                            dym.dates.floatToStr(zlevels[maxidx],dateFormat=dateFormat),maxidx+1) )
        else:
            logging.error("Index range is not correct. Found indexes are %s %s" %(str(minidx),str(maxidx)))
            raise RuntimeError
        
        return self.extractLevelsIndex(minidx,maxidx,outfile)


    def extractSubdomain(self,latmin,latmax,lonmin,lonmax,outfile):
        """
        Extract subdomain from file
        :param latmin: lower latitude boundary of subdomain
        :param latmax: upper latitude boundary of subdomain
        :param lonmin: lower longitude boundary of subdomain
        :param lonmax: upper longitude boundary of subdomain
        :param outfile: file name of extrated file
        :return dymFile object corresponding of extracted data (wrote in output)
        """

        # Prevent opening a bad dym file
        if os.path.exists(outfile):
            os.remove(outfile)

        xLonMin = np.min(self.header_.xLon_)
        if lonmin < xLonMin:
            # logging.warning("{} is less than minimum longitude in file ({}). Adding 360".format(lonmin,xLonMin))
            # lonmin = lonmin + 360
            logging.error("{} is less than minimum longitude in file ({}). Consider adding 360".format(lonmin,xLonMin))
            raise RuntimeError
            
        if lonmax < xLonMin:
            # logging.warning("{} is less than minimum longitude in file ({}). Adding 360".format(lonmax,xLonMin))
            # lonmax = lonmax + 360
            logging.error("{} is less than minimum longitude in file ({}). Consider adding 360".format(lonmin,xLonMin))
            raise RuntimeError
            

        mask_lon = np.logical_and(self.header_.xLon_>=lonmin,self.header_.xLon_<=lonmax)
        #mask_lat = np.logical_and(self.header_.yLat_>=latmin-1.0/12,self.header_.yLat_<=latmax)
        mask_lat = np.logical_and(self.header_.yLat_>=latmin,self.header_.yLat_<=latmax)
        mask_merged  = np.logical_and(mask_lon,mask_lat)
        
        nLon = np.where(mask_merged)[0][-1]-np.where(mask_merged)[0][0]+1
        nLat = np.where(mask_merged)[1][-1]-np.where(mask_merged)[1][0]+1
        xLon = self.header_.xLon_[mask_merged].reshape(nLon,nLat)
        yLat = self.header_.yLat_[mask_merged].reshape(nLon,nLat)
        mask = self.header_.mask_[mask_merged].reshape(nLon,nLat)
        
        outDym = DymFile(outfile)
        outDym.header_.nLon_ = nLon
        outDym.header_.nLat_ = nLat
        outDym.header_.nLev_ = self.header_.nLev_
        outDym.header_.mask_ = mask      
        outDym.header_.xLon_ = xLon
        outDym.header_.yLat_ = yLat
        outDym.header_.zLev_ = self.header_.zLev_
        outDym.header_.firstDate_ = self.header_.firstDate_ 
        outDym.header_.lastDate_  = self.header_.lastDate_  
        
        outDym.header_.writeHeader()
                
        fid_out = open(outDym.fileName_, "rb+")
        
        for t in range(self.header_.nLev_):
        
            in_data = np.float32(self.readData(t+1))
        
            fid_out.seek(0,2)
        
            data = in_data[np.transpose(mask_merged)].reshape(nLat,nLon)
            
            fid_out.write(data)
            
            datamin = np.min(data[np.transpose(mask)!=0])
            datamax = np.max(data[np.transpose(mask)!=0])

            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin

        outDym.header_.updateMinMaxValues(minval, maxval)

        fid_out.close()
        logging.info("Wrote %s" %(outfile))

        return outDym


    #########################################################################
    ## Concat with another file
    #########################################################################

    def concat(self,fileToConcat,outfile):

        inDym2 = DymFile(fileToConcat)
        inDym2.readHeader()
    
        logging.info("Concatenating %s and %s" %(self.fileName_,fileToConcat))

        error=False

        if ( self.header_.nLon_ != inDym2.header_.nLon_ ):
            logging.error("input files has different longitude number")
            error=True

        if ( self.header_.nLat_ != inDym2.header_.nLat_ ):
            logging.error("input files has different latitude number")
            error=True

        if np.sum(self.header_.mask_ - inDym2.header_.mask_) != 0:
            warning.error("input files has different mask")
    
        if np.sum(self.header_.xLon_ - inDym2.header_.xLon_) != 0:
            logging.error("input files has different longitudes")
            error=True
    
        if np.sum(self.header_.yLat_ - inDym2.header_.yLat_) != 0:
            logging.error("input files has different latitude")
            error=True

        if error:
            raise RuntimeError

        zlevels1 = self.header_.zLev_
        nlevels1 = len(zlevels1)

        zlevels2 = inDym2.header_.zLev_
        nlevels2 = len(zlevels2)


        # Check levels consistency
        
        resT1,meanresT1 = dym.dates.timeSteps(zlevels1)
        resT2,meanresT2 = dym.dates.timeSteps(zlevels2)

        # Check only if both files has at least two levels        
        if resT1 is not None and resT2 is not None:

            if abs(meanresT1-meanresT2) > 1e-6:
                logging.warning("Time resolution mismatch: %g != %g" %(meanresT1,meanresT2))
                #raise RuntimeError

        if zlevels2[0]-zlevels1[-1] < 0:
            logging.error("Unconsistent time levels")
            raise RuntimeError

        if resT1 is not None:
            if zlevels2[0]-zlevels1[-1] > meanresT1:
                logging.error("Unconsistent time levels")
                raise RuntimeError

        elif resT2 is not None:
            if zlevels2[0]-zlevels1[-1] > meanresT2:
                logging.error("Unconsistent time levels")
                raise RuntimeError
        else:
            logging.warning("both files have a single record")

        
        outDym = DymFile(outfile)
        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      
        outDym.header_.nLev_      = nlevels1 + nlevels2
        outDym.header_.firstDate_ = np.float32(zlevels1[0])
        outDym.header_.lastDate_  = np.float32(zlevels2[nlevels2-1])
        outDym.header_.zLev_      = np.concatenate((zlevels1,zlevels2))
    
        outDym.header_.writeHeader()
    
        fid_out = open(outDym.fileName_, "rb+")
    
        for t in range(nlevels1):
            in_data = np.float32(self.readData(t+1))
            fid_out.seek(0,2)
            fid_out.write(in_data)
            
            datamin = self.minMaskedData(in_data)
            datamax = self.maxMaskedData(in_data)

            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin

        for t in range(nlevels2):
            in_data = np.float32(inDym2.readData(t+1))
            fid_out.seek(0,2)
            fid_out.write(in_data)
            
            if datamax > maxval :
                maxval = datamax
                    
            if datamin < minval :
                minval = datamin            

        outDym.header_.updateMinMaxValues(minval, maxval)

        fid_out.close()

        logging.info("wrote %s" % outfile)
 
        return outDym

    def concat_list(self,fileList,outfile):
        
        # First check spatial dimension consistency

        zlevels = self.header_.zLev_

        for fileToConcat in fileList:

            logging.info("checking spatial and time dimensions consistency with %s" %fileToConcat)

            inDym2 = DymFile(fileToConcat)
            inDym2.readHeader()
        
            error=False

            if ( self.header_.nLon_ != inDym2.header_.nLon_ ):
                logging.error("input files has different longitude number")
                error=True

            if ( self.header_.nLat_ != inDym2.header_.nLat_ ):
                logging.error("input files has different latitude number")
                error=True

            if np.sum(self.header_.mask_ - inDym2.header_.mask_) != 0:
                logging.warning("input files has different mask")

        
            if np.sum(self.header_.xLon_ - inDym2.header_.xLon_) != 0:
                logging.error("input files has different longitudes")
                error=True
        
            if np.sum(self.header_.yLat_ - inDym2.header_.yLat_) != 0:
                logging.error("input files has different latitude")
                error=True

            if error:
                raise RuntimeError

            # Check levels consistency
            zlevels2 = inDym2.header_.zLev_

            resT,meanresT  = dym.dates.timeSteps(zlevels)
            resT2,meanresT2 = dym.dates.timeSteps(zlevels2)
            
            if resT is not None and resT2 is not None:

                if abs(meanresT-meanresT2) > 1e-6:
                    logging.warning("Time resolution mismatch: %g != %g" %(meanresT,meanresT2))
                    #raise RuntimeError

            if zlevels2[0]-zlevels[-1] <= 0:
                logging.error("Unconsistent time levels")
                raise RuntimeError

            if resT is not None:
                if zlevels2[0]-zlevels[-1] > meanresT:
                    logging.error("Unconsistent time levels")
                    raise RuntimeError
            elif resT2 is not None:
                if zlevels2[0]-zlevels[-1] > meanresT2:
                    logging.error("Unconsistent time levels")
                    raise RuntimeError
            else:
                logging.warning("both files have a single record")


            zlevels = np.concatenate((zlevels,zlevels2))
        
        nlevels = len(zlevels)

        outDym = DymFile(outfile)
        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      
        outDym.header_.nLev_      = nlevels
        outDym.header_.firstDate_ = np.float32(zlevels[0])
        outDym.header_.lastDate_  = np.float32(zlevels[-1])
        outDym.header_.zLev_      = zlevels
    
        outDym.header_.writeHeader()
    
        fid_out = open(outDym.fileName_, "rb+")
            
        # Concatenating data
        # First file
        for t in range(self.header_.nLev_):

            in_data = np.float32(self.readData(t+1))
            fid_out.seek(0,2)
            fid_out.write(in_data)
            
            datamin = self.minMaskedData(in_data)
            datamax = self.maxMaskedData(in_data)

            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin

        for fileToConcat in fileList:
            
            inDym2 = DymFile(fileToConcat)
            
            for t in range(inDym2.header_.nLev_):

                in_data = np.float32(inDym2.readData(t+1))
                fid_out.seek(0,2)
                fid_out.write(in_data)

                datamin = self.minMaskedData(in_data)
                datamax = self.maxMaskedData(in_data)
            
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin                
                
            logging.info("added %s to file" %fileToConcat)

        outDym.header_.updateMinMaxValues(minval, maxval)
        
        fid_out.close()
        logging.info("wrote %s" % outfile)

        return outDym


    def merge(self,fileList,outfile):

        # Merge list of file with non-sorted levels: usefull to merge restart files that have been 
        # produced by parallel run of seapodym

        zlevels = self.header_.zLev_
        
        records = {}
        idx = 0
        for l in zlevels:
            records[l] = (self.fileName_,idx)
            idx = idx + 1

        # Check spatial dimension consistency and get total of records

        for fileToMerge in fileList:

            logging.info("checking spatial consistency with %s" %fileToMerge)

            inDym2 = DymFile(fileToMerge)
            inDym2.readHeader()
        
            error=False

            if ( self.header_.nLon_ != inDym2.header_.nLon_ ):
                logging.error("input files has different longitude number")
                error=True

            if ( self.header_.nLat_ != inDym2.header_.nLat_ ):
                logging.error("input files has different latitude number")
                error=True

            if np.sum(self.header_.mask_ - inDym2.header_.mask_) != 0:
                logging.error("input files has different mask")
                error=True
        
            if np.sum(self.header_.xLon_ - inDym2.header_.xLon_) != 0:
                logging.error("input files has different longitudes")
                error=True
        
            if np.sum(self.header_.yLat_ - inDym2.header_.yLat_) != 0:
                logging.error("input files has different latitude")
                error=True

            if error:
                raise RuntimeError

            zlevels2 = inDym2.header_.zLev_
            idx = 0
            for l in zlevels2:
                records[l] = (fileToMerge,idx)
                idx = idx + 1
            zlevels = np.concatenate((zlevels,zlevels2))

        zlevels.sort()
        nlevels = len(zlevels)

        # Writing output header information
        outDym = DymFile(outfile)
        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      
        outDym.header_.nLev_      = nlevels
        outDym.header_.firstDate_ = np.float32(zlevels[0])
        outDym.header_.lastDate_  = np.float32(zlevels[-1])
        outDym.header_.zLev_      = zlevels
    
        outDym.header_.writeHeader()
    
        fid_out = open(outDym.fileName_, "rb+")
            
        for t in range(nlevels):

            k = zlevels[t]            
            fname = records[k][0]
            rec   = records[k][1]

            inDym = DymFile(fname)

            in_data = np.float32(inDym.readData(rec+1))
            fid_out.seek(0,2)
            fid_out.write(in_data)

            datamin = inDym.minMaskedData(in_data)
            datamax = inDym.maxMaskedData(in_data)

            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin

        outDym.header_.updateMinMaxValues(minval, maxval)
        
        fid_out.close()
        logging.info("wrote %s" % outfile)

        return outDym


    #########################################################################
    ## Spatial interpolation
    #########################################################################

    def interpolate(self,maskFile,maskVarName,outFile,crop=0):
        """
        Spatial interpolate the file over domain defined in maskFile
        :param maskFile   : maskFile containing domain definition
        :param maskVarName: name of variable containing mask information
        :param outFile:  outfile to be written
        :param crop   : crop maskfile size
        :return DymFile object corresponding to outfile
        """

        filval=float(1e34)
        misval=float(-999.0)

        logging.info("interpolating %s" % self.fileName_)

        mask_nc = netcdf.NcFile(maskFile)

        mask_nc.getDimensions()

        varDimName = mask_nc.getVarDimensions(maskVarName)

        if varDimName[-1] == mask_nc.latName_:
            transpose_data = True
        elif varDimName[-1] == mask_nc.lonName_:
            transpose_data = False
        else:
            logging.error("variable %s dimensions seem not correct" %(varDimName[1]))
            quit(1)

        Mnlon  = mask_nc.nLon_ - 2
        Mnlat  = mask_nc.nLat_ - 2

        Mxlon  = np.float32(mask_nc.getLongitudes()[crop:Mnlon+2-crop])
        Mylat  = np.float32(mask_nc.getLatitudes()[crop:Mnlat+2-crop])

        if transpose_data:
            Mask = np.int32(mask_nc.getData(maskVarName)[crop:Mnlon+2-crop, crop:Mnlat+2-crop,])
        else:
            Mask = np.int32(mask_nc.getData(maskVarName)[crop:Mnlat+2-crop, crop:Mnlon+2-crop])

        Resolution = np.abs(Mxlon[1]-Mxlon[0])
        
        if (Resolution - np.abs((Mylat[0]-Mylat[1]))) / Resolution > 1e-3:
            logging.error("input grid is not regular ! %s" %(maskFile,))
            logging.error("longitudinal resolution = {} ".format(Resolution ))
            logging.error("latitudinal  resolution = {} ".format(Mylat[0]-Mylat[1]))
            raise RuntimeError

        (LatMin,LatMax,LonMin,LonMax) = (Mylat[Mnlat-1], Mylat[0], Mxlon[0], Mxlon[Mnlon-1])

        out_nlon, out_nlat = Mask.shape

        # File to be interpolated 
        if os.path.exists(self.fileName_)==False:
            logging.error-("%s: no such file" % self.fileName_)
            raise IOError

        inveclon = np.float32(self.header_.xLon_[:,0])
        inveclat = np.float32(self.header_.yLat_[0,:])
        in_mask  = np.transpose(np.int32(self.header_.mask_))
        in_nlon  = inveclon.size
        in_nlat  = inveclat.size
        in_dx    = inveclon[1]-inveclon[0]
        in_dy    = inveclat[0]-inveclat[1]

        outveclon = Mxlon
        outveclat = np.flipud(Mylat)  # increasing order
        out_nlon  = outveclon.size
        out_nlat  = outveclat.size
        out_dx    = outveclon[1]-outveclon[0] 
        out_dy    = outveclat[1]-outveclat[0]             

        out_mask = copy.copy(Mask)

        logging.info("input  size and resolution: %4d x%4d %10g x%10g" \
                    %(in_nlon,in_nlat,in_dx,in_dy))
        logging.info("output size and resolution: %4d x%4d %10g x%10g" \
                    % (out_nlon,out_nlat,out_dx,out_dy))

        if os.path.exists(outFile):
            os.remove(outFile)

        # Writing Header
        outDym = DymFile(outFile)

        outDym.header_.minVal_    = np.float32(0)
        outDym.header_.maxVal_    = np.float32(1)
        outDym.header_.firstDate_ = np.float32(self.header_.firstDate_)
        outDym.header_.lastDate_  = np.float32(self.header_.lastDate_)
        outDym.header_.zLev_      = np.float32(self.header_.zLev_)
        outDym.header_.nLev_      = np.int32(self.header_.nLev_)
        outDym.header_.nLon_      = np.int32(out_nlon)
        outDym.header_.nLat_      = np.int32(out_nlat)
        outDym.header_.mask_      = np.transpose(out_mask)
        outDym.header_.xLon_      = np.float32(np.transpose(np.tile(outveclon,(out_nlat,1))))
        outDym.header_.yLat_      = np.float32(np.tile(np.flipud(outveclat),(out_nlon,1)))

        outDym.header_.writeHeader()


        fid_out = open(outDym.fileName_, "rb+")

        for t in range(self.header_.nLev_):
            
            in_data = np.float32(self.readData(t+1))
            
            in_tmp =  copy.copy(in_data)
            out_tmp_msk = copy.copy(out_mask)

            # spatial_interp considers masked data with misval value (NetCDF)
            in_tmp[in_mask==0]=misval
            #outdata=np.ones(out_tmp_msk.shape)
            outdata = interpolation.spatial(inveclon, np.flipud(inveclat), np.flipud(in_tmp), outveclon, outveclat,
                                            np.flipud(out_tmp_msk),fill_value=filval,mis_value=misval) 

            # spatial_interp fills mask and isnan outdata with filval (NetCDF)
            # Reset these values to zero for DYM files
            outdata[outdata==filval]=0

            fid_out.seek(0,2)

            out_tmp = copy.copy(outdata)

            out_tmp = np.flipud(out_tmp)
            out_tmp.tofile(fid_out)
            #for j in range(out_nlat):
            #    out_tmp[out_nlat-j-1,:]=outdata[j,:]    
            #fid_out.write(out_tmp)

            logging.info("wrote level %d/%d" %(t+1,self.header_.nLev_))

            datamin = outDym.minMaskedData(out_tmp)
            datamax = outDym.maxMaskedData(out_tmp)
            
            if t == 0:
                minval = datamin
                maxval = datamax
            else:
                if datamax > maxval :
                    maxval = datamax
                    
                if datamin < minval :
                    minval = datamin
                        

        fid_out.close()
        
        outDym.header_.updateMinMaxValues(minval, maxval)

        logging.info("wrote %s" %outFile)

        return outDym

    def time_average(self,outFile,firstDateStr=None,lastDateStr=None,dateFormat="%Y%m%d"):
        
        nlevels = self.header_.nLev_
        zlevels = self.header_.zLev_

        if firstDateStr is None:
            minidx = 0
        else:
            minidx  = dym.dates.findDateStr(zlevels,firstDateStr,dateFormat=dateFormat)

        if lastDateStr is None:
            maxidx = nlevels
        else:    
            maxidx  = dym.dates.findDateStr(zlevels,lastDateStr,dateFormat=dateFormat)

        data = self.readAllData()[minidx:maxidx+1,:,:]

        avg = np.average(data,axis=0)

        outDym = DymFile(outFile)

        outDym.header_.nLon_ = self.header_.nLon_     
        outDym.header_.nLat_ = self.header_.nLat_
        outDym.header_.mask_ = self.header_.mask_      
        outDym.header_.xLon_ = self.header_.xLon_      
        outDym.header_.yLat_ = self.header_.yLat_      
        outDym.header_.nLev_      = 1
        outDym.header_.firstDate_ = np.float32(zlevels[0])
        outDym.header_.lastDate_  = np.float32(zlevels[0])
        outDym.header_.zLev_      = zlevels[0]
    
        outDym.header_.writeHeader()
    
        fid_out = open(outDym.fileName_, "rb+")
            
        # Averaging input data
        
        fid_out.seek(0,2)
        fid_out.write(avg)
            
        minval = outDym.minMaskedData(avg)
        maxval = outDym.maxMaskedData(avg)

        fid_out.close()

        outDym.header_.updateMinMaxValues(minval, maxval)

        logging.info("wrote %s" %outFile)

        return outDym

    
###################################################################################################

def concat(file_list,output=None):

    for fname in file_list:
        if not os.path.exists(fname):
            raise IOError("no such file %s" %fname)

    if output is None:
        output="output.dym"

    inDym = DymFile(file_list[0])

    return inDym.concat_list(file_list[1:],output)


def extractLevels(infile,firstDateStr,lastDateStr,output=None,dateFormat="%Y%m%d"):


    if not os.path.exists(infile):
            raise IOError("no such file %s" %infile)

    if output is None:
        outfile="output.dym"
        
    inDym = DymFile(infile)

    return inDym.extractLevelsDateStr(firstDateStr,lastDateStr,output,dateFormat=dateFormat)



    