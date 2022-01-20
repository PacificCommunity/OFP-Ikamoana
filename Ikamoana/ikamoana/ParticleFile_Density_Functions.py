import xarray as xr
import numpy as np

def getPdata(filename, variables=['time','lon','lat']):
    pfile = xr.open_dataset(filename, decode_cf=True)
    data = {}
    for v in variables:
       data.update({v:np.ma.filled(pfile.variables[v], np.nan)})
    pfile.close()
    return data

def calcDensity(lon,lat, lon_lim=[110,290],lat_lim=[-30,30]):
    N = np.shape(lon)[0]
    x_lon = np.arange(lon_lim[0],lon_lim[1]+1)
    y_lat = np.arange(lat_lim[0], lat_lim[1]+1)
    Density = np.zeros([len(x_lon),len(y_lat)], dtype=np.float32)
    for p in range(N):
        lons = np.round(lon[p]) #round2res(lon[p],1)
        lats = np.round(lat[p]) #round2res(lat[p],1)
        T = len(lons)
        if T > 0:
            Density[[int(l) for l in lons-lon_lim[0]],[int(l) for l in lats-lat_lim[0]]] += 1/T
            
    return Density
