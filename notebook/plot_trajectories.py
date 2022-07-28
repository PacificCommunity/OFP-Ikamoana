import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset

ncP = Dataset('outputfile.nc')
lons = ncP['lon']
lats = ncP['lat']


ts = 6
vs = [0, 1.]

if(False):
    fig, ax = plt.subplots(1,ts+1, figsize=(30,4), gridspec_kw={'width_ratios': [10,10,10,10,10,10,2]})

    for i in range(ts):
        k = 6*i + 2
        nc = Dataset('outputfile_%.4dP.nc'%(k))
        if(i==0):
            lonF = nc['nav_lon'][:]
            latF = nc['nav_lat'][:]
            ax[i].set_ylabel('latitude (N)')
        P = nc['P'][0,0]

        print(P.min())
        im = ax[i].pcolormesh(lonF, latF, P, vmin=vs[0], vmax=vs[1])
        ax[i].scatter(lons[:,k], lats[:,k], c='red', s=1)
        ax[i].set_title('time step %d'%(k))
        ax[i].set_xlabel('longitude (E)')

    plt.colorbar(im, cax=ax[-1], extend='max')
    ax[-1].set_title('prey index')
    plt.show()


ts = 200

if(True):
    for i in range(ts):
        fig, ax = plt.subplots(1,2, figsize=(20,17), gridspec_kw={'width_ratios': [20,3]})
        k = i + 2
        nc = Dataset('outputfile_%.4dP.nc'%(k))
        lonF = nc['nav_lon'][:]-0.05
        latF = nc['nav_lat'][:]-0.05
        P = nc['P'][0,0]

        im = ax[0].pcolormesh(lonF, latF, P, vmin=vs[0], vmax=vs[1])
        ax[0].scatter(lons[:,k], lats[:,k], edgecolor='k', c='coral', s=20, label='tuna particle')
        ax[0].set_title('days %d'%(k))
        ax[0].set_xlabel('longitude (E)')
        ax[0].set_ylabel('latitude (N)')

        ax[0].legend(loc='lower right', bbox_to_anchor=(0.6, 1, 0.5, 0.5))
        plt.colorbar(im, cax=ax[-1])
        ax[1].set_title('prey index')
        plt.savefig('figs/no%.4d.png'%(k), dpi=250)
        plt.close()




