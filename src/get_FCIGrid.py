 # script to generate 1km lat/lon
#Replace the dir_fci with your 1km MTGH data
import hdf5plugin
from satpy.scene import Scene
import numpy as np
from glob import glob
import xarray as xr


## get lat lon using satpy
repeat_cycle = "0085"
dir_fci = f"/USER_DATA/MTG_FRP/MTGH/2025/05/12/"
file_list = glob(f"{dir_fci}/*BODY*{repeat_cycle}_00??.nc")
c = Scene(filenames=file_list,reader=['fci_l1c_nc'])
c.load(["ir_105"], )
lonSPY,latSPY = c["ir_105"].attrs['area'].get_lonlats()
# flipud satpy arrays to match those of LSA SAF
lonSPY = np.flipud(lonSPY)
latSPY = np.flipud(latSPY)

ds = xr.Dataset({'lat': (['y', 'x'], np.ma.fix_invalid(latSPY.astype('f4'))),
                 'lon': (['y', 'x'], np.ma.fix_invalid(lonSPY.astype('f4')))},)
FOUTPUT="LSA_MTG_LATLON_MTGH-FD_202505120000.nc"
ds.to_netcdf(FOUTPUT,encoding={'lat': {"zlib": True, "complevel": 1},
                               'lon': {"zlib": True, "complevel": 1}})
