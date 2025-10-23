import geopandas as gpd
import pandas as pd
import xarray as xr
import glob
import pdb
import sys
import numpy as np 
import os 
from pyproj import CRS
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box
import matplotlib.colors as mcolors

####################################
def generatre_corner_loc(fileGrid):
    #generate grid at corner
    ds = xr.open_dataset(fileGrid)
    
    lat_c = ds['lat'].values
    lon_c = ds['lon'].values

    ny, nx = lat_c.shape
    
    lat_v = np.zeros((ny+1, nx+1), dtype=np.float32)
    lon_v = np.zeros((ny+1, nx+1), dtype=np.float32)

# Internal corners
    lat_v[1:ny, 1:nx] = 0.25 * (
        lat_c[:-1, :-1] + lat_c[1:, :-1] + lat_c[:-1, 1:] + lat_c[1:, 1:]
    )
    lon_v[1:ny, 1:nx] = 0.25 * (
        lon_c[:-1, :-1] + lon_c[1:, :-1] + lon_c[:-1, 1:] + lon_c[1:, 1:]
    )

# --- Edges: use adjacent centers (no shape mismatch) ---
# top and bottom edges
    lat_v[0, 1:-1]   = lat_c[0, :-1]   - 0.5 * (lat_c[1, :-1] - lat_c[0, :-1])
    lat_v[-1, 1:-1]  = lat_c[-1, :-1]  + 0.5 * (lat_c[-1, :-1] - lat_c[-2, :-1])
    lon_v[0, 1:-1]   = lon_c[0, :-1]   - 0.5 * (lon_c[1, :-1] - lon_c[0, :-1])
    lon_v[-1, 1:-1]  = lon_c[-1, :-1]  + 0.5 * (lon_c[-1, :-1] - lon_c[-2, :-1])

# left and right edges
    lat_v[1:-1, 0]   = lat_c[:-1, 0]   - 0.5 * (lat_c[:-1, 1] - lat_c[:-1, 0])
    lat_v[1:-1, -1]  = lat_c[:-1, -1]  + 0.5 * (lat_c[:-1, -1] - lat_c[:-1, -2])
    lon_v[1:-1, 0]   = lon_c[:-1, 0]   - 0.5 * (lon_c[:-1, 1] - lon_c[:-1, 0])
    lon_v[1:-1, -1]  = lon_c[:-1, -1]  + 0.5 * (lon_c[:-1, -1] - lon_c[:-1, -2])

# --- Corners: extrapolate from adjacent cells ---
    lat_v[0, 0]     = lat_c[0, 0]     - 0.5 * (lat_c[1, 1] - lat_c[0, 0])
    lat_v[0, -1]    = lat_c[0, -1]    - 0.5 * (lat_c[1, -2] - lat_c[0, -1])
    lat_v[-1, 0]    = lat_c[-1, 0]    + 0.5 * (lat_c[-1, 0] - lat_c[-2, 1])
    lat_v[-1, -1]   = lat_c[-1, -1]   + 0.5 * (lat_c[-1, -1] - lat_c[-2, -2])

    lon_v[0, 0]     = lon_c[0, 0]     - 0.5 * (lon_c[1, 1] - lon_c[0, 0])
    lon_v[0, -1]    = lon_c[0, -1]    - 0.5 * (lon_c[1, -2] - lon_c[0, -1])
    lon_v[-1, 0]    = lon_c[-1, 0]    + 0.5 * (lon_c[-1, 0] - lon_c[-2, 1])
    lon_v[-1, -1]   = lon_c[-1, -1]   + 0.5 * (lon_c[-1, -1] - lon_c[-2, -2])

# Add to dataset
    ds['lat_corner'] = (('y_corner', 'x_corner'), lat_v)
    ds['lon_corner'] = (('y_corner', 'x_corner'), lon_v)

    ds.to_netcdf(fileGrid.replace('.nc','withCorner.nc'))

    return ds


####################################
def generate_gdf_fromdsGrid(dsout):
    
    x = dsout['x'].values
    y = dsout['y'].values

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

# Compute half-cell offsets
    dx2, dy2 = dx / 2, dy / 2

# Prepare containers
    polygons = []
    ids = []

# Generate polygons
    for j in range(len(y)):
        for i in range(len(x)):
            # center
            xc, yc = x[i], y[j]
            
            # corners (clockwise or counterclockwise order)
            p1 = (xc - dx2, yc - dy2)
            p2 = (xc + dx2, yc - dy2)
            p3 = (xc + dx2, yc + dy2)
            p4 = (xc - dx2, yc + dy2)
            
            poly = Polygon([p1, p2, p3, p4])
            polygons.append(poly)
            ids.append((i, j))

# Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'x_idx': [i for i, j in ids],
                            'y_idx': [j for i, j in ids]},
                           geometry=polygons,
                           crs="EPSG:32631")  # CRS from your dsout.spatial_ref.crs_wkt

    return gdf

##########################
if __name__ == '__main__':
##########################
    dirinHS = '/data/paugam/MTG/hotspot/'
    fileGrid = '/data/paugam/MTG/LSA_MTG_LATLON_MTGH-FD_202505120000.nc'
    #fileGridOut='/data/paugam/FIRES/2025_ribaute/FCI-ribaute-mirT2/ros2_zf4/normalRosArrivalTime.nc'
    fileGridOut='/data/paugam/FIRES/2025_ribaute/FCI-ribaute-mirT2/bmap_mirT2.nc'
    firename = 'ribaute'

    #load Grid
    #----
    if not(os.path.isfile(fileGrid.replace('.nc','withCorner.nc'))):
        ds = generatre_corner_loc(fileGrid)
    else: 
        ds = xr.open_dataset(fileGrid.replace('.nc','withCorner.nc'))

    '''
    gdf_disc = []
    ax = plt.subplot(111)
    for jhs,ihs in np.ndindex(ds.lat.shape):
        pixelcorners1 = (ds.lon_corner[jhs,ihs].values,ds.lat_corner[jhs,ihs].values)
        pixelcorners2 = (ds.lon_corner[jhs,ihs+1].values,ds.lat_corner[jhs,ihs+1].values)
        pixelcorners3 = (ds.lon_corner[jhs+1,ihs+1].values,ds.lat_corner[jhs+1,ihs+1].values)
        pixelcorners4 = (ds.lon_corner[jhs+1,ihs].values,ds.lat_corner[jhs+1,ihs].values)
        try:
            poly = Polygon([pixelcorners1, pixelcorners2, pixelcorners3, pixelcorners4])
            gdf_disc.append( gpd.GeoDataFrame({'id': [1]}, geometry=[poly], crs="EPSG:4326") )
        except: 
            pass

    gdf_disc = gpd.GeoDataFrame(pd.concat(gdf_disc, ignore_index=True))
    gdf_disc.plot(facecolor='none')
    plt.show()
    sys.exit()
    '''
    #load output grid
    #----
    dsout = xr.open_dataset(fileGridOut)
    crsout = CRS.from_wkt(dsout.spatial_ref.crs_wkt)
    gdfout = generate_gdf_fromdsGrid(dsout)

    xmin = float(dsout.x.min())
    xmax = float(dsout.x.max())
    ymin = float(dsout.y.min())
    ymax = float(dsout.y.max())
    bbox = box(xmin, ymin, xmax, ymax)
    bbox_out = gpd.GeoDataFrame(geometry=[bbox], crs=gdfout.crs )

    #load hotspot
    #----
    hsfiles = glob.glob(f'{dirinHS}*.csv')
   
    #create a zeros frp.
    zeros = xr.DataArray(
                            data = 0.0,
                            dims = ("y", "x"),
                            coords = {"y": dsout["y"], "x": dsout["x"]},
                            name = "zeros"
                            )
    #loop through hs
    #----
    frp_frames_all = []
    ratio_frames_all = []
    times_frames_all = []
    for hsfile in hsfiles: 
        hs = gpd.read_file(hsfile)
        hs = hs.rename(columns={'ACQTIME': 'timestamp'})
        
        #flag_nohs = False
        # Group by timestamp
        for it, (timestamp, hs_group) in enumerate(hs.groupby('timestamp')):
            
            hs_group = gpd.GeoDataFrame(hs_group, geometry=gpd.points_from_xy(hs_group.LONGITUDE, hs_group.LATITUDE), crs="EPSG:4326")
            hs_group_loc = gpd.clip(hs_group.to_crs(crsout), bbox_out)
            #if len(hs_group_loc) == 0 : 
            #    flag_nohs = True
            #    continue
            #flag_nohs = False
            if it == 0 : 
                frp_frame = zeros.copy().rename('frp')
                ratio_frame = zeros.copy().rename('frp')

            for ihs, hs_ in hs_group_loc.iterrows():
               
                print(f"  Timestamp: {timestamp} | {len(hs_group)} hotspots | {len(frp_frames_all)} frp frames | {ihs} | {hs_.FRP}" )
                ihs = int(float(hs_['ABS_SAMP']))
                jhs = int(float(hs_['ABS_LINE']))
                pixelcorners1 = (ds.lon_corner[jhs,ihs].values,ds.lat_corner[jhs,ihs].values)
                pixelcorners2 = (ds.lon_corner[jhs,ihs+1].values,ds.lat_corner[jhs,ihs+1].values)
                pixelcorners3 = (ds.lon_corner[jhs+1,ihs+1].values,ds.lat_corner[jhs+1,ihs+1].values)
                pixelcorners4 = (ds.lon_corner[jhs+1,ihs].values,ds.lat_corner[jhs+1,ihs].values)

                poly = Polygon([pixelcorners1, pixelcorners2, pixelcorners3, pixelcorners4])
                gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[poly], crs="EPSG:4326")
                gdf = gdf.to_crs(crsout)
             
                '''
                # Create matching Cartopy CRS
                fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8,8))
                gdfout.to_crs(4326).plot(ax=ax, transform=ccrs.PlateCarree(),
                            edgecolor='k', facecolor='none', linewidth=0.2)

                ax.coastlines(resolution='10m', color='blue', linewidth=0.5)
                ax.gridlines(draw_labels=True)
                gdf.to_crs(4326).plot(ax=ax)
                plt.show()
                '''
                # 1. geometric intersections
                inter = gpd.overlay(gdfout, gdf, how="intersection")

                # 2. measure areas
                inter["ratio_area"] = inter.geometry.area/gdf.geometry.area.iloc[0]

                ratio = zeros.copy()
                ratio.values[inter.y_idx, inter.x_idx] = inter.ratio_area

                '''
                fig, ax = plt.subplots(figsize=(8, 8))

                # Compute valid range for log scale
                vmin = float(ratio.where(ratio > 0).min())
                vmax = float(ratio.max())

                # --- Plot with no interpolation (pixel-perfect)
                im = ratio.plot.imshow(
                    ax=ax,
                    cmap="viridis",
                    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                    add_colorbar=True,
                    interpolation="none"  # works with imshow
                )

                # --- Overlay polygons
                gdf.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)
                inter.plot(ax=ax, facecolor='none', edgecolor='r', linewidth=0.5)

                # --- Add pixel grid
                x_edges = np.concatenate(([ratio.x.values[0] - np.diff(ratio.x.values)[0]/2],
                                          (ratio.x.values[:-1] + ratio.x.values[1:]) / 2,
                                          [ratio.x.values[-1] + np.diff(ratio.x.values)[-1]/2]))
                y_edges = np.concatenate(([ratio.y.values[0] - np.diff(ratio.y.values)[0]/2],
                                          (ratio.y.values[:-1] + ratio.y.values[1:]) / 2,
                                          [ratio.y.values[-1] + np.diff(ratio.y.values)[-1]/2]))
                ax.set_xticks(x_edges, minor=True)
                ax.set_yticks(y_edges, minor=True)
                ax.grid(which='minor', color='gray', linewidth=0.3, linestyle='-')

                ax.set_title("Log-scaled ratio with pixel grid and index display", fontsize=12)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                ax.set_title("Log-scaled ratio (no interpolation, pixel values)")

                # --- Dynamic label for pixel index + value
                text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                               fontsize=10, va="top", ha="left",
                               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

                # --- Mouse event: show pixel coordinates + value
                xv, yv = ratio.x.values, ratio.y.values

                def on_move(event):
                    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                        x, y = event.xdata, event.ydata
                        ix = np.argmin(np.abs(xv - x))
                        iy = np.argmin(np.abs(yv - y))
                        val = ratio.isel(x=ix, y=iy).item()
                        text.set_text(f"x_idx={ix}, y_idx={iy}\nval={val:.3e}")
                        fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", on_move)

                plt.tight_layout()
                plt.show()
                '''
                frp_frame.values = frp_frame.values + ratio.values * float(hs_.FRP)
                ratio_frame.values = ratio_frame.values + ratio.values 
    
        #if flag_nohs: continue   
        frp_frames_all.append(frp_frame)
        ratio_frames_all.append(ratio_frame)
        times_frames_all.append(timestamp)    

    # Convert timestamps to datetime64
    times = pd.to_datetime(times_frames_all, format="%Y%m%d%H%M%S")
    # Combine along a new time dimension
    da_frp = xr.concat(frp_frames_all, dim=xr.DataArray(times, dims="time", name="time"))
    da_frp.attrs['units'] = 'MW'
    da_ratio = xr.concat(ratio_frames_all, dim=xr.DataArray(times, dims="time", name="time"))

    dsout = xr.Dataset({
                        "frp": da_frp,
                        "ratio": da_ratio
                        })
    dsout.to_netcdf(f"{os.path.dirname(fileGridOut)}/../FCI-FRP/{firename}_frp_gridded.nc")

