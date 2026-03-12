import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sys import argv

import matplotlib
import matplotlib.ticker as mticker
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#--------------------------------------------------------
#open dataset
#--------------------------------------------------------
input_file = argv[1]
output_file = argv[2]
lon_min = float(argv[3])
lon_max = float(argv[4])
lat_min = float(argv[5])
lat_max = float(argv[6])
ds = pyasdf.ASDFDataSet(input_file)

#--------------------------------------------------------
#set up map
#--------------------------------------------------------
event_lats = []
event_lons = []
for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)
    event_lats.append(origin.latitude)
    event_lons.append(origin.longitude)

#pad = 3.0
#lon_min = np.min(event_lons) - pad
#lon_max = np.max(event_lons) + pad
#lat_min = np.min(event_lats) - pad
#lat_max = np.max(event_lats) + pad
fig = plt.figure(figsize=[8,6])
ax = fig.add_axes([0.1,0.15,0.78,0.75], projection=ccrs.Mercator())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.STATES, linewidth=0.4)

#--------------------------------------------------------
#plot all events and stations
#--------------------------------------------------------
eq_lats = []
eq_lons = []
eq_mags = []
expl_lats = []
expl_lons = []

for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    if event.event_type == 'explosion':
        expl_lats.append(origin.latitude)
        expl_lons.append(origin.longitude)
    elif event.event_type == 'earthquake':
        eq_lats.append(origin.latitude)
        eq_lons.append(origin.longitude)
        eq_mags.append(event.magnitudes[0].mag)

eq_mags = np.array(eq_mags)
ax.scatter(eq_lons, eq_lats, s=eq_mags * 25., marker='*', edgecolor='red',
           facecolor='none', zorder=99, transform=ccrs.PlateCarree())
ax.scatter(expl_lons, expl_lats, marker='s', edgecolor='blue', facecolor='none',
           zorder=99, transform=ccrs.PlateCarree())

station_lats = []
station_lons = []

for station in ds.waveforms:
    inv = station.StationXML
    station_lats.append(inv[0][0].latitude)
    station_lons.append(inv[0][0].longitude)
ax.scatter(station_lons, station_lats, marker='^', facecolor='black',
           edgecolor='black', transform=ccrs.PlateCarree())

step = 2
gridlines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False
gridlines.right_labels = False
gridlines.xlocator = mticker.FixedLocator(np.arange(-180.,179,step))
gridlines.ylocator = mticker.FixedLocator(np.arange(-80.,81,step))

#bmap.shadedrelief(scale=1.0)
#bmap.etopo()
#bmap.bluemarble()
#plt.show()

plt.savefig(output_file)
plt.close(fig)
