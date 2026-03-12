import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sys import argv

import matplotlib
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
ax = fig.add_axes([0.1,0.15,0.78,0.75])
bmap = Basemap(projection='merc',llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,ax=ax,resolution='i')
bmap.fillcontinents('lightgrey')
bmap.drawstates()

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

x,y = bmap(eq_lons,eq_lats)
eq_mags = np.array(eq_mags)
#bmap.scatter(x,y,s=eq_mags**2*100.,marker='*',edgecolor='red',facecolor='none',zorder=99)
bmap.scatter(x,y,s=eq_mags*25.,marker='*',edgecolor='red',facecolor='none',zorder=99)
x,y = bmap(expl_lons,expl_lats)
bmap.scatter(x,y,marker='s',edgecolor='blue',facecolor='none',zorder=99)

station_lats = []
station_lons = []

for station in ds.waveforms:
    inv = station.StationXML
    station_lats.append(inv[0][0].latitude)
    station_lons.append(inv[0][0].longitude)
x,y = bmap(station_lons,station_lats)
bmap.scatter(x,y,marker='^',facecolor='black')

step = 2
parallels = np.arange(-80.,81,step)
meridians = np.arange(-180.,179,step)
bmap.drawparallels(parallels,labels=[False,True,True,False])
bmap.drawmeridians(meridians,labels=[False,True,True,False])

#bmap.shadedrelief(scale=1.0)
#bmap.etopo()
#bmap.bluemarble()
#plt.show()

plt.savefig(output_file)
