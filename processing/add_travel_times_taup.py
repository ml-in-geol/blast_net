import obspy
import pyasdf
import subprocess
import numpy as np
from sys import argv
from obspy.taup import TauPyModel

km_per_degree = (2*np.pi*6371.0)/360.0

input_file = argv[1]
taup_model = argv[2]

ds = pyasdf.ASDFDataSet(input_file)

#remove travel times if they exist
try:
    del ds.auxiliary_data.travel_times
except:
    pass

for i_ev,event in enumerate(ds.events):
    print('event: {}'.format(i_ev))

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)
    event_depth = origin.depth

    #some explosion event depths are above sea level (negative depth)
    if event_depth < 0:
        event_depth = 0.0

    d = ds.auxiliary_data.distances.distances[event_name]
    distance_dict = d.parameters

    P_time_dict = {}
    S_time_dict = {}
    P_times = []
    S_times = []

    for station in ds.ifilter(ds.q.event == event):

        #get stream
        seis = station.processed
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station

        #get distance
        dist = distance_dict['{}.{}'.format(net_code,sta_code)]
        dist_degree = dist / km_per_degree

        a = subprocess.Popen('taup_time -mod {} -h {} -deg {} -ph P,p --time'.format(taup_model,event_depth,dist_degree),stdout=subprocess.PIPE,shell=True)
        b = subprocess.Popen('taup_time -mod {} -h {} -deg {} -ph S,s --time'.format(taup_model,event_depth,dist_degree),stdout=subprocess.PIPE,shell=True)
        P_arrs = a.stdout.read()
        S_arrs = b.stdout.read()

        P_times = P_arrs.split()
        S_times = S_arrs.split()

        if len(P_times) == 0 or len(S_times) == 0:
            print('travel times missing for {} {}'.format(event_depth,dist_degree))
            print('{} {}'.format(P_times,S_times))
            continue

        else:
            P_time = float(P_arrs.split()[0])
            S_time = float(S_arrs.split()[0])

        P_time_dict['{}.{}'.format(net_code,sta_code)] = P_time
        S_time_dict['{}.{}'.format(net_code,sta_code)] = S_time
        P_times.append(P_time)
        S_times.append(S_time)

    ds.add_auxiliary_data(data = np.array(P_times), data_type = 'travel_times', path = 'P_times/{}'.format(origin.time), parameters = P_time_dict)
    ds.add_auxiliary_data(data = np.array(S_times), data_type = 'travel_times', path = 'S_times/{}'.format(origin.time), parameters = S_time_dict)
