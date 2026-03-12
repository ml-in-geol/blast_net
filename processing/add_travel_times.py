import obspy
import pyasdf
import numpy as np
from sys import argv
from obspy.taup import TauPyModel

#tt_mod = TauPyModel('msh')
tt_mod = TauPyModel('PREM')
km_per_degree = (2*np.pi*6371.0)/360.0
print(km_per_degree)

input_file = argv[1]
ds = pyasdf.ASDFDataSet(input_file)

#remove travel times if they exist
try:
    del ds.auxiliary_data.travel_times
except:
    pass

#fmin = 10.0
#fmax = 18.0

for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)
    event_depth = origin.depth

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

        #seis.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)
        #get P and S times
        dist_degree = dist / km_per_degree
        P_arrs = tt_mod.get_travel_times(source_depth_in_km = event_depth,
                distance_in_degree = dist_degree,phase_list = ['P','p','Pn'])
        S_arrs = tt_mod.get_travel_times(source_depth_in_km = event_depth,
                distance_in_degree = dist_degree,phase_list = ['S','s'])

        if len(P_arrs) > 0:
            P_time = P_arrs[0].time
        else:
            print('no P arrivals for dist {} and depth {}'.format(dist_degree,event_depth))
            continue

        if len(S_arrs) > 0:
            S_time = S_arrs[0].time
        else:
            print('no S arrivals for dist {} and depth {}'.format(dist_degree,event_depth))
            continue

        P_time_dict['{}.{}'.format(net_code,sta_code)] = P_time
        S_time_dict['{}.{}'.format(net_code,sta_code)] = S_time
        P_times.append(P_time)
        S_times.append(S_time)

    ds.add_auxiliary_data(data = np.array(P_times), data_type = 'travel_times', path = 'P_times/{}'.format(origin.time), parameters = P_time_dict)
    ds.add_auxiliary_data(data = np.array(S_times), data_type = 'travel_times', path = 'S_times/{}'.format(origin.time), parameters = S_time_dict)
