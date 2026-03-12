import os
import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from obspy.signal.tf_misfit import cwt
from scipy.interpolate import interp2d
from obspy.signal.invsim import cosine_taper

debug = False
ngood_expl = 0
ngood_eqk = 0
nbad = 0

dist_max = 400.0
f_min = 2.0
f_max = 18.0
snr_thresh = 2.0
#psr_max = 5.0
psr_max = 100.0
#min_mag = 1.5
#max_mag = 3.0
min_mag = 0.5
max_mag = 4.0

#read data
input_file = argv[1]
ds = pyasdf.ASDFDataSet(input_file)
for i_ev,event in enumerate(ds.events):

    print(i_ev,event)

    if debug:
        print('working in debug mode... exiting after 1 event')
        if i_ev > 0:
            continue

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    if event.event_type == 'explosion':
        e_type = 1
    elif event.event_type == 'earthquake':
        e_type = 0
        e_mag = event.magnitudes[0].mag
        if e_mag < min_mag or e_mag > max_mag:
            print('skipping event... magnitude {} is outside range {} - {}'.format(e_mag,min_mag,max_mag))

    distances = ds.auxiliary_data.distances.distances[event_name].parameters
    ps_ratios = ds.auxiliary_data.PS_ratios['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(10.0,18.0)].parameters
    snr = ds.auxiliary_data.SNR['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(10.0,18.0)].parameters
    P_times = ds.auxiliary_data.travel_times.P_times[event_name].parameters
    S_times = ds.auxiliary_data.travel_times.S_times[event_name].parameters

    try:
        print('event type is {}'.format(e_type))
    except:
        print('continued... something absent in metadata')
        continue

    for station in ds.ifilter(ds.q.event == event):

        seis = station.processed
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station
        samprate = seis[0].stats.sampling_rate

        try:
            ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            print('no ps_ratio found for {}.{}'.format(net_code,sta_code))
            nbad += 1
            continue

        try:
            dist_here = distances['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            print('no distance found for {}.{}'.format(net_code,sta_code))
            nbad += 1
            continue

        try:
            snr_here = snr['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            print('no snr found for {}.{}'.format(net_code,sta_code))
            nbad += 1
            continue

        try:
            P_time = P_times['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            print('no P_time found for {}.{}'.format(net_code,sta_code))
            nbad += 1
            continue

        try:
            S_time = S_times['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            print('no S_time found for {}.{}'.format(net_code,sta_code))
            nbad += 1
            continue

        if snr_here >= snr_thresh and ps_here < psr_max and dist_here < dist_max:

            #if debug:
            #print(seis)

            try:
                trz = seis.select(channel='*HZ')[0]
                trr = seis.select(channel='*HR')[0]
                trt = seis.select(channel='*HT')[0]
            except:
                print('didnt find three components')
                continue

            #if debug:
            print('{}.{} passes checks'.format(net_code,sta_code))

            if e_type == 1:
                ngood_expl += 1
            elif e_type == 0:
                ngood_eqk += 1

    # print('good: {}, bad: {}'.format(n_good,nbad))
    print('ngood_eqk, ngood_expl: {}, {}'.format(ngood_eqk,ngood_expl))
    #TODO give reason for why it failed. (snr, dist, psr_max)
