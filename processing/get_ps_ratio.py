import obspy
import pyasdf
import numpy as np
from sys import argv
from obspy.taup import TauPyModel

input_file = argv[1]
ds = pyasdf.ASDFDataSet(input_file)

fmin = 10.0
fmax = 18.0

#delete metadata if it exists
try:
    #del ds.auxiliary_data.PS_ratios['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)]
    #del ds.auxiliary_data.SNR['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)]
    del ds.auxiliary_data.PS_ratios
    del ds.auxiliary_data.SNR
except:
    pass
#del ds.auxiliary_data.PS_ratios
#del ds.auxiliary_data.SNR

#SNR_dict = {}
#PS_ratio_dict = {}
#SNRs = []
#PS_ratios = []

for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)
    event_depth = origin.depth

    ad1 = ds.auxiliary_data.distances.distances[event_name]
    distance_dict = ad1.parameters

    ad2 = ds.auxiliary_data.travel_times.P_times[event_name]
    P_time_dict = ad2.parameters

    ad3 = ds.auxiliary_data.travel_times.S_times[event_name]
    S_time_dict = ad3.parameters

    SNR_dict = {}
    PS_ratio_dict = {}
    SNRs = []
    PS_ratios = []

    for station in ds.ifilter(ds.q.event == event):

        #get stream
        seis = station.processed
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station

        #get distance and travel times
        dist = distance_dict['{}.{}'.format(net_code,sta_code)]
        try:
            P_time = P_time_dict['{}.{}'.format(net_code,sta_code)]
            S_time = S_time_dict['{}.{}'.format(net_code,sta_code)]
            print(P_time,S_time)
        except:
            print('No P or S time found for {} {}'.format(net_code,sta_code))
            continue

        seis.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)
        #calculate P and S ratios, and SNR
        components = [tr.stats.channel[-1] for tr in seis]

        if "R" in components and "T" in components and "Z" in components:

            trZ = seis.select(channel = '*HZ')[0]
            trR = seis.select(channel = '*HR')[0]
            trT = seis.select(channel = '*HT')[0]
            net_code = trZ.stats.network
            sta_code = trZ.stats.station

            #set window
            W = (S_time - P_time)*0.5
            if W < 1:
                continue
            elif W > 3:
                W = 3

            P2 = P_time - (W*0.05)
            S2 = S_time - (W*0.05)
            N = 10.0

            starttime_P = origin.time + P2
            starttime_S = origin.time + S2
            starttime_N = origin.time - N
            endtime_P = starttime_P + W
            endtime_S = starttime_S + W
            endtime_N = starttime_N + W

            P_winZ = trZ.slice(starttime=starttime_P,endtime=endtime_P)
            P_winR = trR.slice(starttime=starttime_P,endtime=endtime_P)
            P_winT = trT.slice(starttime=starttime_P,endtime=endtime_P)
            S_winZ = trZ.slice(starttime=starttime_S,endtime=endtime_S)
            S_winR = trR.slice(starttime=starttime_S,endtime=endtime_S)
            S_winT = trT.slice(starttime=starttime_S,endtime=endtime_S)
            N_winZ = trZ.slice(starttime=starttime_N,endtime=endtime_N)
            N_winR = trR.slice(starttime=starttime_N,endtime=endtime_N)
            N_winT = trT.slice(starttime=starttime_N,endtime=endtime_N)

            #print(P_winZ, np.mean(P_winZ.data))
            #print(P_winR, np.mean(P_winR.data))
            #print(P_winT, np.mean(P_winT.data))
            #print(S_winZ, np.mean(S_winZ.data))
            #print(S_winR, np.mean(S_winR.data))
            #print(S_winT, np.mean(S_winT.data))
            #print(N_winZ, np.mean(N_winZ.data))
            #print(N_winR, np.mean(N_winR.data))
            #print(N_winT, np.mean(N_winT.data))

            P_Z = np.mean((P_winZ.data*1e9)**2)
            P_R = np.mean((P_winR.data*1e9)**2)
            P_T = np.mean((P_winT.data*1e9)**2)
            S_Z = np.mean((S_winZ.data*1e9)**2)
            S_R = np.mean((S_winR.data*1e9)**2)
            S_T = np.mean((S_winT.data*1e9)**2)
            N_Z = np.mean((N_winZ.data*1e9)**2)
            N_R = np.mean((N_winR.data*1e9)**2)
            N_T = np.mean((N_winT.data*1e9)**2)

            #P_eng = P_Z + P_R + P_T
            #S_eng = S_Z + S_R + S_T
            #N_eng = N_Z + N_R + N_T
            #P_sum = np.sqrt(P_eng - N_eng)
            #S_sum = np.sqrt(S_eng - N_eng)
            #N_sum = np.sqrt(N_eng)

            P_sum = np.sqrt( (P_Z+P_R+P_T) - (N_Z+N_R+N_T) )
            S_sum = np.sqrt( (S_Z+S_R+S_T) - (N_Z+N_R+N_T) )
            N_sum = np.sqrt( N_Z+N_R+N_T )

            PS_ratio = P_sum / S_sum
            SNR = P_sum / N_sum

            #print(N_sum,S_sum)
            #print('PS_ratio, SNR: {}, {}'.format(PS_ratio,SNR))

            SNRs.append(SNR)
            PS_ratios.append(PS_ratio)

            SNR_dict['{}.{}'.format(net_code,sta_code)] = SNR
            PS_ratio_dict['{}.{}'.format(net_code,sta_code)] = PS_ratio

        else:
            print('**************************************')
            print('Z, R, and T components not available')
            print(seis)
            print('**************************************')
            pass

    ds.add_auxiliary_data(data = np.array(PS_ratios), data_type = 'PS_ratios',
            path = '{}/f_{:2.2f}_{:2.2f}'.format(event_name,fmin,fmax), parameters = PS_ratio_dict)
    ds.add_auxiliary_data(data = np.array(SNRs), data_type = 'SNR',
            path = '{}/f_{:2.2f}_{:2.2f}'.format(event_name,fmin,fmax), parameters = SNR_dict)

#ds.add_auxiliary_data(data = np.array(PS_ratios), data_type = 'PS_ratios',
#        path = 'f_{:2.2f}_{:2.2f}'.format(fmin,fmax), parameters = PS_ratio_dict)
#ds.add_auxiliary_data(data = np.array(SNRs), data_type = 'SNR',
#        path = 'f_{:2.2f}_{:2.2f}'.format(fmin,fmax), parameters = SNR_dict)
