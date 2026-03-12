import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

input_file = argv[1]
output_dir = argv[2]
output_name = argv[3]
outfile = '{}/{}'.format(output_dir,output_name)

preset = 120.1
offset = 120.1
ds = pyasdf.ASDFDataSet(input_file)
fmin = 10.0
fmax = 18.0
snr_thresh = 2.0
psr_max = 5.0
sf = 5.0
alpha = 0.35

ps_all = []
dist_all = []
snr_all = []
e_types = []

n_good = 0
plot_bad = False

for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    try:
        distances = ds.auxiliary_data.distances.distances[event_name].parameters
        ps_ratios = ds.auxiliary_data.PS_ratios['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters
        snr = ds.auxiliary_data.SNR['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters
        P_times = ds.auxiliary_data.travel_times.P_times[event_name].parameters
        S_times = ds.auxiliary_data.travel_times.S_times[event_name].parameters
    except:
        continue

    print(P_times)

    for station in ds.ifilter(ds.q.event == event):

        seis = station.processed
        seis.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station

        #ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
        #dist_here = distances['{}.{}'.format(net_code,sta_code)]
        #snr_here = snr['{}.{}'.format(net_code,sta_code)]

        try:
            ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
            dist_here = distances['{}.{}'.format(net_code,sta_code)]
            snr_here = snr['{}.{}'.format(net_code,sta_code)]
            P_time = P_times['{}.{}'.format(net_code,sta_code)]
            S_time = S_times['{}.{}'.format(net_code,sta_code)]

            if snr_here >= snr_thresh and ps_here < psr_max:

                n_good += 1

                tr = seis.select(channel='*HZ')[0]
                tr.normalize()
                time = tr.times()
                data = (tr.data*sf)+dist_here
                plt.plot(time-preset,data,alpha=alpha,linewidth=0.5,color='k')
                #plt.scatter(P_time+30.0,dist_here,marker='o',color='r')
                #plt.scatter(S_time+30.0,dist_here,marker='o',color='b')
                plt.scatter(P_time,dist_here,marker='o',color='r')
                plt.scatter(S_time,dist_here,marker='o',color='b')

            else:
                if plot_bad:
                    tr = seis.select(channel='*HZ')[0]
                    tr.normalize()
                    time = tr.times()
                    data = (tr.data*4.0)+dist_here
                    plt.plot(time,data,alpha=0.5,linewidth=0.5,color='r')

        except:
            #print('cant find ratios for {}.{}'.format(net_code,sta_code))
            continue

plt.xlim([0,offset])
plt.title('{} good waveforms'.format(n_good))
#plt.show()
plt.savefig(outfile,transparent=True)
