import os
import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from obspy.signal.tf_misfit import cwt
from scipy.interpolate import interp2d
from obspy.signal.invsim import cosine_taper

import warnings
warnings.filterwarnings("ignore")

plot=True

#read data
input_file = argv[1]
dataset_name = argv[2]
outdir= argv[3]
dset_multiplier = int(argv[4])
augment = True

#make outdir if it doesn't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

ds = pyasdf.ASDFDataSet(input_file)
labels = open('labels_scalogram_{}.csv'.format(dataset_name),'w')

#scalogram characterisics
slice_scalogram = True
#normalize = 'P_amplitude' # 'simple'
normalize = 'spec_max' # 'simple'
preset = 120.1
offset = 120.1
#time_before_P = 30.0
#time_after_P = 60.0
time_before_P = 10.0
time_after_P = 80.0
sig_len = time_before_P + time_after_P

#dist_max = 400.0
dist_max = 250.0
#f_min = 2.0
f_min = 0.1
f_max = 18.0
snr_thresh = 2.0
psr_max = 5.0
min_mag = 1.5
max_mag = 3.0

n_good = 0
n_bad = 0

debug=False

for i_ev,event in enumerate(ds.events):

    if debug:
        #print('working in debug mode... exiting after 1 event')
        if i_ev > 0:
            continue

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)
    evlo = origin.longitude
    evla = origin.latitude
    evdp = origin.depth

    if event.event_type == 'explosion':
        e_type = 1
    elif event.event_type == 'earthquake':
        e_type = 0
        e_mag = event.magnitudes[0].mag
        #if e_mag < min_mag or e_mag > max_mag:
        #    print('skipping event... magnitude {} is outside range {} - {}'.format(e_mag,min_mag,max_mag))

    distances = ds.auxiliary_data.distances.distances[event_name].parameters
    ps_ratios = ds.auxiliary_data.PS_ratios['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(10.0,18.0)].parameters
    snr = ds.auxiliary_data.SNR['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(10.0,18.0)].parameters
    P_times = ds.auxiliary_data.travel_times.P_times[event_name].parameters
    S_times = ds.auxiliary_data.travel_times.S_times[event_name].parameters

    for station in ds.ifilter(ds.q.event == event):

        inv = station.StationXML
        stla = inv[0][0].latitude
        stlo = inv[0][0].longitude

        seis = station.processed
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station
        samprate = seis[0].stats.sampling_rate

        try:
            ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            #print('no ps_ratio found for {}.{}'.format(net_code,sta_code))
            n_bad += 1
            continue

        try:
            dist_here = distances['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            #print('no distance found for {}.{}'.format(net_code,sta_code))
            n_bad += 1
            continue

        try:
            snr_here = snr['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            #print('no snr found for {}.{}'.format(net_code,sta_code))
            n_bad += 1
            continue

        try:
            P_time = P_times['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            #print('no P_time found for {}.{}'.format(net_code,sta_code))
            n_bad += 1
            continue

        try:
            S_time = S_times['{}.{}'.format(net_code,sta_code)]
        except KeyError:
            #print('no S_time found for {}.{}'.format(net_code,sta_code))
            n_bad += 1
            continue

        if snr_here >= snr_thresh and dist_here < dist_max:

            print('{}.{} passes checks'.format(net_code,sta_code))

            outname = '{}_{:04d}'.format(dataset_name,n_good+1)
            n_good += 1

            if slice_scalogram:
                starttime = seis[0].stats.starttime
                new_start = starttime + preset - time_before_P + P_time
                new_end = new_start + sig_len

                noise_start = starttime
                noise_end = noise_start + sig_len

                noise = seis.copy().slice(starttime=noise_start,endtime=noise_end)
                seis = seis.copy().slice(starttime=new_start,endtime=new_end)
                noise.taper(0.05)
                seis.taper(0.05)

                if seis[0].stats.npts != noise[0].stats.npts:
                    continue

                #apply a random shift to whole trace
                t_shift_max = 2.0
                ind_shift_max = int(samprate * t_shift_max)
                ind_shift = np.random.randint(-ind_shift_max,ind_shift_max)
                for tr in seis:
                    tr.data = np.roll(tr.data,ind_shift)

                print('size of seis {}'.format(seis[0].stats.npts))
                print('size of noise {}'.format(noise[0].stats.npts))

            #skip if it can't find one of the channels for some reason
            try:
                trz = seis.select(channel='*HZ')[0]
                trr = seis.select(channel='*HR')[0]
                trt = seis.select(channel='*HT')[0]

                nrz = noise.select(channel='*HZ')[0]
                nrr = noise.select(channel='*HR')[0]
                nrt = noise.select(channel='*HT')[0]
            except:
                #print('didnt find three components')
                continue

            if normalize == 'P_amplitude':
                ind_P0 = int(samprate * (time_before_P - 2.0))
                ind_P1 = int(samprate * (time_before_P + 2.0))

                trc = trz.copy()
                trc.filter('bandpass',freqmin=6.0,freqmax=18.0,corners=2,zerophase=True)

                #find P amp in 6 - 18 Hz band
                P_amp = np.max(trc.data[ind_P0:ind_P1])

                #normalize full trace on P 6-18 Hz band
                trz.data /= P_amp
                trr.data /= P_amp
                trt.data /= P_amp

                nrz.data /= P_amp
                nrr.data /= P_amp
                nrt.data /= P_amp

            elif normalize == 'simple':
                trz.normalize()
                trr.normalize()
                trt.normalize()

                nrz.normalize()
                nrr.normalize()
                nrt.normalize()

            elif normalize == 'spec_max':
                pass

            npts = trz.stats.npts
            dt = trz.stats.delta
            #print(npts*dt)

            #apply cosine taper to shifted data before scalogram calculation
            taper = cosine_taper(len(trz.data),0.1)
            trz.data *= taper
            trr.data *= taper
            trt.data *= taper

            scalogram_z = cwt(trz.data, dt, 8, f_min, f_max)
            scalogram_r = cwt(trr.data, dt, 8, f_min, f_max)
            scalogram_t = cwt(trt.data, dt, 8, f_min, f_max)

            t = np.linspace(0, dt * npts, npts)
            f = np.linspace(f_min,f_max, scalogram_z.shape[0])

            #interpolate onto new grid
            npts_t_new = 400
            #npts_f_new = 100
            npts_f_new = 50
            t_i = np.linspace(0,dt * npts,npts_t_new)
            f_i = np.linspace(f_min,f_max,npts_f_new)

            func_z = interp2d(t,f,np.abs(scalogram_z),kind='cubic')
            func_r = interp2d(t,f,np.abs(scalogram_r),kind='cubic')
            func_t = interp2d(t,f,np.abs(scalogram_t),kind='cubic')

            scalogram_z = func_z(t_i,f_i)
            scalogram_r = func_r(t_i,f_i)
            scalogram_t = func_t(t_i,f_i)

            if normalize == 'spec_max':
                scalogram_z /= np.max(scalogram_z)
                scalogram_r /= np.max(scalogram_r)
                scalogram_t /= np.max(scalogram_t)

            #stack images to single 3 x N x M array
            scalogram_3comp = np.array((scalogram_z,scalogram_r,scalogram_t))
            np.save('{}/{}'.format(outdir,outname),np.abs(scalogram_3comp))

            #write labels
            #labels.write('{}, {}\n'.format(outname,e_type))
            labels.write('{}/{}.npy, {}, {}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}\n'.format(outdir,outname,e_type,event_name,dist_here,evlo,evla,evdp,stlo,stla,ps_here,snr_here))

            if plot:
                plt.imshow(scalogram_z)
                plt.savefig('{}/{}.jpg'.format(outdir,outname))
                plt.close()

            #use data augmentation if its the minority class
            #if event.event_type == 'explosion' and augment:

            if ((event.event_type == 'explosion' and augment) or (event.event_type=='earthquake' and dataset_name=='base')) :
                for j in range(0,dset_multiplier):

                    if e_type == 1 and dataset_name == 'base':
                        continue

                    outname = '{}_{:04d}_augmented'.format(dataset_name,n_good+1)
                    n_good += 1

                    data_z = trz.copy().data
                    data_r = trr.copy().data
                    data_t = trr.copy().data

                    noise_z = nrz.copy().data
                    noise_r = nrr.copy().data
                    noise_t = nrt.copy().data

                    noise_level = 1.
                    scale_max = np.random.random() * noise_level
                    noise_z *= scale_max
                    noise_r *= scale_max
                    noise_t *= scale_max

                    #apply a random shift to noise
                    t_shift_max_noise = sig_len
                    ind_shift_max_noise = int(samprate * t_shift_max_noise)
                    ind_shift_noise = np.random.randint(-ind_shift_max_noise,ind_shift_max_noise)
                    noise_z = np.roll(noise_z,ind_shift_noise)
                    noise_r = np.roll(noise_r,ind_shift_noise)
                    noise_t = np.roll(noise_t,ind_shift_noise)

                    data_z = data_z + noise_z[0:npts]
                    data_r = data_r + noise_r[0:npts]
                    data_t = data_t + noise_t[0:npts]

                    #apply a random shift to whole trace
                    #first, undo the intitial shift so it doesn't get shifted twice
                    data_z = np.roll(data_z,-ind_shift)
                    data_r = np.roll(data_r,-ind_shift)
                    data_t = np.roll(data_t,-ind_shift)

                    #next, randomly shift by up to 2 s
                    t_shift_max_2 = 2.0
                    ind_shift_max_2 = int(samprate * t_shift_max_2)
                    ind_shift_2 = np.random.randint(-ind_shift_max_2,ind_shift_max_2)
                    data_z = np.roll(data_z,ind_shift_2)
                    data_r = np.roll(data_r,ind_shift_2)
                    data_t = np.roll(data_t,ind_shift_2)

                    #apply cosine taper to shifted data before scalogram calculation
                    taper = cosine_taper(len(data_z),0.1)
                    data_z *= taper
                    data_r *= taper
                    data_t *= taper

                    scalogram_z_aug = cwt(data_z, dt, 8, f_min, f_max)
                    scalogram_r_aug = cwt(data_r, dt, 8, f_min, f_max)
                    scalogram_t_aug = cwt(data_t, dt, 8, f_min, f_max)

                    func_z_aug = interp2d(t,f,np.abs(scalogram_z_aug),kind='cubic')
                    func_r_aug = interp2d(t,f,np.abs(scalogram_r_aug),kind='cubic')
                    func_t_aug = interp2d(t,f,np.abs(scalogram_t_aug),kind='cubic')

                    scalogram_z_aug = func_z_aug(t_i,f_i)
                    scalogram_r_aug = func_r_aug(t_i,f_i)
                    scalogram_t_aug = func_t_aug(t_i,f_i)

                    if normalize == 'spec_max':
                        scalogram_z_aug /= np.max(scalogram_z_aug)
                        scalogram_r_aug /= np.max(scalogram_r_aug)
                        scalogram_t_aug /= np.max(scalogram_t_aug)

                    #stack images to single 3 x N x M array
                    scalogram_3comp_aug = np.array((scalogram_z_aug,scalogram_r_aug,scalogram_t_aug))
                    np.save('{}/{}'.format(outdir,outname),np.abs(scalogram_3comp_aug))

                    #write labels
                    labels.write('{}/{}.npy, {}, {}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}\n'.format(outdir,outname,e_type,event_name,dist_here,evlo,evla,evdp,stlo,stla,ps_here,snr_here))

                    if plot:
                        plt.imshow(scalogram_z_aug)
                        plt.savefig('{}/{}.jpg'.format(outdir,outname))
                        plt.clf()
                        plt.close()

            #elif event.event_type == 'earthquake':
            #    print('didnt augment this one... its an earthquake')

        else:
            n_bad += 1

        print('good: {}, bad: {}'.format(n_good,n_bad))
        #TODO give reason for why it failed. (snr, dist, psr_max)
