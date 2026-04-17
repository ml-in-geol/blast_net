import argparse
import math

import numpy as np
import pyasdf


def parse_args():
    parser = argparse.ArgumentParser(description='Compute station-level P/S ratios and SNR.')
    parser.add_argument('input_file')
    parser.add_argument('--fmin', type=float, default=10.0)
    parser.add_argument('--fmax', type=float, default=18.0)
    parser.add_argument(
        '--window-mode',
        choices=['original', 'test1', 'test2', 'test3'],
        default='original',
        help='Windowing experiment to use for the P and S energy windows.',
    )
    parser.add_argument(
        '--arrival-pad-frac',
        type=float,
        default=0.05,
        help='Start each signal window slightly before the arrival by this fraction of the window length.',
    )
    parser.add_argument(
        '--noise-offset',
        type=float,
        default=10.0,
        help='Start the noise window this many seconds before the origin time.',
    )
    return parser.parse_args()


def get_band_key(fmin, fmax, window_mode):
    base_key = 'f_{:2.2f}_{:2.2f}'.format(fmin, fmax)
    if window_mode == 'original':
        return base_key
    return '{}__{}'.format(base_key, window_mode)


def compute_window_length(p_time, s_time, window_mode):
    separation = s_time - p_time
    if separation <= 0.0:
        return None

    # Original parameters retained for provenance:
    # W = 0.5 * (S_time - P_time)
    # if W < 1.0: continue
    # elif W > 3.0: W = 3.0
    if window_mode == 'original':
        window_length = separation * 0.5
        if window_length < 1.0:
            return None
        if window_length > 3.0:
            window_length = 3.0
        return window_length

    # Test 1: wider dynamic windows to absorb source-location uncertainty.
    # W = 0.75 * (S_time - P_time), clipped to [1.0, 5.0]
    if window_mode == 'test1':
        window_length = separation * 0.75
        window_length = max(1.0, min(window_length, 5.0))
        return window_length

    # Test 2: fixed 3 s P and S windows.
    if window_mode == 'test2':
        return 3.0

    # Test 3: fixed 5 s P and S windows.
    if window_mode == 'test3':
        return 5.0

    raise ValueError('Unsupported window_mode {}'.format(window_mode))


def safe_delete_band(ds, event_name, band_key):
    try:
        del ds.auxiliary_data.PS_ratios[event_name][band_key]
    except Exception:
        pass
    try:
        del ds.auxiliary_data.SNR[event_name][band_key]
    except Exception:
        pass


def main():
    args = parse_args()
    ds = pyasdf.ASDFDataSet(args.input_file)
    band_key = get_band_key(args.fmin, args.fmax, args.window_mode)
    total_events = len(ds.events)

    for i_event, event in enumerate(ds.events, start=1):
        origin = event.preferred_origin() or event.origins[0]
        event_name = '{}'.format(origin.time)
        print(
            'working on event {}/{} ({}) [{}]'.format(
                i_event, total_events, event_name, args.window_mode
            )
        )

        ad1 = ds.auxiliary_data.distances.distances[event_name]
        distance_dict = ad1.parameters

        ad2 = ds.auxiliary_data.travel_times.P_times[event_name]
        p_time_dict = ad2.parameters

        ad3 = ds.auxiliary_data.travel_times.S_times[event_name]
        s_time_dict = ad3.parameters

        snr_dict = {}
        ps_ratio_dict = {}
        snrs = []
        ps_ratios = []

        for station in ds.ifilter(ds.q.event == event):
            seis = station.processed.copy()
            if len(seis) == 0:
                continue

            net_code = seis[0].stats.network
            sta_code = seis[0].stats.station
            station_key = '{}.{}'.format(net_code, sta_code)

            _dist = distance_dict.get(station_key)
            try:
                p_time = p_time_dict[station_key]
                s_time = s_time_dict[station_key]
            except Exception:
                print('No P or S time found for {} {}'.format(net_code, sta_code))
                continue

            seis.filter(
                'bandpass',
                freqmin=args.fmin,
                freqmax=args.fmax,
                corners=2,
                zerophase=True,
            )

            components = [tr.stats.channel[-1] for tr in seis]
            if 'R' not in components or 'T' not in components or 'Z' not in components:
                print('**************************************')
                print('Z, R, and T components not available')
                print(seis)
                print('**************************************')
                continue

            tr_z = seis.select(channel='*HZ')[0]
            tr_r = seis.select(channel='*HR')[0]
            tr_t = seis.select(channel='*HT')[0]

            window_length = compute_window_length(p_time, s_time, args.window_mode)
            if window_length is None:
                continue

            pad = window_length * args.arrival_pad_frac
            p_start = origin.time + (p_time - pad)
            s_start = origin.time + (s_time - pad)
            n_start = origin.time - args.noise_offset
            p_end = p_start + window_length
            s_end = s_start + window_length
            n_end = n_start + window_length

            p_win_z = tr_z.slice(starttime=p_start, endtime=p_end)
            p_win_r = tr_r.slice(starttime=p_start, endtime=p_end)
            p_win_t = tr_t.slice(starttime=p_start, endtime=p_end)
            s_win_z = tr_z.slice(starttime=s_start, endtime=s_end)
            s_win_r = tr_r.slice(starttime=s_start, endtime=s_end)
            s_win_t = tr_t.slice(starttime=s_start, endtime=s_end)
            n_win_z = tr_z.slice(starttime=n_start, endtime=n_end)
            n_win_r = tr_r.slice(starttime=n_start, endtime=n_end)
            n_win_t = tr_t.slice(starttime=n_start, endtime=n_end)

            p_z = np.mean((p_win_z.data * 1e9) ** 2)
            p_r = np.mean((p_win_r.data * 1e9) ** 2)
            p_t = np.mean((p_win_t.data * 1e9) ** 2)
            s_z = np.mean((s_win_z.data * 1e9) ** 2)
            s_r = np.mean((s_win_r.data * 1e9) ** 2)
            s_t = np.mean((s_win_t.data * 1e9) ** 2)
            n_z = np.mean((n_win_z.data * 1e9) ** 2)
            n_r = np.mean((n_win_r.data * 1e9) ** 2)
            n_t = np.mean((n_win_t.data * 1e9) ** 2)

            p_signal = (p_z + p_r + p_t) - (n_z + n_r + n_t)
            s_signal = (s_z + s_r + s_t) - (n_z + n_r + n_t)
            noise_energy = n_z + n_r + n_t

            p_sum = np.sqrt(p_signal)
            s_sum = np.sqrt(s_signal)
            n_sum = np.sqrt(noise_energy)

            ps_ratio = p_sum / s_sum
            snr = p_sum / n_sum

            ps_ratios.append(ps_ratio)
            snrs.append(snr)
            ps_ratio_dict[station_key] = ps_ratio
            snr_dict[station_key] = snr

        safe_delete_band(ds, event_name, band_key)
        ds.add_auxiliary_data(
            data=np.array(ps_ratios),
            data_type='PS_ratios',
            path='{}/{}'.format(event_name, band_key),
            parameters=ps_ratio_dict,
        )
        ds.add_auxiliary_data(
            data=np.array(snrs),
            data_type='SNR',
            path='{}/{}'.format(event_name, band_key),
            parameters=snr_dict,
        )

    ds.flush()
    ds._close()
    ds._ASDFDataSet__file = None


if __name__ == '__main__':
    main()
