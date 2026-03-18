import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyasdf

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot six-panel record sections for each processed event.'
    )
    parser.add_argument('input_file', help='Processed ASDF file.')
    parser.add_argument('output_dir', help='Directory for event figure files.')
    parser.add_argument('--snr-threshold', type=float, default=2.0)
    parser.add_argument('--high-band', default='10.0,18.0')
    parser.add_argument('--low-band', default='1.0,10.0')
    parser.add_argument('--scale', type=float, default=4.0)
    parser.add_argument('--xmin', type=float, default=-20.0)
    parser.add_argument('--xmax', type=float, default=120.0)
    parser.add_argument('--dpi', type=int, default=200)
    return parser.parse_args()


def parse_band(text):
    values = [float(value.strip()) for value in text.split(',')]
    if len(values) != 2:
        raise ValueError('Band must have exactly two comma-separated values.')
    return tuple(values)


def get_station_key(stream):
    return '{}.{}'.format(stream[0].stats.network, stream[0].stats.station)


def select_component(stream, component):
    traces = stream.select(channel='*{}'.format(component))
    if len(traces) == 0:
        return None
    return traces[0]


def compute_band_snr(stream, origin_time, p_time, s_time, band):
    filtered = stream.copy()
    filtered.filter(
        'bandpass',
        freqmin=band[0],
        freqmax=band[1],
        corners=2,
        zerophase=True,
    )

    tr_z = select_component(filtered, 'Z')
    tr_r = select_component(filtered, 'R')
    tr_t = select_component(filtered, 'T')
    if tr_z is None or tr_r is None or tr_t is None:
        return None, None

    window_length = (s_time - p_time) * 0.5
    if window_length < 1.0:
        return None, None
    if window_length > 3.0:
        window_length = 3.0

    p_start = origin_time + (p_time - (window_length * 0.05))
    n_start = origin_time - 10.0
    p_end = p_start + window_length
    n_end = n_start + window_length

    p_windows = [
        tr_z.slice(starttime=p_start, endtime=p_end),
        tr_r.slice(starttime=p_start, endtime=p_end),
        tr_t.slice(starttime=p_start, endtime=p_end),
    ]
    n_windows = [
        tr_z.slice(starttime=n_start, endtime=n_end),
        tr_r.slice(starttime=n_start, endtime=n_end),
        tr_t.slice(starttime=n_start, endtime=n_end),
    ]

    if any(window.stats.npts < 2 for window in p_windows + n_windows):
        return None, None

    p_energy = sum(np.mean((window.data * 1.0e9) ** 2) for window in p_windows)
    n_energy = sum(np.mean((window.data * 1.0e9) ** 2) for window in n_windows)
    signal_energy = p_energy - n_energy

    if signal_energy <= 0.0 or n_energy <= 0.0:
        return None, None

    snr = np.sqrt(signal_energy) / np.sqrt(n_energy)
    if not np.isfinite(snr):
        return None, None

    return float(snr), filtered


def collect_band_records(ds, event, band, snr_threshold):
    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    try:
        distances = ds.auxiliary_data.distances.distances[event_name].parameters
        p_times = ds.auxiliary_data.travel_times.P_times[event_name].parameters
        s_times = ds.auxiliary_data.travel_times.S_times[event_name].parameters
    except Exception:
        return []

    station_records = []
    for station in ds.ifilter(ds.q.event == event):
        stream = station.processed.copy()
        if len(stream) == 0:
            continue

        station_key = get_station_key(stream)
        try:
            distance_km = distances[station_key]
            p_time = p_times[station_key]
            s_time = s_times[station_key]
        except KeyError:
            continue

        snr_value, filtered_stream = compute_band_snr(stream, origin.time, p_time, s_time, band)
        if snr_value is None or snr_value < snr_threshold:
            continue

        station_records.append(
            {
                'station_key': station_key,
                'distance_km': distance_km,
                'snr': snr_value,
                'stream': filtered_stream,
            }
        )

    station_records.sort(key=lambda item: item['distance_km'])
    return station_records


def plot_component_section(ax, records, component, origin_time, scale, x_limits, title):
    n_plotted = 0
    for record in records:
        trace = select_component(record['stream'], component)
        if trace is None:
            continue

        amplitude = np.max(np.abs(trace.data))
        if not np.isfinite(amplitude) or amplitude == 0.0:
            continue

        times = trace.times(reftime=origin_time)
        data = (trace.data / amplitude) * scale + record['distance_km']
        ax.plot(times, data, color='k', linewidth=0.5, alpha=0.75)
        n_plotted += 1

    ax.axvline(0.0, color='0.7', linewidth=0.75, linestyle='--')
    ax.set_xlim(x_limits)
    ax.set_title('{} (n={})'.format(title, n_plotted))


def safe_event_filename(event_name):
    return event_name.replace(':', '-').replace('/', '_')


def plot_event_record_sections(
    ds,
    event,
    output_dir,
    snr_threshold=2.0,
    high_band=(10.0, 18.0),
    low_band=(1.0, 10.0),
    scale=4.0,
    x_limits=(-20.0, 120.0),
    dpi=200,
):
    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    high_records = collect_band_records(ds, event, high_band, snr_threshold)
    low_records = collect_band_records(ds, event, low_band, snr_threshold)
    if not high_records and not low_records:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    band_specs = [
        (high_band, high_records, 0),
        (low_band, low_records, 1),
    ]
    components = ['Z', 'R', 'T']

    for band, records, row_index in band_specs:
        for col_index, component in enumerate(components):
            band_label = '{:g}-{:g} Hz'.format(band[0], band[1])
            panel_title = '{} | {}'.format(component, band_label)
            plot_component_section(
                axes[row_index, col_index],
                records,
                component,
                origin.time,
                scale,
                x_limits,
                panel_title,
            )

    for ax in axes[:, 0]:
        ax.set_ylabel('Distance (km)')
    for ax in axes[1, :]:
        ax.set_xlabel('Time Relative to Origin (s)')

    event_type = getattr(event, 'event_type', None) or 'unknown'
    fig.suptitle(
        '{} | type={} | SNR>{:g}'.format(event_name, event_type, snr_threshold),
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    output_path = os.path.join(
        output_dir,
        '{}_record_sections.png'.format(safe_event_filename(event_name)),
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_record_sections(
    input_file,
    output_dir,
    snr_threshold=2.0,
    high_band=(10.0, 18.0),
    low_band=(1.0, 10.0),
    scale=4.0,
    x_limits=(-20.0, 120.0),
    dpi=200,
):
    os.makedirs(output_dir, exist_ok=True)
    ds = pyasdf.ASDFDataSet(input_file, mpi=False, mode='r')

    outputs = []
    for event in ds.events:
        output_path = plot_event_record_sections(
            ds,
            event,
            output_dir,
            snr_threshold=snr_threshold,
            high_band=high_band,
            low_band=low_band,
            scale=scale,
            x_limits=x_limits,
            dpi=dpi,
        )
        if output_path is not None:
            outputs.append(output_path)

    return outputs


def main():
    args = parse_args()
    high_band = parse_band(args.high_band)
    low_band = parse_band(args.low_band)

    outputs = plot_record_sections(
        args.input_file,
        args.output_dir,
        snr_threshold=args.snr_threshold,
        high_band=high_band,
        low_band=low_band,
        scale=args.scale,
        x_limits=(args.xmin, args.xmax),
        dpi=args.dpi,
    )

    print('Wrote {} event figure(s) to {}'.format(len(outputs), os.path.abspath(args.output_dir)))


if __name__ == '__main__':
    main()
