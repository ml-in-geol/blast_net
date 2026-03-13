import obspy
import pyasdf
import numpy as np
from pathlib import Path
import tempfile
from sys import argv
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

km_per_degree = (2*np.pi*6371.0)/360.0

input_file = argv[1]
ds = pyasdf.ASDFDataSet(input_file)


def get_model_dir():
    repo_root = Path(__file__).resolve().parents[1]
    candidate_dirs = [
        repo_root / 'data' / 'vel_models',
        repo_root / 'data' / 'vel_model',
    ]
    for model_dir in candidate_dirs:
        if model_dir.exists():
            return model_dir
    return candidate_dirs[-1]


def get_region_name(input_path):
    stem = Path(input_path).stem
    if stem.endswith('_processed'):
        return stem[:-len('_processed')]
    if stem.endswith('_raw'):
        return stem[:-len('_raw')]
    return stem


def build_region_model(input_path):
    model_dir = get_model_dir()
    region = get_region_name(input_path)

    model_nd = model_dir / '{}.nd'.format(region)
    if not model_nd.exists():
        model_nd = model_dir / 'enam.nd'

    if not model_nd.exists():
        raise FileNotFoundError('No matching velocity model found for {} and fallback enam.nd is missing'.format(region))

    with tempfile.TemporaryDirectory() as temp_dir:
        normalized_nd = Path(temp_dir) / model_nd.name
        with open(model_nd, 'r') as f_in, open(normalized_nd, 'w') as f_out:
            for line in f_in:
                stripped = line.strip()
                if not stripped:
                    f_out.write(line)
                    continue

                parts = stripped.split()
                try:
                    float(parts[0])
                except ValueError:
                    f_out.write(line)
                    continue

                if len(parts) == 3:
                    f_out.write('{} {} {} 0.0\n'.format(parts[0], parts[1], parts[2]))
                else:
                    f_out.write(line)

        build_taup_model(str(normalized_nd), output_folder=str(model_dir), verbose=False)
    model_npz = model_dir / '{}.npz'.format(model_nd.stem)

    if not model_npz.exists():
        raise FileNotFoundError('Failed to build TauP model {}'.format(model_npz))

    print('using velocity model {}'.format(model_nd))
    return TauPyModel(model=str(model_npz))


tt_mod = build_region_model(input_file)

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
        P_arrs = tt_mod.get_travel_times(source_depth_in_km=event_depth,
                distance_in_degree=dist_degree, phase_list=['p', 'P', 'Pn'])
        S_arrs = tt_mod.get_travel_times(source_depth_in_km=event_depth,
                distance_in_degree=dist_degree, phase_list=['s', 'S', 'Sn'])

        if len(P_arrs) > 0:
            P_time = min(arr.time for arr in P_arrs)
        else:
            print('no P arrivals for dist {} and depth {}'.format(dist_degree,event_depth))
            continue

        if len(S_arrs) > 0:
            S_time = min(arr.time for arr in S_arrs)
        else:
            print('no S arrivals for dist {} and depth {}'.format(dist_degree,event_depth))
            continue

        P_time_dict['{}.{}'.format(net_code,sta_code)] = P_time
        S_time_dict['{}.{}'.format(net_code,sta_code)] = S_time
        P_times.append(P_time)
        S_times.append(S_time)

    ds.add_auxiliary_data(data = np.array(P_times), data_type = 'travel_times', path = 'P_times/{}'.format(origin.time), parameters = P_time_dict)
    ds.add_auxiliary_data(data = np.array(S_times), data_type = 'travel_times', path = 'S_times/{}'.format(origin.time), parameters = S_time_dict)

ds.flush()
ds._close()
ds._ASDFDataSet__file = None
