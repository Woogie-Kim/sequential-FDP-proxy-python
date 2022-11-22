import os.path
import pickle

from sampler import DataSampling
from proxymodel import ProxyModel
from optimization import PSO
from utils import load_matfile
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    args.filepath = 'data'
    args.simulation_directory = 'simulation'
    args.save_directory = 'variables'
    args.ecl_filename = '2D_JY_Eclrun'
    args.frs_filename = '2D_JY_Frsrun'
    args.perm_filename = '2D_PERMX'
    args.position_filename = '2D_POSITION'
    args.constraint_filename = '2D_CONSTRAINT'

    args.independent_area = 40  # acre
    args.num_of_x = 60
    args.num_of_y = 60
    args.num_of_z = 1
    args.num_of_max_well = 14
    args.num_of_rigs = 3
    args.num_of_particles = 3
    args.length_of_x = 120
    args.length_of_y = 120
    args.length_of_z = 120
    args.num_of_ensemble = 10

    args.discount_rate = 0.1
    args.observed_term = 30
    args.discount_term = 365
    args.oil_price = 60
    args.injection_cost = -5
    args.disposal_cost = -3
    args.drilling_cost = 0

    args.production_time = 7200
    args.tstep = 30
    args.dstep = 90  # drilling span
    args.streamline_time = 30
    args.max_tof = 10000
    args.max_pressure = 3500

    args.num_of_train_sample = 500
    args.train_ratio = 0.7
    args.validate_ratio = 0.15
    args.train_model_saved_dir = './model/1'
    args.span_of_retrain = 20

    args.ratio_of_infeasible = 0.3
    args.well_type = {'P': 1, 'No': 0, 'I': -1}
    args.type_fix = False
    args.drilling_time_fix = True
    args.location_fix = True
    args.control_fix = True

    args.well_placement_optimization = True
    args.well_operation_optimization = True
    args.simultaneous_optimization = False
    args.well_location_index = None
    args.well_type_real = None
    args.well_placement_wset = [1500, 1500, 5500, 5500]
    args.well_operation_wset = [1500, 2500, 4500, 6000]

    args.perm_mat = './data/PERMX.mat'

    args.parallel = True

    args.cached_dir = './cached'
    args.well_placement_sample_file = 'sample_wp(1).pkl'
    args.well_operation_sample_file = 'sample_wo(1).pkl'
    args.num_of_epochs = 20
    args.batch_size = 150

    args.optimization = True
    args.optimization_algorithm = 'PSO'
    args.num_of_generations = 200

    args.gen_of_retrain = range(args.span_of_retrain, args.num_of_generations + 1, args.span_of_retrain)


    assert args.validate_ratio != 0, 'validate_ratio should be greater than 0'
    assert (1 - args.train_ratio - args.validate_ratio) > 0, '(train_ratio + validate_ratio) should not be 1'
    if args.well_operation_optimization and not args.well_placement_optimization:
        # if you only want to optimize well operation conditions, provide well position settings by yourself
        assert args.well_placement_optimization, 'if you only want to optimize well operation conditions, provide ' \
                                                 'well position settings by yourself. You must set the default well ' \
                                                 'locations for a defined number of wells. ' \
                                                 '- well_location_index, well_type_real'

    if not os.path.exists(args.train_model_saved_dir):
        print('model_saved_dir not exists')
        os.mkdir(args.train_model_saved_dir)

    perm = load_matfile(args.perm_mat, 'original')
    perm_idx = load_matfile(args.perm_mat, 'selected')

    """ 1. Well Placement """
    if args.well_placement_optimization:
        ''' 1.1. make samples to train a proxy model '''
        PlacementSample = DataSampling(args, wset=args.well_placement_wset, well_type=args.well_type,
                                       location_fix=False, type_fix=False, drilling_time_fix=True,
                                       control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                       num_of_wells=args.num_of_max_well)

        if os.path.exists(os.path.join(args.cached_dir, args.well_placement_sample_file)):
            with open(os.path.join(args.cached_dir, args.well_placement_sample_file), 'rb') as f:
                samples_p = pickle.load(f)
        else:
            samples_p = []
            for idx in range(args.num_of_ensemble):
                print(f'ensemble #{idx+1}')
                initial_p = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_train_sample)
                samples_p += PlacementSample.make_train_data(initial_p[idx], perm[perm_idx[0][idx]-1])
            with open(os.path.join(args.cached_dir, args.well_placement_sample_file), 'wb') as f:
                pickle.dump(samples_p, f)

        ''' 1.2. train a proxy model '''
        Model_p = ProxyModel(args, samples_p, model_name='CNN')
        if os.path.exists(f'{args.train_model_saved_dir}/saved_model.pth'):
            Model_p.model.load_state_dict(torch.load(f'{args.train_model_saved_dir}/saved_model.pth'))
        else:
            Model_p.model = Model_p.train_model(samples_p, train_ratio=args.train_ratio, validate_ratio=args.validate_ratio,
                                                saved_dir=args.train_model_saved_dir)

        ''' 1.3. optimization '''
        placement_positions = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)

        WPO = PSO(args, placement_positions, PlacementSample.perm[:, PlacementSample.perm_idx - 1])
        for gen in range(args.num_of_generations):
            placement_positions_evaluated = WPO.evaluate(WPO.positions_all[-1], Model_p)
            WPO.update(placement_positions_evaluated)

            # retrain the proxy model
            if gen in args.gen_of_retrain:
                samples_p += WPO.positions_all[-1] * args.num_of_ensemble
                Model_p.model = Model_p.train_model(samples_p, train_ratio=args.train_ratio,
                                                    validate_ratio=args.validate_ratio,
                                                    saved_dir=args.train_model_saved_dir)
        best_location_index = [w.location['index'] for w in WPO.gbest[-1].wells if w.type['index'] != 0]
        best_type_real = [w.type['index'] for w in WPO.gbest[-1].wells if w.type['index'] != 0]
    else:
        assert any([args.well_location_index, args.well_type_real]), ValueError(
            'The well location index and well type are required for well operation optimization only.')
        best_location_index = args.well_location_index
        best_type_real = args.well_type_real

    """ 2. Well Operation """
    if args.well_operation_optimization:
        ''' 2.1. make samples to train a proxy model '''
        OperationSample = DataSampling(args, wset=args.well_operation_wset, well_type=args.well_type,
                                       type_fix=True, location_fix=True, drilling_time_fix=False,
                                       control_fix=False, num_of_ensemble=args.num_of_ensemble,
                                       num_of_wells=len(best_location_index), violation_check=False)

        if os.path.exists(os.path.join(args.cached_dir, args.well_operation_sample_file)):
            with open(args.cached_file, 'rb') as f:
                samples_o = pickle.load(f)
        else:
            args.drilling_cost = 2E+06
            initial_o = OperationSample.make_candidate_solutions(num_of_candidates=args.num_of_train_sample,
                                                                 location=best_location_index,
                                                                 type_real=best_type_real)
            samples_o = OperationSample.make_train_data(initial_o, [perm[idx[0]-1] for idx in perm_idx], use_frontsim=False)
            with open(os.path.join(args.cached_dir, args.well_operation_sample_file), 'wb') as f:
                pickle.dump(samples_o, f)

        ''' 2.2. train a proxy model '''
        Model_o = ProxyModel(args, samples_o, model_name='LSTM')
        if os.path.exists(f'{args.train_model_saved_dir}/saved_model.pth'):
            Model_o.model.load_state_dict(torch.load(f'{args.train_model_saved_dir}/saved_model.pth'))
        else:
            Model_o.model = Model_o.train_model(samples_o, train_ratio=args.train_ratio, validate_ratio=args.validate_ratio,
                                                saved_dir=args.train_model_saved_dir)

        ''' 2.3. optimization '''
        operation_positions = OperationSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)

        WOO = PSO(args, operation_positions, OperationSample.perm[:, OperationSample.perm_idx - 1])
        for gen in range(args.num_of_generations):
            operation_positions_evaluated = WOO.evaluate(WOO.positions_all[-1], Model_o)
            WOO.update(operation_positions_evaluated)

            # retrain the proxy model
            if gen in args.gen_of_retrain:
                samples_o += WOO.positions_all[-1] * args.num_of_ensemble
                Model_o.model = Model_o.train_model(samples_o, train_ratio=args.train_ratio,
                                                    validate_ratio=args.validate_ratio,
                                                    saved_dir=args.train_model_saved_dir)


if __name__ == "__main__":
    main()
