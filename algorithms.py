import copy
from dlmodels import WPDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from os.path import exists
from multiprocessing import Process
from time import sleep
def check_domination(positions):
    dominated_idx = []
    for idxA, positionA in enumerate(positions):
        for idxB, positionB in enumerate(positions):
            if idxA == idxB:
                continue
            dominate = _filter_method(positionA, positionB)
            if dominate == 1:
                dominated_idx.append(idxB)
    dominated_idx = set(dominated_idx)
    all_idx = set(range(0, len(positions)))

    return list(all_idx.difference(dominated_idx))


def _filter_method(A, B):
    if A.violation > 0 and B.violation > 0:
        dominate = int(A.violation < B.violation)
    elif any([v == 0 for v in [A.violation, B.violation]]) and not all([v == 0 for v in [A.violation, B.violation]]):
        dominate = int(A.violation == 0)
    else:  # consider multi-objectives
        if not isinstance(A.fit, list):
            A_fit, B_fit = [A.fit], [B.fit]
        else:
            A_fit, B_fit = A.fit, B.fit
        dominate = all([a <= b for a, b in zip(A_fit, B_fit)]) \
                   and any([a < b for a, b in zip(A_fit, B_fit)])
    return dominate


class PSO:
    def __init__(self,
                 args,
                 positions,
                 perms):
        self.args = args
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494

        self.positions_all = [positions]
        self.gbest = []
        self.pbest = []
        self.perms = perms

    def evaluate(self, positions, neural_model=None):
        """
        :param positions:
        :param neural_model: pre-trained model
        :return:
        """

        args = self.args

        position_all = []
        predictions_ens = []
        if neural_model:
            for k in self.perms:

                if neural_model.model_name in ['CNN', 'ResNet']:
                    pfs = []
                    for idx, position in enumerate(positions):
                        pf = Process(target=position.frontsim_parallel, args=(idx + 1, position, k))
                        pf.start()
                        pfs.append(pf)
                        if (((idx + 1) % self.args.max_process == 0) and not (idx + 1) == 0) or (idx + 1) == len(
                                positions):
                            for p in pfs: p.join()
                            while not exists(
                                    position.simulation_directory + '/' + position.frs_filename + f'_{idx + 1}.X0001'): sleep(
                                0.1)
                            pfs = []
                        if (idx + 1) == len(positions):
                            for idx in range(len(positions)): positions[idx].frs_result(idx + 1, k)

                    dataset = WPDataset(data=positions, maxtof=args.max_tof, maxP=args.max_pressure,
                                        res_oilsat=args.res_oilsat,nx=args.num_of_x, ny=args.num_of_y, transform=None,
                                        flag_input=args.input_flag)
                elif neural_model.model_name == 'LSTM':
                    dataset = WPDataset(positions, args.production_time, args.dstep, args.tstep, None)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                predictions, _ = neural_model.inference(neural_model.model, dataloader, label_exist=False)
                predictions_ens.append([p for p in predictions])

        else:
            for k in self.perms:
                predictions = []
                ps = []
                for idx, position in enumerate(positions, desc=f'now ecl simulate: '):

                    pe = Process(target=position.eclipse_parallel, args=(idx + 1, position, k))
                    pe.start()
                    ps.append(pe)
                    if (((idx + 1) % self.args.max_process == 0) and not (idx + 1) == 0) or (idx + 1) == len(positions):
                        for p in ps: p.join()
                        while not exists(
                                position.simulation_directory + '/' + position.ecl_filename + f'_{idx + 1}.RSM'): sleep(0.1)
                        ps = []
                    if (idx + 1) == len(positions):
                        for idx in range(len(positions)): positions[idx].ecl_result(idx + 1, positions[idx])
                predictions.append(position.fit)
            predictions_ens.append(predictions)

        predictions = np.mean(np.array(predictions_ens), axis=0).squeeze().tolist()
        for position, pred in zip(positions, predictions):
            position.fit = pred

        return positions

    def update(self, positions):
        positions_next = copy.deepcopy(positions)

        # find a personal best position
        pbest_position = self._find_best_position(positions_next)
        self.pbest.append(pbest_position)

        # update a global best position
        gbest_position = self._get_gbest_position()
        self.gbest.append(gbest_position)

        for position in positions_next:
            position = self._cal_velocity_and_update(position)
            for w in position.wells:
                w._cut_boundaries()

        self.positions_all.append(positions_next)

    def _cal_velocity_and_update(self, position):
        pbest_position = self.pbest[-1]
        gbest_position = self.gbest[-1]

        attributes = position.wells[0].attributes
        w, c1, c2 = self.w, self.c1 * random.random(), self.c2 * random.random()
        for well, well_p, well_g in zip(position.wells, pbest_position.wells, gbest_position.wells):
            for var, elem in attributes.items(): # attr = location, var = ['x', 'y', 'z']
                current = getattr(well, var)
                pbest = getattr(well_p, var)
                gbest = getattr(well_g, var)
                if not current:     # not defined variable type
                    continue
                for e in elem:
                    if current[e]:
                        vel = w * current['velocity'][e] + c1*(pbest[e]-current[e]) + c2*(gbest[e]-current[e])
                        current['velocity'][e] += vel
                        current[e] += current['velocity'][e]

        return position

    def _find_best_position(self, positions):
        dominate = check_domination(positions)
        return positions[dominate[0]]

    def _get_gbest_position(self):
        if not self.gbest:
            return self.pbest[-1]
        elif check_domination([self.gbest[-1], self.pbest[-1]]) == 0:
            return self.gbest[-1]
        else:
            return self.pbest[-1]
