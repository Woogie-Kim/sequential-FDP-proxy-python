from tqdm import tqdm_notebook as tqdm
from position import PositionExample
from utils import load_matfile
from multiprocessing import Process

class DataExample:
    def __init__(self, position, fitness, violation, matrix, time, tof, pressure):
        self.position = position
        self.fitness = fitness
        self.violation = violation
        self.matrix = matrix
        self.time = time
        self.tof = tof
        self.pressure = pressure
        self.positions = None


class DataSampling:
    def __init__(self,
                 args,
                 ratio_of_infeasible=0.3,
                 wset=None,
                 location_fix=False,
                 type_fix=False,
                 drilling_time_fix=False,
                 control_fix=False,
                 well_type=None,
                 violation=None,
                 violation_check=True,
                 num_of_ensemble=None,
                 positions=None,
                 num_of_wells=None,
                 ):
        self.args = args
        self.perm = load_matfile(args.perm_mat, 'original')
        self.perm_idx = load_matfile(args.perm_mat, 'selected')
        self.ratio_of_infeasible = ratio_of_infeasible
        if not wset:
            wset = [1500, 3000, 4000, 5500]     # default - Prod: 1500~3000 psi, Inj: 4000~5500 psi
        self.wset = wset
        self.type = type
        if not well_type:
            well_type = [-1, 0, 1]  # default - consider all type (-1: Inj. 0: No well, 1: Prod.)
        self.well_type = well_type
        self.location_fix = location_fix
        self.type_fix = type_fix
        self.drilling_time_fix = drilling_time_fix
        self.control_fix = control_fix
        self.violation = violation
        self.violation_check = violation_check
        self.num_of_ensemble = num_of_ensemble
        self.num_of_wells = num_of_wells

        self.positions = positions

    def make_train_data(self, positions, perms, use_eclipse=True, use_frontsim=True):
        # train_data = []
        # for i in range(self.num_of_ensemble):
        #     perm = self.perm[:, self.perm_idx[:,i]-1]
        for idx, position in enumerate(tqdm(positions, desc=f'now simulate: ')):
            if use_eclipse:
                position.eclipse(idx+1, position, perms)
            if use_frontsim:
                position.frontsim(idx+1, position, perms)

        return positions


    def make_train_data_parallel(self, positions, perms, use_eclipse=True, use_frontsim=True):
        # train_data = []
        # for i in range(self.num_of_ensemble):
        #     perm = self.perm[:, self.perm_idx[:,i]-1]
        ps1 = []
        for idx, position in enumerate(tqdm(positions, desc=f'now Eclipse simulate: ')):
            if use_eclipse:
                pe = Process(target=position.eclipse, args=(idx+1, position, perms))
            ps1.append(pe)
            pe.start()

        for p in ps1:
            p.join()

        ps2 = []
        for idx, position in enumerate(tqdm(positions, desc=f'now Frontsim simulate: ')):
            if use_frontsim:
                pf = Process(target=position.frontsim, args=(idx+1, position, perms))
            ps2.append(pf)
            pf.start()

        for p in ps2:
            p.join()

        return positions
    def make_candidate_solutions(self, num_of_candidates, location=None, type_real=None,
                                 drilling_time=None, control=None):
        """
        :param num_of_candidates: number of candidate solutions
        :param ratio_of_infeasible: infeasible means not satisfying defined constraints
        :param well_type: 1: "production", -1: "injection", 0: "no well".
                for consider all well type, well_type = [-1,0,1]
        :param type_fix: if type_fix True, then well_type must be defined
        :return: randomly initialized candidate solutions
        """

        positions, violations = [], []
        for _ in tqdm(range(num_of_candidates), desc="Please wait.."):
            P = PositionExample(self.args, wset=self.wset, well_type=self.well_type, type_fix=self.type_fix,
                                location_fix=self.location_fix, drilling_time_fix=self.drilling_time_fix,
                                control_fix=self.control_fix, violation_check=self.violation_check,
                                num_of_wells=self.num_of_wells)
            P.initialize(ratio_of_infeasible=self.ratio_of_infeasible, location=location, type_real=type_real,
                         drilling_time=drilling_time, control=control)

            positions.append(P)

        return positions



