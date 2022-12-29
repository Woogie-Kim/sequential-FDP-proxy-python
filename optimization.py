import algorithms
from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm

class GlobalOpt:
    def __init__(self,
                 args,
                 positions,
                 perms,
                 alg_name='PSO',
                 nn_model=None,
                 sample=None,
                 fine_tune=None,
                 ):
        self.args = args
        self.perms = perms
        self.alg_name = alg_name
        self.nn_model = nn_model
        self.sample = sample
        self.fine_tune = fine_tune

        self.algorithm = getattr(algorithms, alg_name)(args,
                                                       positions,
                                                       perms)

    def iterate(self, num_of_generations):
        args = self.args

        iterbar = tqdm(range(num_of_generations))
        for gen in iterbar:
            positions_evaluated = self.algorithm.evaluate(self.algorithm.positions_all[-1], self.nn_model)
            self.algorithm.update(positions_evaluated)

            iterbar.set_description(f"best fit = {self.algorithm.gbest[-1].fit}")

            if gen in args.gen_of_retrain:
                print(f"now retraining")
                self._retrain(self.algorithm.positions_all[-1])

    def get_solution(self, location=False, type=False, drilling_time=False, control=False):
        best = {}
        if location:
            best['location'] = [w.location['index'] for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if type:
            best['type'] = [w.type['index'] for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if drilling_time:
            best['drilling_time'] = [w.drilling_time['time']
                                     for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if control:
            best['control'] = [(w.well_control['alpha'], w.well_control['beta'], w.well_control['gamma'])
                               for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]

        return best

    def _retrain(self, sample):
        args = self.args
        if self.nn_model:
            self.sample += sample * args.num_of_ensemble
            if self.fine_tune:
                for conv in self.nn_model.layer.parameters():
                    conv.requires_grad = False
                for fc in self.nn_model.fc_layer.parameters():
                    fc.requires_grad = True
            else:
                self.nn_model.model = self.nn_model.train_model(self.sample, train_ratio=args.train_ratio,
                                                            validate_ratio=args.validate_ratio,
                                                            saved_dir=self.nn_model.saved_dir)


