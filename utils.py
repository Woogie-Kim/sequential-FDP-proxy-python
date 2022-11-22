from scipy import io


def load_matfile(filename, data_type):
    data = io.loadmat(filename)
    return data["PERMX"][0, 0][data_type].transpose().tolist()


def make_permfield(filename, perm):
    with open(filename, 'w') as f:
        f.write('PERMX\n')
        for p in perm:
            try:
                f.write(f'{p[0]}\n')
            except:
                f.write(f'{p}\n')
        f.write('/')
