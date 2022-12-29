from scipy import io
import matplotlib.pyplot as plt
import os
from math import *
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

def calculate_boundary(coord, nx, ny, length_x = 120, length_y = 120):
    '''
    Reservoir boundary까지 거리를 계산하기 위한 메서드
    :param coord: (x,y)로 구성된 좌표 정보
    :param nx: x좌표 격자 수
    :param ny: y좌표 격자 수
    :param length_x: x좌표 격자별 길이
    :param length_y: y좌표 격자별 길이
    :return: reservoir boundary distance
    '''
    x, y = coord
    x_boundary = length_x * min(nx - x, abs(x))
    y_boundary = length_y * min(ny - y, abs(y))
    return min(x_boundary, y_boundary)


def get_regress(Model, args, filename=None, show=None):
    real = Model.reals
    prediction = Model.predictions
    real = [r[0]/1e6 for r in real]
    prediction = [p[0]/1e6 for p in prediction]
    value_range = [0, 1.05 * max(max(real, prediction))]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.scatter(real, prediction, s=6, c='k')
    ax.plot(value_range, value_range, color='r', linewidth=1.2)
    ax.set_aspect('equal', adjustable='box')
    plt.title(fr"R$^{2}$: {Model.metric['r2_score'][0]:.4f}", fontweight='bold', )
    plt.xlabel('True NPV (MM$)',fontname='Times New Roman')
    plt.ylabel('Predicted NPV (MM$)', fontname='Times New Roman')
    plt.xlim(value_range)
    plt.ylim(value_range)
    plt.locator_params(nbins=value_range[-1]//100)


    if filename:
        file_path = os.path.join(args.train_model_saved_dir, args.train_model_figure_saved_dir)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        plt.savefig(os.path.join(file_path, filename)+'.png', facecolor='white')
    if show:
        plt.show()
    plt.close(fig)