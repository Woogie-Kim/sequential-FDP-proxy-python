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
    x_boundary = length_x * min(nx - x, x -1)
    y_boundary = length_y * min(ny - y, y -1)
    return min(x_boundary, y_boundary)