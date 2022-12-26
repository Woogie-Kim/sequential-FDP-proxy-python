import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import calculate_boundary

class WPDataset(Dataset):
    '''
    :param data: list of initialized and simulated position samples; samples_p
    :param maxtof: maximum tof for distinguishing what grid cells drainaged
    :param maxP: initial maximum pressure
    :param nx: size of x-coordinate
    :param ny: size of y-coordinate
    :param transform: if you want to assign this argument, type nn.Compose([torchvision.transforms.ToTensor()])
    :param flag_input: select type of inputs
    :return: torch.FloatTensor(self.x)
    :return: torch.FloatTensor(self.y)
    '''
    def __init__(self, data, maxtof, maxP, res_oilsat, nx, ny, transform=None, flag_input=None):
        self.data = data
        self.maxtof = maxtof
        self.maxP = maxP
        self.res_oilsat = res_oilsat
        self.nx, self.ny = nx, ny
        self.transform = transform
        self.flag_input = flag_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ''' Input_comb
        딕셔너리 자료형으로 CNN의 input을 선택적으로 변환가능하도록 하였음
        Input_comb = {
        'TOFP': TOFP,
        'TOFI': TOFI,
        'Pressure': pressure,
        'ResPressure': maximum pressure - pressure,
        'Sat': oil saturation,
        'Config': well configuration,
        'Perm': permeability,
        'LogPerm': ln(permeability),
        'Bekzad': (oil saturation - residual oil saturation) * (Pressure - Min.BHP) * ln(permeability) * porosity * ln(grid_boundary)
        'Jang': TOFP * TOFI * oil saturation * permeability
        }
        '''
        self.tof_p, self.tof_i, self.well_config = np.array([]), np.array([]), np.array([])
        maxtof, maxP, nx, ny = self.maxtof, self.maxP, self.nx, self.ny

        d = self.data[index]
        well_config = np.zeros(nx * ny)
        tof_p, tof_i = np.array(d.tof['TOF_beg']), np.array(d.tof['TOF_end'])
        pressure, swat = np.array(d.Dynamic['Pressure']), np.array(d.Dynamic['Swat'])
        perm = np.array(d.perm)

        for well in d.wells:
            wi = well.location['index']
            wt = well.type['index']
            well_config[wi - 1] = wt
            if wt != -1 and tof_i[wi - 1] == 0:
                tof_i[wi - 1] = maxtof
        tof_p = (tof_p - maxtof / 2) / maxtof
        tof_i = (tof_i - maxtof / 2) / maxtof
        self.tof_p = tof_p.reshape(1, nx, ny)
        self.tof_i = tof_i.reshape(1, nx, ny)
        self.well_config = well_config.reshape(nx, ny).transpose()
        self.well_config = self.well_config.reshape(1, nx, ny)
        pressure_res = maxP - pressure
        self.Pressure = pressure.reshape(1, nx, ny)
        self.Pressure_res = pressure_res.reshape(1, nx, ny)
        self.Sat = 1 - swat.reshape(1, nx, ny)
        self.perm = perm.reshape(1, nx, ny)

        input_comb = {'TOFP': self.tof_p, 'TOFI': self.tof_i, 'Pressure': self.Pressure, 'ResPressure': self.Pressure_res, 'Sat': self.Sat, 'Config': self.well_config, 'Perm': self.perm, 'LogPerm': np.log(self.perm)}
        self.Input_comb = input_comb

        ''' new parameter 1 => Bekzad et.al., 2020
        # grid_boundary: (1,nx,ny) 사이즈의 격자별 boundary에서 떨어진 거리
        # Centeroid로 계산하였음. 즉, 저류층 경계면에 닿아있는 cell이라도 0이 아님.
        '''
        grid_boundary = np.array([calculate_boundary((x-0.5,y-0.5), nx, ny) for x in range(1,nx + 1) for y in range(1, ny + 1)])
        self.grid_boundary = grid_boundary.reshape(1, nx, ny)
        self.Input_comb['Bekzad'] = (self.Input_comb['Sat'] - self.res_oilsat) * (self.Input_comb['Pressure'] - 1500) \
                                    * (self.Input_comb['LogPerm']) * 0.2 * np.log(self.grid_boundary)

        ''' new parameter 2 => 장민수, 2016 (정규화 X)
        # 정규화 없이 TOPF, TOPFI, 오일포화도, 유체투과율을 아다마르 곱 연산한 것
        '''
        self.Input_comb['Jang'] = self.Input_comb['TOFP'] * self.Input_comb['TOFI'] * self.Input_comb['Sat'] \
                                  * self.Input_comb['Perm']

        '''
        flag_input을 설정하면(e.g. ['TOFI', 'TOFP', 'Perm']), Input_comb의 key로 작용하여 설정한 파라미터들이 CNN의 input으로 사용됨
        default parameter는 None으로, 따로 설정하지 않으면 원래처럼 TOFP, TOFI, Well configuration이 input으로 구성됨
        '''
        if self.flag_input:
            input_flaged = tuple(self.Input_comb[flag] for flag in self.flag_input)
            self.input_flaged = input_flaged
            self.x = np.concatenate(self.input_flaged, axis=0)
        elif self.flag_input==None:
            self.x = np.concatenate((self.tof_p, self.tof_i, self.well_config), axis=0)

        self.y = np.array(d.fit_norm) if hasattr(d, 'fit_norm') else np.array(0.0)

        if self.transform:
            self.x = self.transform(torch.FloatTensor(self.x))
        return torch.FloatTensor(self.x), torch.FloatTensor(self.y)

class CNN(nn.Module):
    # argument로 args를 받아와, input_flag가 설정되어 있으면 그 수에 맞게 채널 수를 조정해주고, 없으면 기본값인 3으로 설정됨
    def __init__(self, args):
        super().__init__()
        if args.input_flag: self.num_of_channels = len(args.input_flag)
        else: self.num_of_channels = 3

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.num_of_channels, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(stride=2, kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(stride=2, kernel_size=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(stride=2, kernel_size=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(stride=2, kernel_size=(2, 2))
        )

        self.layer.apply(self._init_weight)

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.layer(x)
        out = torch.nn.Flatten()(out)
        out = self.fc_layer(out)
        return out

    def _init_weight(self, layer, init_type="He"):
        if isinstance(layer, nn.Conv2d):
            if init_type == "Xavier":
                torch.nn.init.xavier_uniform(layer.weight)
            elif init_type == "He":
                torch.nn.init.kaiming_uniform_(layer.weight)


# https://cryptosalamander.tistory.com/156
class BasicBlock(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    # argument로 args를 받아와, input_flag가 설정되어 있으면 그 수에 맞게 채널 수를 조정해주고, 없으면 기본값인 3으로 설정됨
    def __init__(self, block, num_blocks, num_classes=10, args=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if args.input_flag: self.num_of_channels = len(args.input_flag)
        else: self.num_of_channels = 3

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        self.conv1 = nn.Conv2d(self.num_of_channels, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(args=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1, args=args)


def ResNet34(args=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1, args=args)


class WODataset(Dataset):
    def __init__(self, data, production_time, drilling_term, production_term, transform=None):
        self.data = data
        self.transform = transform
        self.production_steps = int(production_time / production_term)
        self.drilling_steps = int(production_time / drilling_term)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        x, y = [], []
        for well in d.wells:
            x.append(well.control_norm)
        for prod in d.prod_data_norm:
            y.append(prod)
            # y.append([prod[idx] for idx in range(0, self.production_steps, int(self.production_steps/self.drilling_steps))])
        self.x = np.array(x).transpose()
        # self.y = np.array(d.fit_norm) if hasattr(d, 'fit_norm') else np.array(0.0)
        self.y = np.array(y).transpose()

        if self.transform:
            self.x = self.transform(torch.FloatTensor(self.x))
        return torch.FloatTensor(self.x), torch.FloatTensor(self.y)


class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 sequence_length,
                 num_layers):
        super().__init__()

        self.sequence_length = sequence_length
        self.output_size = output_size

        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, output_size * sequence_length),
        )

    def forward(self, x):
        out, _ = self.layer(x)
        out = self.fc_layer(out.contiguous().view(out.shape[0], -1))
        out = out.view(out.shape[0], -1, self.output_size)
        return out
