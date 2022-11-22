import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CNNDataset(Dataset):
    def __init__(self, data, maxtof, nx, ny, transform=None):
        self.data = data
        self.maxtof = maxtof
        self.nx, self.ny = nx, ny
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.tof_p, self.tof_i, self.well_config = np.array([]), np.array([]), np.array([])
        maxtof, nx, ny = self.maxtof, self.nx, self.ny

        d = self.data[index]
        well_config = np.zeros(nx * ny)
        tof_p, tof_i = np.array(d.tof['TOF_beg']), np.array(d.tof['TOF_end'])
        for well in d.wells:
            wi = well.location['index']
            wt = well.type['index']
            well_config[wi-1] = wt
            if wt != -1 and tof_i[wi-1] == 0:
                tof_i[wi-1] = maxtof
        tof_p = (tof_p - maxtof / 2) / maxtof
        tof_i = (tof_i - maxtof / 2) / maxtof
        self.tof_p = tof_p.reshape(1, nx, ny)
        self.tof_i = tof_i.reshape(1, nx, ny)
        self.well_config = well_config.reshape(nx, ny).transpose()
        self.well_config = self.well_config.reshape(1, nx, ny)

        self.x = np.concatenate((self.tof_p, self.tof_i, self.well_config), axis=0)
        self.y = np.array(d.fit_norm) if hasattr(d, 'fit_norm') else np.array(0.0)

        if self.transform:
            self.x = self.transform(torch.FloatTensor(self.x))
        return torch.FloatTensor(self.x), torch.FloatTensor(self.y)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding='same'),
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
            nn.Linear(64*3*3, 128),
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

    def _init_weight(self, layer):
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform(layer.weight)


class LSTMDataset(Dataset):
    def __init__(self, data, production_time, drilling_term, production_term, transform=None):
        self.data = data
        self.transform = transform
        self.production_steps = int(production_time/production_term)
        self.drilling_steps = int(production_time/drilling_term)

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
            nn.Linear(hidden_size*sequence_length, output_size*sequence_length),
        )

    def forward(self, x):
        out, _ = self.layer(x)
        out = self.fc_layer(out.contiguous().view(out.shape[0], -1))
        out = out.view(out.shape[0], -1, self.output_size)
        return out


