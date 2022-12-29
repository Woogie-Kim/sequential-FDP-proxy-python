import torch.optim
import numpy as np
import os
from tqdm import tqdm_notebook as tqdm
from sampler import DataSampling
from dlmodels_modified import WPDataset, WODataset, CNN, LSTM, ResNet18
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


class ProxyModel:
    def __init__(self,
                 args,
                 positions,
                 model_name='CNN'):
        self.model_name = model_name
        if model_name == 'CNN':
            self.model = CNN(args=args)
        elif model_name == 'ResNet':
            self.model = ResNet18(args=args)
        elif model_name == 'LSTM':
            self.model = LSTM(
                            input_size=positions[0].num_of_wells,
                            hidden_size=100,
                            output_size=3,
                            sequence_length=len(positions[0].wells[0].control),
                            num_layers=2)
        self.args = args
        self.saved_dir = f'{args.train_model_saved_dir}/{model_name}'

        self.data_sampling = DataSampling(args)
        self.perm = self.data_sampling.perm

        self.MaxTOF = args.max_tof
        self.MaxP = args.max_pressure

        self.positions = positions

        self.fit_mean = np.mean(np.array([d.fit for d in positions]))
        self.fit_std = np.std(np.array([d.fit for d in positions]))
        for position in positions:
            position.fit_norm = (position.fit - self.fit_mean) / self.fit_std

        self.metric = {"r2_score": [], "MAPE": []}

    def preprocess(self, data, model_name):
        def __merge_schedule_control__(schedule, control):
            return [s * c for s, c in zip(schedule, control)]

        args = self.args

        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        if model_name == 'LSTM':
            production_time = args.production_time
            production_steps = int(production_time / args.tstep)
            drilling_steps = int(production_time / args.dstep)

            # normalize input data
            data_input = []
            for d in data:
                scheduled_control = [__merge_schedule_control__(well.schedule, well.control) for well in d.wells]
                data_input.append(np.array(scheduled_control).reshape(-1, ))
            data_input = np.array(data_input)
            scaler_input.fit(data_input)
            data_input_norm = scaler_input.transform(data_input)

            input_preprocessed = []
            for d, norm in zip(data, data_input_norm):
                norm = norm.reshape(len(d.wells), -1)
                input_preprocessed.append(norm)
                for well, data_input in zip(d.wells, norm):
                    well.control_norm = data_input.tolist()

            # normalize output data
            data_output = []
            for d in data:
                prod_all = np.array(d.prod_data.filter(regex='T_discounted').dropna()).transpose()
                time_index = [idx for idx in range(0, production_steps, int(production_steps / drilling_steps))]
                prod_all = prod_all[:, time_index].reshape(-1, )
                data_output.append(prod_all)
            data_output = np.array(data_output)
            scaler_output.fit(data_output)
            data_output_norm = scaler_output.transform(data_output)

            output_preprocessed = []
            for d, norm in zip(data, data_output_norm):
                norm = norm.reshape(3, -1)
                d.prod_data_norm = norm.tolist()
                output_preprocessed.append(norm)

        elif model_name in ['CNN', 'ResNet']:
            data_output = [d.fit for d in data]
            data_output = np.array(data_output).reshape(-1, 1)
            scaler_output.fit(data_output)
            data_output_norm = scaler_output.transform(data_output)
            for d, norm in zip(data, data_output_norm):
                d.fit_norm = norm[0]

        self.scaler = scaler_output

        return data

    def make_dataloader(self, data, train_ratio, validate_ratio):
        args = self.args

        test_ratio = 1 - train_ratio - validate_ratio

        data = self.preprocess(data, model_name=self.model_name)
        if self.model_name in ['CNN', 'ResNet']:
            dataset = WPDataset(data=data, maxtof=args.max_tof, maxP=args.max_pressure, res_oilsat= args.res_oilsat,
                                nx=args.num_of_x, ny=args.num_of_y, transform=None, flag_input=args.input_flag)
        elif self.model_name in ['LSTM']:
            dataset = WODataset(data, args.production_time, args.dstep, args.tstep, None)
        else:
            NotImplementedError('Model not supported')

        train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                       [int(len(dataset) * ratio) for
                                                                        ratio in
                                                                        [train_ratio,
                                                                         validate_ratio,
                                                                         test_ratio]])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader

    def train_model(self, data, train_ratio=0.7, validate_ratio=0.15, saved_dir='./model', saved_model='saved_model'):

        train_dataloader, valid_dataloader, test_dataloader = \
            self.make_dataloader(data, train_ratio=train_ratio, validate_ratio=validate_ratio)


        self.model = self.train(self.model, train_dataloader, valid_dataloader, test_dataloader, saved_dir, saved_model)

        return self.model

    def train(self, model, train_dataloader, valid_dataloader, test_dataloader, saved_dir='./model', saved_model='saved_model'):

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        min_valid_loss = np.inf

        eps = 1e-7
        iter_bar = tqdm(range(self.args.num_of_epochs))
        for epoch in iter_bar:
            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                x, y = batch

                optimizer.zero_grad()

                pred = model(x)

                loss = torch.sqrt(criterion(pred.squeeze(), y) + eps)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            iter_bar.set_description(f"epoch: {epoch} - loss: {train_loss / len(train_dataloader)} ")

            valid_loss = 0.0
            model.eval()  # Optional when not using Model Specific layer
            for batch in valid_dataloader:
                x_v, y_v = batch

                target = model(x_v)
                loss = torch.sqrt(criterion(target.squeeze(), y_v) + eps)
                valid_loss += loss.item()

            valid_loss /= len(valid_dataloader)
            print(
                f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t '
                f'Validation Loss: {valid_loss}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                if not os.path.exists(saved_dir):
                    os.mkdir(saved_dir)
                torch.save(model.state_dict(), f'{saved_dir}/{saved_model}.pth')

        print(f'Now test to test_dataset')
        model.load_state_dict(torch.load(f'{saved_dir}/{saved_model}.pth'))

        self.inference(model, test_dataloader)

        return model

    def inference(self, model, dataloader, label_exist=True):
        model.eval()

        predictions = []
        reals = []
        for batch in dataloader:
            x_t, y_t = batch

            target = model(x_t)
            if target.dim() > 2:
                target = target.view(target.shape[0], -1)
                y_t = y_t.view(y_t.shape[0], -1)
            prediction = self.scaler.inverse_transform(target.detach().numpy())
            if y_t.dim() == 1:
                y_t = y_t.reshape(-1, 1)
            real = self.scaler.inverse_transform(y_t)
            predictions.extend(prediction)
            reals.extend(real)

        if label_exist:
            self.predictions = predictions
            self.reals = reals
            self.metric['r2_score'].append(r2_score(reals, predictions))
            self.metric['MAPE'].append(100 * mean_absolute_percentage_error(reals, predictions))
            print(f"R_2: {self.metric['r2_score'][0]:.4f}")
            print(f"MAPE: {self.metric['MAPE'][0]:.1f}%")
        return predictions, reals