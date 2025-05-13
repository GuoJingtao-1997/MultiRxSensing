import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from collections import defaultdict

__all__ = [
    "setup_seed",
    "Accumulator",
    "read_npz_data",
    "CsiLabeled",
    "draw_matrix_acc",
    "get_mean_and_std",
]


class DataProcessor:
    def __init__(self, args):
        self.args = args

    def process_data(self, mode="stop", load_test_only=False, load_train_only=False):
        # self.max_train_size = 0
        # test_dates_list = [self.args.test_date]

        self.data_by_date = {}
        self.data_by_status = {}

        if not load_test_only:

            self.train_dates_list = (
                self.args.train_date.split("_")
                if self.args.concate_date
                else [self.args.train_date]
            )

            for i, date in enumerate(self.train_dates_list):
                if date == "1031":
                    mode = "stop"
                self.read_data(
                    date, mode=mode, append=(i > 0), data_type="train", type="src"
                )
                # self.read_data(date, mode=mode, append=(i > 0), data_type="val")
                # self.read_data(date, mode=mode, append=(i > 0), data_type="test")

                # if self.max_train_size < self.data_by_date[date]["size"]:
                #     self.max_train_size = self.data_by_date[date]["size"]

                # self.concatenate_all_data()

                # self.get_data(
                #     date,
                #     "end",
                #     self.all_end_data,
                #     self.all_end_label,
                #     self.all_end_rssi,
                # )
                # self.get_data(
                #     date,
                #     "head",
                #     self.all_head_data,
                #     self.all_head_label,
                #     self.all_head_rssi,
                # )
            # self.concatenate_all_data()
            # self.concatenate_train_all_data()

        if not load_train_only:

            self.test_dates_list = (
                self.args.test_date.split("_")
                if self.args.concate_date
                else [self.args.test_date]
            )

            for i, date in enumerate(self.test_dates_list):
                if date == "1031":
                    mode = "stop"
                if date not in self.train_dates_list:
                    self.read_data(
                        date, mode=mode, append=(i > 0), data_type="train", type="tag"
                    )
                self.read_data(date, mode=mode, append=(i > 0), data_type="val")
                self.read_data(date, mode=mode, append=(i > 0), data_type="test")

            # self.concatenate_all_data()

            # self.get_data(
            #     date,
            #     "end",
            #     self.all_end_data,
            #     self.all_end_label,
            #     self.all_end_rssi,
            # )
            # self.get_data(
            #     date,
            #     "head",
            #     self.all_head_data,
            #     self.all_head_label,
            #     self.all_head_rssi,
            # )

            self.concatenate_test_all_data()

    def read_data(self, date, mode, append=False, data_type="train", type="src"):
        # if mode == "both":
        #     end_data, end_label, end_rssi, end_pos_label = self.combine_data(
        #         date, data_type, "end"
        #     )
        #     head_data, head_label, head_rssi, head_pos_label = self.combine_data(
        #         date, data_type, "head"
        #     )
        # else:
        #     end_data, end_label, end_rssi, end_pos_label = self.read_aug_data(
        #         date, data_type, "end", mode
        #     )
        #     head_data, head_label, head_rssi, head_pos_label = self.read_aug_data(
        #         date, data_type, "head", mode
        #     )

        if mode == "both":
            end_data, end_label, end_rssi = self.combine_data(date, data_type, "end")
            head_data, head_label, head_rssi = self.combine_data(
                date, data_type, "head"
            )
        else:
            end_data, end_label, end_rssi = self.read_aug_data(
                date, data_type, "end", mode
            )
            head_data, head_label, head_rssi = self.read_aug_data(
                date, data_type, "head", mode
            )

        # pad_value = 0
        # if date != "815" and end_pos_label.shape[-1] < int(self.args.passenger):
        #     right_pad = np.full(
        #         (
        #             end_pos_label.shape[0],
        #             int(self.args.passenger) - end_pos_label.shape[-1],
        #         ),
        #         pad_value,
        #     )
        #     end_pos_label = np.concatenate([end_pos_label, right_pad], axis=1)
        #     head_pos_label = np.concatenate([head_pos_label, right_pad], axis=1)

        if append:
            self.append_end_data(data_type, end_data, end_label, end_rssi)
            self.append_head_data(data_type, head_data, head_label, head_rssi)

        else:
            if data_type == "train":
                self.train_end_data = end_data
                self.train_head_data = head_data
                self.train_end_rssi = end_rssi
                self.train_head_rssi = head_rssi
                self.train_end_label = end_label
                self.train_head_label = head_label
                # self.train_end_pos_label = end_pos_label
                # self.train_head_pos_label = head_pos_label

            elif data_type == "val":
                self.val_end_data = end_data
                self.val_head_data = head_data
                self.val_end_rssi = end_rssi
                self.val_head_rssi = head_rssi
                self.val_end_label = end_label
                self.val_head_label = head_label
                # self.val_end_pos_label = end_pos_label
                # self.val_head_pos_label = head_pos_label
            else:
                self.all_test_end_data = end_data
                self.all_test_head_data = head_data
                self.all_test_end_rssi = end_rssi
                self.all_test_head_rssi = head_rssi
                self.all_test_end_label = end_label
                self.all_test_head_label = head_label
                # self.all_test_end_pos_label = end_pos_label
                # self.all_test_head_pos_label = head_pos_label

        if data_type == "train":

            self.get_adv_data(
                type,
                "end",
                self.train_end_data,
                self.train_end_label,
                self.train_end_rssi,
            )

            self.get_adv_data(
                type,
                "head",
                self.train_head_data,
                self.train_head_label,
                self.train_head_rssi,
            )

        # self.get_data(
        #     date,
        #     "end",
        #     end_data,
        #     end_label,
        #     end_rssi,
        # )

        # self.get_data(
        #     date,
        #     "head",
        #     head_data,
        #     head_label,
        #     head_rssi,
        # )

        self.print_shapes(f"{data_type}_{date}_end", end_data, end_label, end_rssi)
        self.print_shapes(f"{data_type}_{date}_head", head_data, head_label, head_rssi)

    def append_end_data(self, type, data, label, rssi):
        if type == "train":
            self.train_end_data = np.concatenate([self.train_end_data, data])
            self.train_end_label = np.concatenate([self.train_end_label, label])
            self.train_end_rssi = np.concatenate([self.train_end_rssi, rssi])
        elif type == "val":
            self.val_end_data = np.concatenate([self.val_end_data, data])
            self.val_end_label = np.concatenate([self.val_end_label, label])
            self.val_end_rssi = np.concatenate([self.val_end_rssi, rssi])
        else:
            self.all_test_end_data = np.concatenate([self.all_test_end_data, data])
            self.all_test_end_label = np.concatenate([self.all_test_end_label, label])
            self.all_test_end_rssi = np.concatenate([self.all_test_end_rssi, rssi])

    def append_head_data(self, type, data, label, rssi):
        if type == "train":
            self.train_head_data = np.concatenate([self.train_head_data, data])
            self.train_head_label = np.concatenate([self.train_head_label, label])
            self.train_head_rssi = np.concatenate([self.train_head_rssi, rssi])
        elif type == "val":
            self.val_head_data = np.concatenate([self.val_head_data, data])
            self.val_head_label = np.concatenate([self.val_head_label, label])
            self.val_head_rssi = np.concatenate([self.val_head_rssi, rssi])
        else:
            self.all_test_head_data = np.concatenate([self.all_test_head_data, data])
            self.all_test_head_label = np.concatenate([self.all_test_head_label, label])
            self.all_test_head_rssi = np.concatenate([self.all_test_head_rssi, rssi])

    def concatenate_all_data(self):
        self.all_end_data = self.concatenate_data(
            [self.train_end_data, self.val_end_data, self.all_test_end_data]
        )
        self.all_head_data = self.concatenate_data(
            [self.train_head_data, self.val_head_data, self.all_test_head_data]
        )
        self.all_end_rssi = self.concatenate_data(
            [self.train_end_rssi, self.val_end_rssi, self.all_test_end_rssi]
        )
        self.all_head_rssi = self.concatenate_data(
            [self.train_head_rssi, self.val_head_rssi, self.all_test_head_rssi]
        )
        self.all_end_label = self.concatenate_data(
            [self.train_end_label, self.val_end_label, self.all_test_end_label]
        )
        self.all_head_label = self.concatenate_data(
            [self.train_head_label, self.val_head_label, self.all_test_head_label]
        )

        self.print_shapes(
            "all_end",
            self.all_end_data,
            self.all_end_label,
            self.all_end_rssi,
        )
        self.print_shapes(
            "all_head",
            self.all_head_data,
            self.all_head_label,
            self.all_head_rssi,
        )

    def concatenate_test_all_data(self):
        self.all_test_end_data = self.concatenate_data(
            [self.val_end_data, self.all_test_end_data]
        )
        self.all_test_head_data = self.concatenate_data(
            [self.val_head_data, self.all_test_head_data]
        )
        self.all_test_end_rssi = self.concatenate_data(
            [self.val_end_rssi, self.all_test_end_rssi]
        )
        self.all_test_head_rssi = self.concatenate_data(
            [self.val_head_rssi, self.all_test_head_rssi]
        )
        self.all_test_end_label = self.concatenate_data(
            [self.val_end_label, self.all_test_end_label]
        )
        self.all_test_head_label = self.concatenate_data(
            [self.val_head_label, self.all_test_head_label]
        )

        self.print_shapes(
            "all_test_end",
            self.all_test_end_data,
            self.all_test_end_label,
            self.all_test_end_rssi,
        )
        self.print_shapes(
            "all_test_head",
            self.all_test_head_data,
            self.all_test_head_label,
            self.all_test_head_rssi,
        )

    def concatenate_train_all_data(self):
        self.all_train_end_data = self.concatenate_data(
            [self.train_end_data, self.val_end_data]  # , self.all_test_end_data]
        )
        self.all_train_head_data = self.concatenate_data(
            [self.train_head_data, self.val_head_data]  # , self.all_test_head_data]
        )
        self.all_train_end_rssi = self.concatenate_data(
            [self.train_end_rssi, self.val_end_rssi]  # , self.all_test_end_rssi]
        )
        self.all_train_head_rssi = self.concatenate_data(
            [self.train_head_rssi, self.val_head_rssi]  # , self.all_test_head_rssi]
        )
        self.all_train_end_label = self.concatenate_data(
            [self.train_end_label, self.val_end_label]  # , self.all_test_end_label]
        )
        self.all_train_head_label = self.concatenate_data(
            [self.train_head_label, self.val_head_label]  # , self.all_test_head_label]
        )

        self.print_shapes(
            "all_train_end",
            self.all_train_end_data,
            self.all_train_end_label,
            self.all_train_end_rssi,
        )
        self.print_shapes(
            "all_train_head",
            self.all_train_head_data,
            self.all_train_head_label,
            self.all_train_head_rssi,
        )

    def combine_data(self, date, data_type, position):
        stop_data, stop_label, stop_rssi = self.read_aug_data(
            date, data_type, position, "stop"
        )
        drive_data, drive_label, drive_rssi = self.read_aug_data(
            date, data_type, position, "drive"
        )
        combined_data = self.concatenate_data([stop_data, drive_data])
        combined_label = self.concatenate_data([stop_label, drive_label])
        combined_rssi = self.concatenate_data([stop_rssi, drive_rssi])
        # combined_pos = self.concatenate_data([stop_pos, drive_pos])
        return combined_data, combined_label, combined_rssi  # , combined_pos

    def combine_head_end_data(self):
        self.combine_train_data = self.concatenate_data(
            [self.train_end_data, self.train_head_data]
        )
        self.combine_train_label = self.concatenate_data(
            [self.train_end_label, self.train_head_label]
        )
        self.combine_train_rssi = self.concatenate_data(
            [self.train_end_rssi, self.train_head_rssi]
        )

    def get_adv_data(self, type, position, data, label, rssi):
        self.data_by_date[f"{type}_{position}_data"] = data
        self.data_by_date[f"{type}_{position}_label"] = label
        self.data_by_date[f"{type}_{position}_rssi"] = rssi

        print(f"{type}_{position}_data {data.shape}")
        print(f"{type}_{position}_label {label.shape}")
        print(f"{type}_{position}_rssi {rssi.shape}")

    def get_data(self, date, position, data, label, rssi):
        if f"{date}_{position}_data" not in self.data_by_date:
            self.data_by_date[f"{date}_{position}_data"] = data
            self.data_by_date[f"{date}_{position}_label"] = label
            self.data_by_date[f"{date}_{position}_rssi"] = rssi
            # self.data_by_date[date]["size"] = data.shape[0]
        else:
            self.data_by_date[f"{date}_{position}_data"] = self.concatenate_data(
                [self.data_by_date[f"{date}_{position}_data"], data]
            )
            self.data_by_date[f"{date}_{position}_label"] = self.concatenate_data(
                [self.data_by_date[f"{date}_{position}_label"], label]
            )
            self.data_by_date[f"{date}_{position}_rssi"] = self.concatenate_data(
                [self.data_by_date[f"{date}_{position}_rssi"], rssi]
            )
            # self.data_by_date[date]["size"] += data.shape[0]

    def read_aug_data(self, date, data_type, position, action):
        return read_aug_data(self.args, date, data_type, position, action)

    def concatenate_data(self, data_list):
        return np.concatenate(data_list, axis=0)

    def print_shapes(self, date, *data_arrays):
        for data_array in data_arrays:
            print(f"{date} {data_array.shape}")


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        assert (
            0 <= label_smoothing < 1
        ), "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = (
                target * positive_smoothed_labels
                + (1 - target) * negative_smoothed_labels
            )

        loss = self.bce_with_logits(input, target)
        return loss


class apply_time_masking:
    """
    Apply time masking to the csi_data.

    Parameters:
    csi_data (torch.Tensor): The input csi_data tensor.
    time_mask_param (int): Maximum possible length of the mask.

    Returns:
    torch.Tensor: The csi_data with time masking applied.
    """

    def __init__(self, time_mask_param):
        self.time_mask_param = time_mask_param

    def __call__(self, csi_data):
        num_frames = csi_data.shape[1]
        t = np.random.randint(0, self.time_mask_param)
        t0 = np.random.randint(0, num_frames - t)
        csi_data[t0 : t0 + t, :] = 0
        return csi_data


class apply_freq_masking:
    """
    Apply freq masking to the csi_data.

    Parameters:
    csi_data (torch.Tensor): The input csi_data tensor.
    freq_mask_param (int): Maximum possible length of the mask.

    Returns:
    torch.Tensor: The csi_data with freq masking applied.
    """

    def __init__(self, freq_mask_param):
        self.freq_mask_param = freq_mask_param

    def __call__(self, csi_data):
        num_subcarriers = csi_data.shape[2]
        f = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, num_subcarriers - f)
        csi_data[:, f0 : f0 + f] = 0
        return csi_data


def mixup_process(out, target, alpha):
    if alpha > 0.0:
        lam = torch.from_numpy(np.array([np.random.beta(alpha, alpha)])).float().cuda()
    else:
        lam = 1.0

    batch_size = out.size()[0]
    indices = torch.randperm(batch_size).cuda()

    out = out * lam + out[indices, :] * (1 - lam)
    y_a, y_b = target, target[indices]
    return out, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


def aggregate_parameters(global_model, uploaded_models):

    for params in global_model.parameters():
        params.data.zero_()

    for client_model in uploaded_models:
        add_parameters(global_model, client_model)


def add_parameters(global_model, client_model):
    for server_param, client_param in zip(
        global_model.parameters(), client_model.parameters()
    ):
        if server_param.data.shape[0] == client_param.data.shape[0]:
            server_param.data += client_param.data.clone() / 2


def set_parameters(local_model, global_model):
    for new_param, old_param in zip(
        global_model.parameters(), local_model.parameters()
    ):
        if new_param.data.shape[0] == old_param.data.shape[0]:
            old_param.data = new_param.data.clone()


class AM_Softmax(nn.Module):  # requires classification layer for normalization
    def __init__(
        self,
        m=0.3,
        s=30,
        num_classes=625,
        label_smoothing=0.1,
        classifier=None,
    ):
        super(AM_Softmax, self).__init__()
        self.m = m
        self.s = s
        self.num_classes = num_classes
        self.CrossEntropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if classifier:
            self.classifier = classifier
            # # print(self.classifier.weight.shape)
            # bound = 1 / math.sqrt(d)
            # nn.init.uniform_(self.classifier.weight, -bound, bound)

    def forward(self, features, labels):
        """
        x : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        """
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        features = nn.functional.normalize(
            features, p=2, dim=1
        )  # normalize the features
        # with torch.no_grad():
        #     classifier.weight.div_(torch.norm(classifier.weight, dim=1, keepdim=True))
        # print(features.shape)
        # cos_angle = classifier(features)
        with torch.no_grad():
            self.classifier.weight.div_(
                torch.norm(self.classifier.weight, dim=1, keepdim=True)
            )

        cos_angle = self.classifier(features)
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] - self.m
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels)
        return log_probs


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8
    )

    mean = torch.zeros(1)
    std = torch.zeros(1)
    for inputs, targets in dataloader:
        for i in range(1):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class Reconstruct_Noise_Frequency:
    "reconstruct the noise frequency"

    def __init__(self, noise) -> None:
        self.noise = noise

    def __call__(self, data):
        noise_freq_data = self.noise(torch.fft.rfft(data, 1))
        return torch.fft.irfft(noise_freq_data, 1)


class Add_Noise:
    "add different type of noise"

    def __init__(self, noise_type) -> None:
        self.noise_type = noise_type

    def __call__(self, data):

        scale = torch.full(
            size=data.shape,
            fill_value=np.random.uniform(low=0.0, high=0.1),
            dtype=torch.float32,
        )
        mean = torch.full(
            size=data.shape,
            fill_value=np.random.uniform(low=0.0, high=0.0),
            dtype=torch.float32,
        )
        # perturb_coe = np.random.uniform(low=0.0, high=0.1)
        if self.noise_type == "gaussian":
            dist = torch.distributions.normal.Normal(mean, scale)
        elif self.noise_type == "laplacian":
            dist = torch.distributions.laplace.Laplace(mean, scale)
        elif self.noise_type == "exponential":
            rate = 1 / scale
            dist = torch.distributions.exponential.Exponential(rate)
        else:
            dist = torch.distributions.normal.Normal(0, scale)
        noise = dist.sample()

        # return data * (1 - perturb_coe) + noise
        return data + noise
        # return data + perturb_coe * noise


class CsiRssi(Dataset):
    def __init__(
        self,
        data,
        transform=None,
        rssi_transform=None,
    ) -> None:
        super().__init__()

        self.data = data[0]
        self.rssi = data[1]
        self.transform = transform
        self.rssi_transform = rssi_transform

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        rssi = torch.tensor(self.rssi[index])

        if self.transform != None:
            data = self.transform(data)

        if self.rssi_transform != None:
            rssi = self.rssi_transform(rssi.unsqueeze_(-1)).squeeze_(-1)

        return data, rssi

    def __len__(self):
        return self.data.shape[0]


class CsiRssiLabeled(Dataset):
    def __init__(
        self,
        data,
        labels,
        transform=None,
        rssi_transform=None,
    ) -> None:
        super().__init__()

        self.data = data[0]
        self.rssi = data[1]
        self.labels = labels[0]
        # self.pos_labels = labels[1]
        self.transform = transform
        self.rssi_transform = rssi_transform

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        rssi = torch.tensor(self.rssi[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        # pos_label = self.pos_labels[index]

        if self.transform != None:
            data = self.transform(data)

        if self.rssi_transform != None:
            rssi = self.rssi_transform(rssi.unsqueeze_(-1)).squeeze_(-1)

        return data, label, rssi  # , pos_label

    def __len__(self):
        return self.data.shape[0]


class CsiLabeled(Dataset):
    def __init__(
        self,
        data,
        labels,
        transform=None,
    ) -> None:
        super().__init__()

        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        label = self.labels[index]

        if self.transform != None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return self.data.shape[0]


# def plot_retrieval_score():


def plot_confusion_matrix(cm, labels, result_path, fig):
    plt.imshow(cm, aspect="auto", interpolation="nearest", cmap=plt.get_cmap("turbo"))
    plt.colorbar()
    font1 = {"size": 17}
    xlocations = np.array(range(len(labels)))  # （1，c）类别个数作为位置，
    plt.xticks(
        xlocations, labels, fontsize=17
    )  # 将x坐标的label旋转0度 ，每个位置上放好对应的label
    plt.yticks(xlocations, labels, fontsize=17)
    plt.ylabel("True label", font1)
    plt.xlabel("Predicted label", font1)
    plt.savefig(os.path.join(result_path, fig), dpi=300, bbox_inches="tight")


def draw_matrix_acc(y_pred, y_true, labels, result_path, fig):
    cm = confusion_matrix(y_true, y_pred)  # 求解confusion matrix
    tick_marks = np.array(range(cm.shape[0])) + 0.5
    ind_array = np.arange(cm.shape[0])
    # np.set_printoptions(precision=0)
    # cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(20, 8), dpi=120)

    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]  # cm_normalized[y_val][x_val]
        # if c > 0.01:
        plt.text(
            x_val,
            y_val,
            c,
            # "%0.3f" % (c * 100,),
            color="white",
            fontsize=17,
            va="center",
            ha="center",
        )
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    plt.grid(True, which="minor", linestyle="-")
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm, labels, result_path, fig)
    # show confusion matrix
    # plt.show()


def read_aug_data(args, date, status, place, bus_state):
    file_name = os.path.join(
        os.getcwd(),
        f"../ssd/csi_data/datasets/{date}_{place}_{bus_state}_{args.aug}_tp2_{status}.npz",
    )
    with np.load(file_name, allow_pickle=True) as data:
        if args.aug == "hp_dwt_sg" or "raw" in args.aug:
            x = (
                data["data"]
                .tolist()["x"]
                .astype("float32")[
                    :,
                    np.newaxis,
                    :,
                ]
            )
            rssi = (
                data["data"]
                .tolist()["rssi"]
                .astype("float32")[
                    :,
                    np.newaxis,
                    :,
                ]
            )
        elif "lp_sg" in args.aug:
            x = (
                data["data"]
                .tolist()["x"]
                .astype("float32")[
                    :,
                    np.newaxis,
                    :,
                ]
            )
            rssi = (
                data["data"]
                .tolist()["rssi"]
                .astype("float32")[
                    :,
                    np.newaxis,
                    :,
                ]
            )
            # pos_info = data["data"].tolist()["p"].astype("int64")
            y = data["data"].tolist()["y"].astype("int64")
            return x, y, rssi  # , pos_info
        elif args.aug == "select_dfs":
            print(data["data"].tolist()["x"].shape)
            if date == "1031":
                x = (
                    data["data"]
                    .tolist()["x"]
                    .reshape(-1, 13, 121, 101)
                    .astype("float32")
                )
            else:
                x = (
                    data["data"]
                    .tolist()["x"]
                    .reshape(-1, 13, 121, 91)
                    .astype("float32")
                )
            if status == "train":
                rssi = (
                    data["data"]
                    .tolist()["rssi"]
                    .repeat(2, axis=0)
                    .astype("float32")[
                        :,
                        np.newaxis,
                        :,
                    ]
                )
                y = data["data"].tolist()["y"].repeat(2).astype("int64")
            else:
                rssi = (
                    data["data"]
                    .tolist()["rssi"]
                    .astype("float32")[
                        :,
                        np.newaxis,
                        :,
                    ]
                )
                y = data["data"].tolist()["y"].astype("int64")
            return x, y, rssi
        if "real_time" in args.aug:
            y = np.array([])
        else:
            y = data["data"].tolist()["y"].astype("int64")
            return x, y, rssi


def read_npz_data(args, file_name, train=True, val=False, split=False):
    if train:
        label = "train_data"
        with np.load(file_name, allow_pickle=True) as data:
            x = data[label].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")

    elif val:
        label = "val_data"
        with np.load(file_name, allow_pickle=True) as data:
            x = data[label].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")
    else:
        label = "test_data"
        with np.load(file_name, allow_pickle=True) as data:
            x = data[label].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")

    if args.aug == "hp_sg_dfs":
        x = x.squeeze()
    return x, y


def read_atten_npz_data(file_name, train=True, val=False, split=False):
    label = "wifi_data"
    if train:
        if split:
            label = "train_data"
        with np.load(file_name, allow_pickle=True) as data:
            # x = data["train"].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")

        aug_file_name = f'{file_name.split(".npz")[0]}_atten.npz'
        with np.load(aug_file_name, allow_pickle=True) as data:
            x = data["train"].tolist()["x"].astype("float32")
    elif val:
        if split:
            label = "val_data"
        with np.load(file_name, allow_pickle=True) as data:
            # x = data["train"].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")

        aug_file_name = f'{file_name.split(".npz")[0]}_atten.npz'
        with np.load(aug_file_name, allow_pickle=True) as data:
            x = data["val"].tolist()["x"].astype("float32")
    else:
        if split:
            label = "test_data"
        with np.load(file_name, allow_pickle=True) as data:
            # x = data["test"].tolist()["x"].astype("float32")
            y = data[label].tolist()["y"].astype("int64")

        aug_file_name = f'{file_name.split(".npz")[0]}_atten.npz'
        with np.load(aug_file_name, allow_pickle=True) as data:
            x = data["test"].tolist()["x"].astype("float32")

    return x, y


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    temp = np.exp(x - x_max)
    s = np.sum(temp, axis=axis, keepdims=True)
    return temp / s


class Accumulator:
    """For accumulating sums over 'n' variables"""

    def __init__(self, acc_type, n):
        self.data = acc_type * n

    def add(self, *arg):
        self.data = [a + b for a, b in zip(self.data, arg)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Using cudnn.deterministic.")


from typing import Tuple


def get_maxprob_and_onehot(
    probs: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    maxprob_list = []
    idx_list = []

    for i in range(len(probs)):
        maxprob_list.append(np.max(probs[i]))
        idx_list.append(np.argmax(probs[i]))

    maxprob_list = np.array(maxprob_list)
    idx_list = np.array(idx_list)
    one_hot_labels = labels == idx_list

    return maxprob_list, one_hot_labels
