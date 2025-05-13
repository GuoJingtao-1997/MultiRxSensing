import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from mobileones import mobileone, reparameterize_model
import os
from nexcsi import decoder
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from utils import CsiRssiLabeled, CsiRssi, DataProcessor, setup_seed
from adapt_algo import AdaNPC, KNN, CSIDataset, RTCSI
from scipy.stats import zscore
import misc
import torch.nn as nn
import argparse
from merge_model import AttentionMerge
import copy
from scipy.signal import butter, filtfilt, savgol_filter

csi_dataset_path = os.path.join(os.getcwd(), r"../")
# Sample rate and desired cutoff frequency (in Hz)
fs = 1000.0  # Sample rate, in Hz
cutoff = 30.0  # Desired cutoff frequency of the filter, in Hz
# Normalize the frequency
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

order = 4  # Filter order
b, a = butter(order, normal_cutoff, btype="low", analog=False)


class ReadPCAPData:
    def __init__(self, args):
        self.idx = args.idx
        self.fine_tune = args.fine_tune
        self.test_date = args.test_date
        self.train_samples = args.few_shot_samples
        self.diff_time = args.diff_time
        self.diff_pos = args.diff_pos
        self.device = "raspberrypi"
        self.train_csi = []
        self.train_rssi = []
        self.train_label = []

        self.test_csi = []
        self.test_rssi = []
        self.test_label = []

    def read_pcap(self, pcap_path):
        pcap_folder = os.path.join(csi_dataset_path, pcap_path)
        print(f"{pcap_folder}")
        for folder in os.listdir(pcap_folder):
            label = int(folder)
            files = os.listdir(os.path.join(pcap_folder, folder))
            num_files = len(files)
            print(f"folder {folder} files {num_files}")
            if self.test_date == "48" and self.fine_tune:
                self.train_samples = int(0.8 * num_files)
            for i, pcap in enumerate(files):
                if folder not in ["1", "3", "5", "6"]:
                    break
                try:
                    pcap_file = os.path.join(pcap_folder, folder, pcap)
                    sample = decoder(self.device).read_pcap(pcap_file)
                    rssi = sample["rssi"]
                    csi = decoder(self.device).unpack(sample["csi"])
                    csi = np.delete(csi, csi.dtype.metadata["nulls"], axis=1)
                    csi = filtfilt(b, a, csi, axis=0)
                    csi = savgol_filter(np.abs(csi), 500, 5, axis=0)
                    if i < self.train_samples:
                        self.train_csi.append(csi[np.newaxis, :, :])
                        self.train_rssi.append(rssi[np.newaxis, :])
                        self.train_label.append(label)
                    else:
                        # if (self.diff_pos or self.diff_time) and "middle_stop" in pcap_path:
                        #     print(f"{pcap_path}")
                        #     continue
                        self.test_csi.append(csi[np.newaxis, :, :])
                        self.test_rssi.append(rssi[np.newaxis, :])
                        self.test_label.append(label)
                except Exception as e:
                    print(f"Error reading {pcap_file}: {e}")
                    continue

    def read_to_np(self):

        self.train_csi = np.array(self.train_csi)
        self.train_rssi = np.array(self.train_rssi)
        self.train_label = np.array(self.train_label)

        self.test_csi = np.array(self.test_csi)
        self.test_rssi = np.array(self.test_rssi)
        self.test_label = np.array(self.test_label)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument(
        "-lr",
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="learning rate (default: 3e-4)",
    )

    parser.add_argument(
        "-wd",
        "--wd",
        help="weight decay parameter (default: 1e-4);",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--use_knn",
        action="store_true",
    )

    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        metavar="N",
        help="index of different collection time",
    )

    parser.add_argument(
        "--skip_model_save",
        action="store_true",
        help="skip model dict saving",
    )

    parser.add_argument(
        "--long_duration_pred",
        action="store_true",
        help="generate long duration prediction",
    )

    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="use data to fine tune the model",
    )

    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        choices=(0.01, 0.05, 0.1, 0.25, 0.5, 1.5),
        help="temperature of softmax",
    )

    parser.add_argument(
        "--num_class",
        type=int,
        default=21,
        metavar="N",
        help="# of passenger in pre-trained dataset",
    )

    parser.add_argument(
        "--num_test_class",
        type=int,
        default=21,
        metavar="N",
        help="# of passenger in pre-trained dataset",
    )

    parser.add_argument(
        "--duration_samples",
        type=int,
        default=30,
        metavar="N",
        help="# of samples per test duration",
    )

    parser.add_argument(
        "--number_combination",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--aug",
        type=str,
        default="lp_sg_1000",
        metavar="N",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="stop",
        metavar="N",
    )

    parser.add_argument(
        "--bank_type",
        type=str,
        default="memory",
        metavar="N",
    )

    parser.add_argument(
        "--few_shot_samples",
        type=int,
        default=1,
        metavar="N",
        help="# of ground true sample in new environment",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=100,
        metavar="N",
        help="number of k for knn",
    )

    parser.add_argument(
        "--concate_date",
        action="store_true",
        help="concate train data from different date",
    )

    parser.add_argument(
        "--test_date",
        type=str,
        default="613",
        metavar="N",
        help="data collection date",
    )

    parser.add_argument(
        "--train_date",
        type=str,
        default="613",
        metavar="N",
        help="model pretraining date",
    )

    parser.add_argument(
        "--pretrain_date",
        type=str,
        default="613",
        metavar="N",
        help="model pretraining date",
    )

    parser.add_argument(
        "--diff_time",
        action="store_true",
        help="different collection time",
    )

    parser.add_argument(
        "--diff_pos",
        action="store_true",
        help="different collection position",
    )

    args = parser.parse_args()
    return args


def normalize(data, mean, std):
    data = (data - mean) / std
    return data


def compute_weights_from_entropy(entropy_values):
    return 1.0 / (1e-6 + entropy_values)


def compute_normalized_weights(outputs):
    """
    Convert logits or probabilities to normalized weights (sum to 1).
    """
    # 1) Compute entropy (optional step if you need an entropy-based weighting)
    ent = softmax_entropy(outputs)

    print(f"ent {ent}")

    # 2) Compute unnormalized weights with the existing function
    unnormalized = compute_weights_from_entropy(ent)

    print(f"unnormalized {unnormalized}")

    # 3) Normalize weights with softmax to ensure they sum to 1
    normalized_weights = torch.softmax(unnormalized, dim=0).detach().cpu().numpy()

    print(f"normalized_weights {normalized_weights}")

    return normalized_weights


def remove_outliers_sliding_window(
    pred_label_array, pred_prob_array, window_size=11, z_threshold=2.5
):
    """
    Identify and remove outliers from an array using a sliding window approach.

    Parameters:
    array (numpy.ndarray): Input array from which outliers need to be removed.
    window_size (int): Size of the sliding window. Default is 11.
    z_threshold (float): Z-score threshold for outlier detection. Default is 2.5.

    Returns:
    numpy.ndarray: Array with outliers removed.
    """
    outliers = []
    half_window = window_size // 2

    for i in range(len(pred_label_array)):
        # Define the window boundaries
        start = max(0, i - half_window)
        end = min(len(pred_label_array), i + half_window + 1)
        window = pred_label_array[start:end]

        if len(window) < 2:  # Avoid calculation if window size is too small
            continue

        # Calculate Z-scores for the window
        window_z_scores = zscore(window)

        # Find the Z-score of the current element within its window
        current_index_in_window = i - start
        current_z_score = window_z_scores[current_index_in_window]

        if np.abs(current_z_score) > z_threshold:
            outliers.append(i)

    print("outliers: ", outliers)

    cleaned_pred_prob_array = np.delete(pred_prob_array, outliers, axis=0)
    return cleaned_pred_prob_array


def knn_merge_short_test(merge_model, loaders, device, cls):
    test_accumulator = 0
    ent = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for xh, yh in loaders[0]:
            xh = xh.to(device)
            yh = yh.to(device)
            output = merge_model(xh, adapt=False)

            print(f"output shape {output.shape}")

            print(f"output {output}")
            print(f"y {yh}")
            print(f"output {torch.argmax(output, dim=-1)}")

            test_accumulator += np.sum(yh.shape[0])
            ent += softmax_entropy(torch.tensor(output)).sum().item()

            y_prob.append(torch.argmax(output, dim=-1).detach().cpu().numpy())
            y_true.append(yh.detach().cpu().numpy())

    print(f"y_prob {y_prob}")
    print(f"y_true {y_true}")

    y_prob = np.concatenate(y_prob, axis=0)

    y_true = np.concatenate(y_true, axis=0)

    eval_metrics = [
        metrics.accuracy_score(y_true, y_prob),
        metrics.precision_score(y_true, y_prob, average="macro"),
        metrics.recall_score(y_true, y_prob, average="macro"),
        metrics.f1_score(y_true, y_prob, average="macro"),
        metrics.confusion_matrix(y_true, y_prob),
        metrics.mean_absolute_error(y_true, y_prob),
    ]

    auc = metrics.roc_auc_score(
        label_binarize(y_true, classes=np.arange(cls)),
        label_binarize(y_prob, classes=np.arange(cls)),
        average="micro",
        multi_class="ovr",
    )

    print(f"Test Accurancy: {eval_metrics[0]:.4%}")
    print(f"Test Precision: {eval_metrics[1]:.4%}")
    print(f"Test Recall: {eval_metrics[2]:.4%}")
    print(f"Test F1-Score: {eval_metrics[3]:.4%}")
    print(f"Test MAE: {eval_metrics[5]:.4}")
    print(f"Test AUC: {auc:.4f}")

    return ent / test_accumulator, auc, eval_metrics, y_prob, y_true


def knn_merge_long_test(merge_model, loaders, device, cls):
    test_accumulator = 0
    ent = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for xh, yh in loaders[0]:

            xh = xh.to(device)
            yh = yh.to(device)
            output = merge_model(xh, adapt=False)

            print(f"output {output}")

            entropy_weights = compute_normalized_weights(output)

            print(f"entropy {entropy_weights}")

            print(f"entropy shape {entropy_weights.shape}")

            print(f"yh {yh}")

            print(f"output shape {output.shape}")

            y_prob_merge = torch.softmax(output, dim=1).detach().cpu().numpy()

            print(f"y_prob_merge shape {y_prob_merge.shape}")

            y_label = torch.argmax(output, dim=1).detach().cpu().numpy()

            print(f"y_label_head shape {y_label.shape}")

            y_prob_merge = remove_outliers_sliding_window(
                y_label, y_prob_merge, window_size=3, z_threshold=2
            )

            print(f"y_prob_merge {y_prob_merge}")

            # merge_output = np.argmax(merge_output, axis=1)

            # # Get unique values and their counts
            # unique_values, counts = np.unique(merge_output, return_counts=True)

            # # Find the index of the maximum count
            # max_count_index = np.argmax(counts)

            # # Select the value with the maximum count
            # final_pred = unique_values[max_count_index]

            # # Print the result
            # print(
            #     f"The value that appears most frequently is {final_pred}, appearing {counts[max_count_index]} times."
            # )

            # prob_avg = np.zeros_like(merge_output[0])
            # count = 0

            # for prob in merge_output:
            #     count += 1
            #     prob_avg += prob

            # prob_avg /= count

            prob_weight_avg = np.sum(
                (entropy_weights[:, np.newaxis] * y_prob_merge), axis=0
            )

            final_pred = np.argmax(prob_weight_avg)
            y = torch.sum(yh) // yh.shape[0]

            # print(f"output {output}")
            print(f"prob_avg {prob_weight_avg}")
            print(f"y {y}")
            print(f"final pred {final_pred}")

            test_accumulator += np.sum(xh.shape[0])
            ent += softmax_entropy(torch.tensor(output)).sum().item()

            y_prob.append(final_pred)
            y_true.append(y.detach().cpu().numpy())

    ent /= test_accumulator

    print(f"ent {ent}")
    print(f"y_prob {y_prob}")
    print(f"y_true {y_true}")

    # y_prob = np.concatenate(y_prob)
    # y_true = np.concatenate(y_true)

    eval_metrics = [
        metrics.accuracy_score(y_true, y_prob),
        metrics.precision_score(y_true, y_prob, average="macro"),
        metrics.recall_score(y_true, y_prob, average="macro"),
        metrics.f1_score(y_true, y_prob, average="macro"),
        metrics.confusion_matrix(y_true, y_prob),
        metrics.mean_absolute_error(y_true, y_prob),
    ]

    auc = metrics.roc_auc_score(
        label_binarize(y_true, classes=np.arange(cls)),
        label_binarize(y_prob, classes=np.arange(cls)),
        average="micro",
        multi_class="ovr",
    )

    print(f"Test Accurancy: {eval_metrics[0]:.4%}")
    print(f"Test Precision: {eval_metrics[1]:.4%}")
    print(f"Test Recall: {eval_metrics[2]:.4%}")
    print(f"Test F1-Score: {eval_metrics[3]:.4%}")
    print(f"Test MAE: {eval_metrics[5]:.4}")
    print(f"Test AUC: {auc:.4f}")

    return (
        ent / test_accumulator,
        auc,
        eval_metrics,
        y_prob,
        y_true,
    )


def cnn_merge_long_test(merge_model, loaders, device, cls):
    test_accumulator = 0
    ent = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for xh, yh in loaders[0]:

            xh = xh.to(device)
            yh = yh.to(device)
            output = merge_model(xh)

            print(f"yh {yh}")

            print(f"output shape {output.shape}")

            y_prob_merge = torch.softmax(output, dim=1).detach().cpu().numpy()

            print(f"y_prob_merge shape {y_prob_merge.shape}")

            y_label = torch.argmax(output, dim=1).detach().cpu().numpy()

            print(f"y_label_head shape {y_label.shape}")

            merge_output = remove_outliers_sliding_window(
                y_label, y_prob_merge, window_size=5, z_threshold=3
            )

            print(f"merge_output shape {merge_output.shape}")

            prob_avg = np.zeros_like(merge_output[0])
            count = 0

            for prob in merge_output:
                count += 1
                prob_avg += prob

            prob_avg /= count

            final_pred = np.argmax(prob_avg)
            y = torch.sum(yh) // yh.shape[0]

            # print(f"output {output}")
            print(f"prob_avg {prob_avg}")
            print(f"y {y}")
            print(f"final pred {final_pred}")

            test_accumulator += np.sum(xh.shape[0])
            ent += softmax_entropy(torch.tensor(output)).sum().item()

            y_prob.append(final_pred)
            y_true.append(y.detach().cpu().numpy())

    ent /= test_accumulator

    print(f"ent {ent}")
    print(f"y_prob {y_prob}")
    print(f"y_true {y_true}")

    # y_prob = np.concatenate(y_prob)
    # y_true = np.concatenate(y_true)

    eval_metrics = [
        metrics.accuracy_score(y_true, y_prob),
        metrics.precision_score(y_true, y_prob, average="macro"),
        metrics.recall_score(y_true, y_prob, average="macro"),
        metrics.f1_score(y_true, y_prob, average="macro"),
        metrics.confusion_matrix(y_true, y_prob),
        metrics.mean_absolute_error(y_true, y_prob),
    ]

    auc = metrics.roc_auc_score(
        label_binarize(y_true, classes=np.arange(cls)),
        label_binarize(y_prob, classes=np.arange(cls)),
        average="micro",
        multi_class="ovr",
    )

    print(f"Test Accurancy: {eval_metrics[0]:.4%}")
    print(f"Test Precision: {eval_metrics[1]:.4%}")
    print(f"Test Recall: {eval_metrics[2]:.4%}")
    print(f"Test F1-Score: {eval_metrics[3]:.4%}")
    print(f"Test MAE: {eval_metrics[5]:.4}")
    print(f"Test AUC: {auc:.4f}")

    return (
        ent / test_accumulator,
        auc,
        eval_metrics,
        y_prob,
        y_true,
    )


def on_publish(client, userdata, mid, reason_code, properties):
    # reason_code and properties will only be present in MQTTv5. It's always unset in MQTTv3
    try:
        print("sending....")
    except KeyError:
        print("on_publish() is called with a mid not present in unacked_publish")
        print("This is due to an unavoidable race-condition:")
        print("* publish() return the mid of the message sent.")
        print("* mid from publish() is added to unacked_publish by the main thread")
        print("* on_publish() is called by the loop_start thread")
        print(
            "While unlikely (because on_publish() will be called after a network round-trip),"
        )
        print(" this is a race-condition that COULD happen")
        print("")
        print(
            "The best solution to avoid race-condition is using the msg_info from publish()"
        )
        print(
            "We could also try using a list of acknowledged mid rather than removing from pending list,"
        )
        print("but remember that mid could be re-used !")


def on_connect(client, userdata, flag, reason_code, properties):
    print(f"Connected with result code {reason_code}")


def generate_real_time_merge_featurelized_loader(
    head_loader, end_loader, network, classifier, device, batch_size=64
):
    """
    The classifier adaptation does not need to repeat the heavy forward path,
    We speeded up the experiments by converting the observations into representations.
    """
    z_list = []
    p_list = []
    network.eval()
    classifier.eval()
    with torch.no_grad():
        for (xh, rh), (xe, re) in zip(head_loader, end_loader):
            xh, xe = xh.to(device), xe.to(device)
            rh, re = rh.to(device), re.to(device)
            z = network(xh, xe, rh, re)
            p = classifier(z)

            z_list.append(z.detach().cpu())
            p_list.append(p.detach().cpu())
            # p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1 = RTCSI(z)
    loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return loader1, ent, z


def generate_merge_featurelized_loader(
    head_loader,
    end_loader,
    network,
    # knn_classifier,
    # mlp_classifier,
    device,
    batch_size,
    # feature_per_sample,
    # num_class,
):
    """
    The classifier adaptation does not need to repeat the heavy forward path,
    We speeded up the experiments by converting the observations into representations.
    """
    z_list = []
    y_list = []
    # knn_p_list = []
    # mlp_p_list = []
    network.eval()
    # knn_classifier.eval()
    # mlp_classifier.eval()
    with torch.no_grad():
        for (xh, yh, rh), (xe, _, re) in zip(head_loader, end_loader):
            xh, xe = xh.to(device), xe.to(device)
            rh, re = rh.to(device), re.to(device)
            z = network(
                {
                    "x1": xh,
                    "x2": xe,
                    "rssi1": rh,
                    "rssi2": re,
                }
            )
            # knn_p = knn_classifier(z)
            # mlp_p = mlp_classifier(z)

            z_list.append(z.detach().cpu())
            y_list.append(yh.detach().cpu())
            # knn_p_list.append(knn_p.detach().cpu())
            # mlp_p_list.append(mlp_p.detach().cpu())
            # knn_p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    # knn_classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    # knn_p = torch.cat(knn_p_list)
    # mlp_p = torch.cat(mlp_p_list)
    # knn_ent = softmax_entropy(knn_p)
    # mlp_ent = softmax_entropy(mlp_p)
    # knn_py = knn_p.argmax(1).float().cpu().detach()
    dataset1 = CSIDataset(z, y)  # , CSIDataset(z, knn_py)
    loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, drop_last=False
    )
    # loader2 = torch.utils.data.DataLoader(
    #     dataset2, batch_size=batch_size, shuffle=False, drop_last=False
    # )
    return (
        loader1,
        z,
        y,
    )  # knn_ent, mlp_ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)


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


def model_load(checkpoint_path: str, device):
    model = mobileone(num_classes=21, variant="s4", inference_mode=False).to(device)
    checkpoint = torch.load(
        os.path.join(
            os.getcwd(),
            "head_end_20_null_True_knn_False_300_tp2_atten_filter_prob_avg",
            f"613_{checkpoint_path}_mos4.pth",
        ),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint)
    # weight_norm_layer = list(model.children())[-1]
    model = reparameterize_model(model)
    model.eval()
    return model


def get_balanced_indices(labels, num_classes, num_samples_per_class):
    balanced_indices = []
    for class_id in range(num_classes):
        # Get actual indices where this class appears
        class_indices = np.where(labels == class_id)[0]
        print(f"{class_id} class_indices {len(class_indices)}")
        # Randomly sample exactly num_samples_per_class indices
        selected_indices = np.random.choice(
            class_indices, size=num_samples_per_class, replace=False
        )
        balanced_indices.extend(selected_indices)
    return balanced_indices


if __name__ == "__main__":
    setup_seed(42)
    print("seed: 42")
    print("PyTorch Version: ", torch.__version__)
    print("setup torch device")
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # diff_time = ["middle_stop", "middle_different_stop_time"]
    if args.test_date == "19":
        diff_pos = {
            0: "middle_stop",
            1: "middle_run",
            2: "rear_stop",
            3: "front_stop",
            4: "middle_different_stop_time",
        }
    elif args.test_date in ["26", "27"]:
        diff_pos = ["middle_stop", "rear_stop", "front_stop"]
        diff_time = ["middle_v2_stop", "middle_v3_stop", "middle_stop"]
    elif args.test_date == "48":
        diff_pos = {0: "fm", 1: "mm", 2: "rm", 3: "mm_v2", 4: "mm_v3", 5: "ms"}

    read_head_data = ReadPCAPData(args)
    read_end_data = ReadPCAPData(args)

    if args.test_date in ["26", "27"]:
        if args.diff_time:
            pos = diff_time
            for i in args.idx:
                read_head_data.read_pcap(f"{args.train_date}_head_netgear_{pos[i]}")
                read_end_data.read_pcap(f"{args.test_date}_end_netgear_{pos[i]}")
        elif args.diff_pos:
            pos = diff_pos
            for i in args.idx:
                read_head_data.read_pcap(f"{args.train_date}_head_netgear_{pos[i]}")
                read_end_data.read_pcap(f"{args.test_date}_end_netgear_{pos[i]}")
        elif args.diff_date:
            diff_date = ["26", "27"]
            pos = ["middle_v1_stop", "middle_stop"]
            for i in args.idx:
                read_head_data.read_pcap(f"{diff_date[i]}_head_netgear_{pos[i]}")
                read_end_data.read_pcap(f"{diff_date[i]}_end_netgear_{pos[i]}")
    elif args.test_date == "48":
        pos = diff_pos
        for i in args.idx:
            print(f"{args.test_date}_head_{pos[i]}")
            read_head_data.read_pcap(f"{args.test_date}_head_{pos[i]}")
            read_end_data.read_pcap(f"{args.test_date}_end_{pos[i]}")
    elif args.test_date == "19":
        pos = diff_pos
        for i in args.idx:
            print(f"{args.test_date}_head_{pos[i]}")
            read_head_data.read_pcap(f"{args.test_date}_head_netgear_{pos[i]}")
            read_end_data.read_pcap(f"{args.test_date}_end_netgear_{pos[i]}")

    read_end_data.read_to_np()
    read_head_data.read_to_np()

    print(f"read_head_data.train_csi {read_head_data.train_csi.shape}")
    print(f"read_head_data.test_csi {read_head_data.test_csi.shape}")
    print(f"read_end_data.train_csi {read_end_data.train_csi.shape}")
    print(f"read_end_data.test_csi {read_end_data.test_csi.shape}")

    test_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[526.3885],
                std=[309.8335],
            ),  # lp_sg 1000 612 815 1017 1031
        ]
    )

    print(f"read_head_data.train_csi {read_head_data.train_csi.shape}")
    print(f"read_head_data.test_csi {read_end_data.test_csi.shape}")

    train_head_data, train_head_rssi, train_head_label = (
        read_head_data.train_csi,
        read_head_data.train_rssi,
        read_head_data.train_label,
    )

    test_head_data, test_head_rssi, test_head_label = (
        read_head_data.test_csi,
        read_head_data.test_rssi,
        read_head_data.test_label,
    )

    train_end_data, train_end_rssi, train_end_label = (
        read_end_data.train_csi,
        read_end_data.train_rssi,
        read_end_data.train_label,
    )

    test_end_data, test_end_rssi, test_end_label = (
        read_end_data.test_csi,
        read_end_data.test_rssi,
        read_end_data.test_label,
    )

    # # Usage:
    # min_samples_train = 24
    # few_shot_train_idx = get_balanced_indices(
    #     data_processor.train_head_label, args.num_class, min_samples_train
    # )

    # print(f"few_shot_train_idx {len(few_shot_train_idx)}")

    # min_samples_test = min(
    #     [np.sum(data_processor.all_test_head_label == i) for i in range(args.num_class)]
    # )
    # few_shot_test_idx = get_balanced_indices(
    #     data_processor.all_test_head_label, args.num_class, min_samples_test
    # )

    # print(f"few_shot_test_idx {len(few_shot_test_idx)}")

    train_end_dataset = CsiRssiLabeled(
        [train_end_data, train_end_rssi],
        [train_end_label],
        transform=test_transform,
    )

    train_head_dataset = CsiRssiLabeled(
        [train_head_data, train_head_rssi],
        [train_head_label],
        transform=test_transform,
    )

    train_end_dataloader = DataLoader(
        train_end_dataset,
        batch_size=32,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    train_head_dataloader = DataLoader(
        train_head_dataset,
        batch_size=32,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"train_head_data shape {train_head_data.shape}")
    print(f"train_end_data shape {train_end_data.shape}")
    print(f"train_end_rssi shape {train_end_rssi.shape}")
    print(f"train_end_label shape {train_end_label.shape}")

    test_end_dataset = CsiRssiLabeled(
        [test_end_data, test_end_rssi], [test_end_label], transform=test_transform
    )

    test_head_dataset = CsiRssiLabeled(
        [test_head_data, test_head_rssi],
        [test_head_label],
        transform=test_transform,
    )

    test_end_dataloader = DataLoader(
        test_end_dataset,
        batch_size=args.duration_samples,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    test_head_dataloader = DataLoader(
        test_head_dataset,
        batch_size=args.duration_samples,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"test_head_data shape {test_head_data.shape}")
    print(f"test_end_data shape {test_end_data.shape}")
    print(f"test_end_rssi shape {test_end_rssi.shape}")
    print(f"test_end_label shape {test_end_label.shape}")

    if args.diff_time:
        print(f"args.diff_time {args.diff_time}")
        checkpoint_path = os.path.join(
            os.getcwd(),
            f"{args.pretrain_date}_adapt_results/{args.pretrain_date}_diff_time_all_memory.pkl",
        )
    elif args.diff_pos:
        print(f"args.diff_pos {args.diff_pos}")
        checkpoint_path = os.path.join(
            os.getcwd(),
            f"{args.pretrain_date}_adapt_results/{args.pretrain_date}_diff_pos_all_memory.pkl",
        )
    elif args.test_date == "19":
        checkpoint_path = os.path.join(
            os.getcwd(),
            f"{args.pretrain_date}_adapt_results/{args.pretrain_date}_all_memory.pkl",
        )

    checkpoint = torch.load(
        checkpoint_path,
    )

    algorithm_dict = checkpoint["model_dict"]

    print(
        f'algorithm_dict["classifier.memory"].shape: {algorithm_dict["classifier.memory"].shape}'
    )
    print(f'classes : {torch.unique(algorithm_dict["classifier.memory_label"])}')

    merge_model = AttentionMerge(args, num_classes=15)
    # mlp_classifier = copy.deepcopy(merge_model.classifier)
    # mlp_classifier.to(device)
    merge_model.classifier = nn.Identity()

    hparams = {
        "queue_size": algorithm_dict["classifier.memory"].shape[0],
        "lr": checkpoint["model_hparams"]["lr"],
        "weight_decay": checkpoint["model_hparams"]["weight_decay"],
        "temperature": args.temperature,
        "k": args.k,
    }

    adapt_hparams = {
        "beta": 0.9,
        "k": args.k,
        # 'eps': 0.9,
        "temperature": args.temperature,
    }

    knn_model = KNN(
        backbone=merge_model,
        num_classes=args.num_test_class,
        # num_classes=args.num_class,
        num_domains=1,
        hparams=hparams,
        input_shape=(1000, 242),
        device=device,
        scaler=None,
        bank=args.bank_type,
    )

    knn_model.load_state_dict(algorithm_dict)

    # backbone_checkpoint = torch.load(
    #     os.path.join(
    #         os.getcwd(),
    #         f"../ssd/results/{backbone_date[1]}_adapt_results/{backbone_date[1]}_merge_model.pth",
    #     )
    # )

    # knn_model.featurizer.load_state_dict(backbone_checkpoint, state_dict=False)
    # # knn_model.featurizer.backbone = reparameterize_model(
    # #     knn_model.featurizer.backbone
    # # )
    # knn_model.featurizer.eval()

    knn_model.to(device)

    adapted_model_algorithm = AdaNPC(
        input_shape=(1000, 242),
        num_classes=args.num_test_class,
        # num_classes=args.num_class,
        num_domains=1,
        hparams=adapt_hparams,
        algorithm=knn_model,
        bank=args.bank_type,
    )

    print("model loaded")

    unique_label = torch.unique(knn_model.classifier.memory_label).tolist()
    print(f"unique_label {unique_label}")
    for i in unique_label:
        print(
            f"len of label {i} : {torch.sum(knn_model.classifier.memory_label == i).item()}",
        )

    # if few_shot_idx != []:

    loader1, z, y = generate_merge_featurelized_loader(
        train_head_dataloader,
        train_end_dataloader,
        network=knn_model.featurizer,
        # knn_classifier=knn_model.classifier,
        # mlp_classifier=mlp_classifier,
        batch_size=50,
        device=device,
    )
    knn_model.classifier.update_class_queue(z.to(device), y.to(device))
    print(f"merge model.classifier.memory.shape {knn_model.classifier.memory.shape}")

    # else:
    #     loader1, loader2, train_knn_ent, z, y, train_mlp_ent = (
    #         generate_merge_featurelized_loader(
    #             train_head_dataloader,
    #             train_end_dataloader,
    #             network=head_model.featurizer,
    #             knn_classifier=head_model.classifier,
    #             mlp_classifier=mlp_classifier,
    #             batch_size=50,
    #             device=device,
    #         )
    #     )
    #     head_model.classifier.update_class_queue(z.to(device), y.to(device))
    #     print(
    #         f"merge model.classifier.memory.shape {head_model.classifier.memory.shape}"
    #     )

    if args.bank_type == "anchor":
        knn_model.classifier.build_anchor_bank()
        print(
            f"merge model.classifier.anchor_memory.shape {knn_model.classifier.anchor_memory.shape}"
        )
    print("extend feature memory bank")
    # print(f"train knn_ent {train_knn_ent}")
    # print(f"train mlp_ent {train_mlp_ent}")

    # # If knn_ent and mlp_ent are PyTorch tensors, convert them to NumPy arrays
    # knn_ent_np = train_knn_ent.cpu().numpy()
    # mlp_ent_np = train_mlp_ent.cpu().numpy()

    # generate test feature loader
    loaders = []

    loader1, z, y = generate_merge_featurelized_loader(
        test_head_dataloader,
        test_end_dataloader,
        network=knn_model.featurizer,
        # knn_classifier=knn_model.classifier,
        # mlp_classifier=mlp_classifier,
        batch_size=args.duration_samples,
        device=device,
    )
    loaders.append(loader1)
    # knn_model.classifier.extend_test(z.to(device), y.to(device))
    print(f"merge model.classifier.memory.shape {knn_model.classifier.memory.shape}")
    print("generate feature loader")
    # print(f"test knn_ent {test_knn_ent}")
    # print(f"test mlp_ent {test_mlp_ent}")

    results = adapt_hparams

    print("start testing")

    unique_label = torch.unique(knn_model.classifier.memory_label).tolist()
    print(f"unique_label {unique_label}")
    for i in unique_label:
        print(
            f"len of label {i} : {torch.sum(knn_model.classifier.memory_label == i).item()}",
        )
        print(
            f"len of label {i} : {torch.where(knn_model.classifier.memory_label == i)}",
        )

    if args.long_duration_pred:
        if args.use_knn:

            ent, auc, eval_metrics, y_prob, y_true = knn_merge_long_test(
                adapted_model_algorithm,
                loaders,
                device,
                args.num_class,
            )

        # else:
        #     ent, auc, eval_metrics, y_prob, y_true = cnn_merge_long_test(
        #         mlp_classifier,
        #         loaders,
        #         device,
        #         args.num_class,
        #     )

    else:
        ent, auc, eval_metrics, y_prob, y_true = knn_merge_short_test(
            adapted_model_algorithm,
            loaders,
            device,
            args.num_class,
        )

    results["accuracy"] = eval_metrics[0]
    results["precision"] = eval_metrics[1]
    results["recall"] = eval_metrics[2]
    results["f1score"] = eval_metrics[3]
    results["mae"] = eval_metrics[5]
    results["auc"] = auc
    results["ent"] = ent

    results_keys = sorted(results.keys())

    misc.print_row(results_keys, colwidth=20)
    misc.print_row([results[key] for key in results_keys], colwidth=20)

    args.cls = np.unique(train_head_label).shape[0]
    print(f"args.cls {args.cls}")

    print(
        f"adapted_head_algorithm.model.classifier.memory.shape {adapted_model_algorithm.model.classifier.memory.shape}"
    )

    # save_checkpoint(
    #     args=args,
    #     model=adapted_head_algorithm.model,
    #     filename=os.path.join(
    #         os.getcwd(),
    #         f"../ssd/results/{args.train_date}_adapt_results/{args.train_date}_all_memory.pkl",
    #     ),
    #     input_shape=(1000, 234),
    # )
