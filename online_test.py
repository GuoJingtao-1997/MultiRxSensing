"""
Author: Guo Jingtao
Date: 2025-04-28 23:29:53
LastEditTime: 2025-04-28 23:29:53
LastEditors: Guo Jingtao
Description:
FilePath: /MultiRxSensing/online_data_collect_test copy.py

"""

import numpy as np
import torch
import torchvision.transforms as transforms
from mobileones import mobileone, reparameterize_model
import os
import multiprocessing as mp
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from adapt_algo import AdaNPC, KNN
from utils import setup_seed
from fusion import realtime_csi_read
from merge_model import AttentionMerge
import logging
import sys
from knn import MomentumQueue
import threading

# import subprocess
import time
import paho.mqtt.client as paho
import argparse
from scipy.stats import zscore
import torch.nn as nn
from multiprocessing import active_children
from collections import defaultdict

PCAP_PATH = os.path.join(os.getcwd(), "../ssd/csi_data")
VIEW_LIST = ["head", "end"]
PACKETS = 1000
NULL_SC = True
PILOT_SC = False
# paused = False
lock = threading.Lock()

ACCESS_TOKEN = "edgeai"
broker = "demo.thingsboard.io"  # host name
port = 1883  # data listening port

args = None
global_num_train_times = 0
last_pred = "Loading..."


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


test = paho.Client(paho.CallbackAPIVersion.VERSION2, "device1")
test.on_publish = on_publish
test.on_connect = on_connect
test.username_pw_set(ACCESS_TOKEN)
test.connect(broker, port, keepalive=60)
test.loop_start()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument(
        "--k",
        default=100,
        type=int,
        choices=(1, 4, 5, 10, 20, 25, 50, 75, 100, 150, 200),
        help="number of nearest neighbors",
    )

    parser.add_argument(
        "--haddr",
        type=str,
        default="192.168.1.38",
        metavar="N",
        help="eth address of head csi data",
    )

    parser.add_argument(
        "--eaddr",
        type=str,
        default="192.168.1.26",
        metavar="N",
        help="eth address of end csi data",
    )

    parser.add_argument(
        "--threshold",
        type=str,
        default="0.1",
        metavar="N",
        help="threshold for result uploading",
    )

    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--empty_reset", action="store_true")

    parser.add_argument(
        "--adapt",
        action="store_true",
        help="auto extend memory bank based on the predicted label",
    )

    parser.add_argument(
        "--single_load",
        action="store_true",
        help="single status model loading",
    )

    parser.add_argument(
        "--double_load",
        action="store_true",
        help="double status model loading",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.1,
        metavar="N",
        help="temperature for softmax",
    )

    parser.add_argument(
        "--pretrained_date",
        type=str,
        default="725",
        metavar="N",
        help="pretrained model date",
    )

    parser.add_argument(
        "--pretrained_status",
        type=str,
        default="ft_all",
        metavar="N",
        help="pretrained model status",
    )

    parser.add_argument(
        "--save_date",
        type=str,
        default="801",
        metavar="N",
        help="data collection date",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="stationary",
        metavar="N",
        help="status of the data collection",
    )

    parser.add_argument(
        "--passenger",
        type=int,
        default=20,
        metavar="N",
        help="# of passenger in dataset",
    )

    parser.add_argument(
        "--predict_tolerance",
        type=int,
        default=3,
        metavar="N",
        help="prediction tolerance",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.15,
        metavar="N",
        help="beta for AdaNPC",
    )

    parser.add_argument(
        "--predict_duration",
        type=int,
        default=50,
        metavar="N",
        help="# of files (1s * #) per prediction",
    )

    parser.add_argument(
        "--aug",
        type=str,
        default="hp_sub_sg",
        metavar="N",
        help="data pattern",
    )

    parser.add_argument(
        "--num_class",
        type=int,
        default=14,
        metavar="N",
        help="# of class for testing",
    )

    args = parser.parse_args()
    return args


def save_checkpoint(args, model, filename, input_shape):
    if args.skip_model_save:
        return
    print("save model to ", filename)
    save_dict = {
        "args": vars(args),
        "model_input_shape": input_shape,
        # "model_num_classes": args.cls,
        "model_num_domains": 1,
        "model_hparams": model.hparams,
        "model_dict": model.cpu().state_dict(),
    }
    torch.save(save_dict, filename)


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


def model_inference(
    args,
    CSI_DICT,
    merge_model,
    left_csi_minibatches,
    right_csi_minibatches,
    adapt,
):
    print('len(CSI_DICT["head"]) ', len(CSI_DICT["head"]))
    print(f"pid is {os.getpid()}")

    print("Finish Testing and Data Collection")
    print("start model inference")

    y_prob, y_true, final_prob = collaborate_knn_test(
        args=args,
        merge_model=merge_model,
        left_minibatch=left_csi_minibatches,
        right_minibatch=right_csi_minibatches,
        adapt=adapt,
    )

    return y_prob, y_true, final_prob


def knn_model_load(args, checkpoint, device):

    pretrained_args = checkpoint["args"]
    print(f"pre-trained args {pretrained_args}")
    print(f"model_hparams {checkpoint['model_hparams']}")
    # print(f"model_dict {checkpoint['model_dict'].keys()}")

    backbone = AttentionMerge(args, num_classes=args.num_class)
    backbone.classifier = nn.Identity()

    algorithm_dict = checkpoint["model_dict"]

    print(
        f'algorithm_dict["classifier.memory"].shape: {algorithm_dict["classifier.memory"].shape}'
    )
    print(f'classes : {torch.unique(algorithm_dict["classifier.memory_label"])}')

    hparams = {
        "queue_size": algorithm_dict["classifier.memory"].shape[0],
        "lr": checkpoint["model_hparams"]["lr"],
        "weight_decay": checkpoint["model_hparams"]["weight_decay"],
        "temperature": args.temp,
        "k": args.k,
    }

    print(f"hparams {hparams}")

    adapt_hparams = {
        "beta": args.beta,
        "k": args.k,
        # 'eps': 0.9,
        "temperature": args.temp,
    }

    model = KNN(
        backbone=backbone,
        num_classes=args.num_class,
        num_domains=1,
        hparams=hparams,
        input_shape=(1000, 242),
        device=device,
        scaler=None,
        bank="memory",
    )

    model.load_state_dict(algorithm_dict)

    # model.featurizer.backbone = reparameterize_model(model.featurizer.backbone)
    model.featurizer.eval()

    model = model.to(device)

    adapted_algorithm = AdaNPC(
        input_shape=(1000, 242),
        num_classes=args.num_class,
        num_domains=1,
        hparams=adapt_hparams,
        algorithm=model,
        bank="memory",
    )

    return adapted_algorithm


def model_load(args, checkpoint_path: str, device):
    model = mobileone(
        num_classes=args.num_class, variant="s4", inference_mode=False
    ).to(device)
    checkpoint = torch.load(
        os.path.join(
            os.getcwd(),
            "612_adapt_results/head_end_13_knn_False",
            f"612_ft_stop_drive_{checkpoint_path}_mos4.pth",
        ),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint)
    # weight_norm_layer = list(model.children())[-1]
    model = reparameterize_model(model)
    model.eval()
    return model


def collaborate_knn_test(
    args, merge_model, left_minibatch, right_minibatch, adapt=True
):
    with torch.no_grad():
        for (xh, yh, rh), (xe, _, re) in zip(left_minibatch, right_minibatch):

            z = merge_model.model.featurizer(
                {
                    "x1": xh,
                    "x2": xe,
                    "rssi1": rh,
                    "rssi2": re,
                }
            )

            output = merge_model(z, adapt_empty=adapt)

            print(f"yh {yh}")

            print(f"output shape {output.shape}")

            y_prob_merge = torch.softmax(output, dim=1).detach().cpu().numpy()

            print(f"y_prob_merge shape {y_prob_merge.shape}")

            y_label = torch.argmax(output, dim=1).detach().cpu().numpy()

            print(f"y_label_head shape {y_label.shape}")

            prob_avg = np.zeros_like(y_prob_merge[0])
            count = 0

            for prob in y_prob_merge:
                count += 1
                prob_avg += prob

            prob_avg /= count

            final_pred = np.argmax(prob_avg)
            final_prob = prob_avg[final_pred]
            y = torch.sum(yh) // yh.shape[0]

            # print(f"output {output}")
            print(f"prob_avg {prob_avg}")
            print(f"final_prob {final_prob}")
            print(f"y {y}")
            print(f"final pred {final_pred}")

        y_prob = final_pred
        y_true = y.detach().cpu().numpy()

    return y_prob, y_true, final_prob


def setup_logger():
    logger = logging.getLogger("worker")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main_task():

    global args, last_pred, SRC_ADDR_DICT, global_num_train_times
    global_num_train_times = args.num_train_times
    predict_tolerance = args.predict_tolerance
    history_passenger = args.passenger
    SRC_ADDR_DICT = {"head": args.haddr, "end": args.eaddr}
    print(f"SRC_ADDR_DICT {SRC_ADDR_DICT}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    save_folder = os.path.join(
        os.getcwd(),
        f"{args.save_date}_adapt_results/",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if args.single_load:
        if os.path.exists(
            os.path.join(
                save_folder,
                f"{args.save_date}_{args.status}.pkl",
            )
        ):
            print(f"load recent updated model {args.save_date}")
            checkpoint = torch.load(
                os.path.join(
                    os.getcwd(),
                    f"{args.save_date}_adapt_results/{args.save_date}_{args.status}.pkl",
                )
            )
        else:
            print(f"load history model {args.pretrained_date}")
            checkpoint = torch.load(
                os.path.join(
                    os.getcwd(),
                    f"{args.pretrained_date}_adapt_results",
                    f"{args.pretrained_date}_{args.pretrained_status}.pkl",
                )
            )

        print("load model")
        merge_model = knn_model_load(args, checkpoint, device)
    elif args.double_load:
        print(f"load recent updated model with moving and stationary")
        checkpoint_stationary = torch.load(
            os.path.join(
                os.getcwd(),
                f"{args.save_date}_adapt_results/{args.save_date}_ms.pkl",
            )
        )
        checkpoint_moving = torch.load(
            os.path.join(
                os.getcwd(),
                f"{args.save_date}_adapt_results/{args.save_date}_mm.pkl",
            )
        )

        print("load model stop")
        merge_model = knn_model_load(args, checkpoint_stationary, device)

        print("load model move")
        merge_model_move = knn_model_load(args, checkpoint_moving, device)

        merge_model.model.classifier.extend_test(
            merge_model_move.model.classifier.memory,
            merge_model_move.model.classifier.memory_label,
        )

    print(f"merge_model {merge_model.model.classifier.memory.shape}")
    print(f"merge_model_label {merge_model.model.classifier.memory_label.shape}")

    test_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[526.3885],
                std=[309.8335],
            ),
        ]
    )
    person_classes = [str(i) for i in range(args.num_class)]
    print("person_classes ", person_classes)

    print("start testing")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(f"Total CPU Processes: {mp.cpu_count()}")
    CSI_DICT = defaultdict(list)
    RSSI_DICT = defaultdict(list)
    pred_results = []
    start.record()
    adapt = args.adapt
    i = 1
    while args.passenger != -1:
        try:
            if os.path.exists(
                os.path.join(
                    PCAP_PATH,
                    f"{args.save_date}_head_{args.status}/{args.passenger}",
                )
            ):
                i = (
                    len(
                        os.listdir(
                            os.path.join(
                                PCAP_PATH,
                                f"{args.save_date}_head_{args.status}/{args.passenger}",
                            )
                        )
                    )
                    + 1
                )
            else:
                i = 1
            p = mp.Pool(6)
            if len(CSI_DICT["head"]) == args.predict_duration:
                CSI_DICT = defaultdict(list)
                RSSI_DICT = defaultdict(list)
            result_list = {}
            for view in VIEW_LIST:
                result_list[view] = p.apply_async(
                    realtime_csi_read,
                    args=(
                        i,
                        args.passenger,
                        view,
                        args.save_date,
                        args.status,
                        SRC_ADDR_DICT,
                    ),
                )

            CSI_DICT["head"].append(result_list["head"].get()[0][np.newaxis, :])
            CSI_DICT["end"].append(result_list["end"].get()[0][np.newaxis, :])

            RSSI_DICT["head"].append(result_list["head"].get()[1][np.newaxis, :])
            RSSI_DICT["end"].append(result_list["end"].get()[1][np.newaxis, :])

            print(f"len of CSI_DICT {len(CSI_DICT['head'])}")
            print(f"file index {i}")

            i += 1

            # use ground truth label for testing
            CSI_DICT["head_true_label"].append(args.passenger)
            CSI_DICT["end_true_label"].append(args.passenger)

            with lock:
                if len(CSI_DICT["head"]) == args.predict_duration:
                    left_csi_minibatches = [
                        (
                            test_transform(
                                torch.tensor(
                                    np.array(CSI_DICT["head"]),
                                    dtype=torch.float32,
                                )
                            ).to(device),
                            torch.tensor(
                                np.array(CSI_DICT["head_true_label"]),
                                dtype=torch.long,
                            ).to(device),
                            test_transform(
                                torch.tensor(
                                    np.array(RSSI_DICT["head"]),
                                    dtype=torch.float32,
                                )
                            ).to(device),
                        )
                    ]

                    right_csi_minibatches = [
                        (
                            test_transform(
                                torch.tensor(
                                    np.array(CSI_DICT["end"]),
                                    dtype=torch.float32,
                                )
                            ).to(device),
                            torch.tensor(
                                np.array(CSI_DICT["end_true_label"]),
                                dtype=torch.long,
                            ).to(device),
                            test_transform(
                                torch.tensor(
                                    np.array(RSSI_DICT["end"]),
                                    dtype=torch.float32,
                                )
                            ).to(device),
                        )
                    ]

                    pred_results.append(
                        model_inference(
                            args,
                            CSI_DICT,
                            merge_model,
                            left_csi_minibatches,
                            right_csi_minibatches,
                            person_classes,
                            adapt,
                        )
                    )

                    last_pred = pred_results[-1][0]
                    final_prob = pred_results[-1][2]

                    if final_prob > args.threshold:
                        payload = "{"
                        payload += f'"People":{last_pred}'
                        payload += "}\n"
                        ret = test.publish("v1/devices/me/telemetry", payload)

                    print(f"There are {person_classes[last_pred]} passengers\n")
                    print("Finish model inference")
                    print(f"passenger {args.passenger}")

            p.close()
            p.join()

        except KeyboardInterrupt as e:
            children = active_children()
            print(f"Active children before terminating: {len(children)}")
            print(f"{e}, terminating the pool")
            p.terminate()
            print("Pool terminated")
            children = active_children()
            print(f"Active children before terminating: {len(children)}")
            break

    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    infer_time = start.elapsed_time(end)

    print(f"total inference time {infer_time} ms, {infer_time / 1000} s\n")

    if args.single_load:
        status = args.status
    elif args.double_load:
        status = "mm_ms"

    save_checkpoint(
        args=args,
        model=merge_model.model,
        filename=os.path.join(
            save_folder,
            f"{args.save_date}_{status}.pkl",
        ),
        input_shape=(1000, 242),
    )

    y_prob = [res[0] for res in pred_results]
    y_true = [int(res[1]) for res in pred_results]
    print(pred_results)

    print(f"y_prob {y_prob}")
    print(f"y_true {y_true}")

    eval_metrics = [
        metrics.accuracy_score(y_true, y_prob),
        metrics.precision_score(y_true, y_prob, average="macro"),
        metrics.recall_score(y_true, y_prob, average="macro"),
        metrics.f1_score(y_true, y_prob, average="macro"),
        metrics.confusion_matrix(y_true, y_prob),
        metrics.mean_absolute_error(y_true, y_prob),
    ]

    print(f"Test Accurancy: {eval_metrics[0]:.4%}")
    print(f"Test Precision: {eval_metrics[1]:.4%}")
    print(f"Test Recall: {eval_metrics[2]:.4%}")
    print(f"Test F1-Score: {eval_metrics[3]:.4%}")
    print(f"Test MAE: {eval_metrics[5]:.4}")

    auc = metrics.roc_auc_score(
        label_binarize(y_true, classes=np.arange(args.passenger, args.passenger + 1)),
        label_binarize(y_prob, classes=np.arange(args.passenger, args.passenger + 1)),
        average="micro",
        multi_class="ovr",
    )

    print(f"Test AUC: {auc:.4f}")

    args.cls = args.num_class


if __name__ == "__main__":
    setup_logger()
    setup_seed(42)
    print("seed: 42")
    print("PyTorch Version: ", torch.__version__)
    print("setup torch device")
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    main_task()
