import numpy as np
import torch
import torchvision.transforms as transforms
from mobileones import mobileone, reparameterize_model
import os
import multiprocessing as mp
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import label_binarize·
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

args = None
global_num_train_times = 0


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

    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--empty_reset", action="store_true")
    parser.add_argument("--exit", action="store_true")

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

    # parser.add_argument(
    #     "--update_queue",
    #     action="store_true",
    #     help="update the memory bank with a fixed size",
    # )

    parser.add_argument(
        "--ground_truth_add_count",
        type=int,
        default=0,
        metavar="N",
        help="add ground truth label to the knn memory bank",
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
        "--num_train_times",
        type=int,
        default=1,
        metavar="N",
        help="# of files update in memory bank per counting scenario",
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


# def csi_collect(stop_event, args):
#     i = 1
#     while not stop_event.is_set():
#         p = mp.Pool(2)
#         for view in VIEW_LIST:
#             p.apply_async(
#                 realtime_csi_read,
#                 args=(
#                     i,
#                     args.passenger,
#                     view,
#                     args.save_date,
#                     args.status,
#                     SRC_ADDR_DICT,
#                 ),
#             )
#         p.close()
#         p.join()
#         i += 1


def publisher_thread(stop_event, test):
    while not stop_event.is_set():
        payload = "{"
        payload += f'"People":{args.passenger}'
        payload += "}\n"
        test.publish("v1/devices/me/telemetry", payload)
        print(f"Published: {payload}")
        time.sleep(10)  # Publish every second


def edit_args():
    global args, test

    # history_status = args.status
    stop_publisher = threading.Event()

    # Start publisher process
    publisher = threading.Thread(target=publisher_thread, args=(stop_publisher, test))
    publisher.daemon = True
    publisher.start()

    # csi_collection = mp.Process(target=csi_collect, args=(stop_publisher, args))
    # csi_collection.daemon = False

    while True:
        print("\nCurrent Settings:")
        print(f"Passenger: {args.passenger}")
        print(f"Predict Duration: {args.predict_duration}")
        print(f"Number of Train Files: {args.num_train_times}")
        print(f"Status: {args.status}")

        print("\nEdit Options:")
        print("1. Edit Passenger")
        print("2. Edit Predict Duration")
        print("3. Edit Number of Train Files")
        print("4. Edit Name of Status")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            while True:
                new_input = input("Enter new passenger (or 'b' to go back): ")
                if new_input.lower() == "b":
                    break
                try:
                    new_passenger = int(new_input)
                    if new_passenger >= 0:
                        args.passenger = new_passenger
                        print(f"Passenger updated to: {args.passenger}")
                        break
                    else:
                        print("Error: Passenger must be positive")
                except ValueError:
                    print("Error: Please enter a valid number")

        elif choice == "2":
            while True:
                new_input = input("Enter new predict duration (or 'b' to go back): ")
                if new_input.lower() == "b":
                    break
                try:
                    new_duration = int(new_input)
                    if new_duration > 0:
                        args.predict_duration = new_duration
                        print(f"Predict duration updated to: {args.predict_duration}")
                        break
                    else:
                        print("Error: Duration must be positive")
                except ValueError:
                    print("Error: Please enter a valid number")

        elif choice == "3":
            while True:
                new_input = input(
                    "Enter new number of training times (or 'b' to go back): "
                )
                if new_input.lower() == "b":
                    break
                try:
                    num_train_times = int(new_input)
                    if num_train_times >= 0:
                        args.num_train_times = num_train_times
                        print(
                            f"Number of train times updated to: {args.num_train_times}"
                        )
                        break
                    else:
                        print("Error: Number of times must be positive")
                except ValueError:
                    print("Error: Please enter a valid number")

        elif choice == "4":
            while True:
                new_input = input("Enter new name of folder (or 'b' to go back): ")
                if new_input.lower() == "b":
                    break
                try:
                    args.status = new_input
                except ValueError:
                    print("Error: Please enter a valid number")

        # elif choice == "4":
        #     while True:
        #         new_input = input(
        #             "Enter new name of folder (or 'b' to go back): "
        #         ).strip()
        #         if new_input.lower() == "b":
        #             break
        #         if new_input:  # Check if string is not empty
        #             args.status = f"netgear_{new_input}"
        #             print(f"Name of folder updated to: {args.status}")
        #             csi_collection.start()
        #             break
        #         else:
        #             print("Error: Name of folder must not be empty")

        elif choice == "5":
            # args.status = history_status
            print("Exiting edit mode")
            stop_publisher.set()  # Signal the publisher thread to stop
            publisher.join()  # Wait for publisher thread to finish
            # csi_collection.join()
            # Ensure processes are terminated
            # if publisher.is_alive():
            #     publisher.terminate()
            # if csi_collection.is_alive():
            #     csi_collection.terminate()
            break
        else:
            print("Invalid choice. Please select 1-5")


def pause_input_handler():
    """Handles user input to pause/resume processing or quit the program."""
    global args, paused

    while True:
        try:
            user_input = (
                input(
                    "Enter 'p' to toggle pause and edit args, 'q' to quit the whole program: "
                )
                .lower()
                .strip()
            )

            print(f"User input: {user_input}")

            if user_input == "p":
                with lock:
                    try:
                        # paused = not paused
                        edit_args()
                    except Exception as e:
                        print(f"Error editing args: {str(e)}")
                        print("Continuing with existing args...")

            elif user_input == "q":
                print("Exiting program...")
                args.exit = True
                sys.exit(0)

            elif not user_input:
                print("Empty input, please try again")
                continue

            else:
                print(f"Invalid input: '{user_input}'. Please enter 'p' or 'q'")
                continue

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Use 'q' to quit properly.")
            continue

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print("Please try again")
            continue


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
    # predict_duration,
    args,
    CSI_DICT,
    merge_model,
    left_csi_minibatches,
    right_csi_minibatches,
    person_classes,
    adapt,
):
    print('len(CSI_DICT["head"]) ', len(CSI_DICT["head"]))
    print(f"pid is {os.getpid()}")

    print("Finish Testing and Data Collection")
    print("start model inference")

    y_prob, y_true = collaborate_knn_test(
        args=args,
        merge_model=merge_model,
        left_minibatch=left_csi_minibatches,
        right_minibatch=right_csi_minibatches,
        adapt=adapt,
    )

    return y_prob, y_true


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

    if "empty_memory.memory" in algorithm_dict:
        print(
            f'algorithm_dict["empty_memory.memory"].shape: {algorithm_dict["empty_memory.memory"].shape}'
        )
        print(f'classes : {torch.unique(algorithm_dict["empty_memory.memory_label"])}')

        model.extend_classifier = MomentumQueue(
            model.featurizer.backbone.num_features,
            algorithm_dict["extend_classifier.memory"].shape[0],
            model.hparams["temperature"],
            model.hparams["k"],
            model.num_classes,
            bank=model.bank,
        )
        model.empty_memory = MomentumQueue(
            model.featurizer.backbone.num_features,
            algorithm_dict["empty_memory.memory"].shape[0],
            model.hparams["temperature"],
            model.hparams["k"],
            model.num_classes,
            bank=model.bank,
        )

        model.load_state_dict(algorithm_dict)

        # if args.empty_reset:
        #     model.empty_memory = MomentumQueue(
        #         model.featurizer.backbone.num_features,
        #         0,
        #         model.hparams["temperature"],
        #         model.hparams["k"],
        #         model.num_classes,
        #         bank=model.bank,
        #     )

    else:
        model.load_state_dict(algorithm_dict)

        model.extend_classifier = MomentumQueue(
            model.featurizer.backbone.num_features,
            algorithm_dict["classifier.memory"].shape[0],
            model.hparams["temperature"],
            model.hparams["k"],
            model.num_classes,
            bank=model.bank,
        )
        model.empty_memory = MomentumQueue(
            model.featurizer.backbone.num_features,
            0,
            model.hparams["temperature"],
            model.hparams["k"],
            model.num_classes,
            bank=model.bank,
        )

    # model.featurizer.backbone = reparameterize_model(model.featurizer.backbone)
    model.featurizer.eval()

    print(
        f"model.extend_classifier.memory.shape: {model.extend_classifier.memory.shape}"
    )

    print(f"model.empty_memory.memory.shape: {model.empty_memory.memory.shape}")

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

            # output = merge_model(z, adapt_empty=adapt)
            output = merge_model.model.empty_memory(z)

            print(f"yh {yh}")

            print(f"output shape {output.shape}")

            y_prob_merge = torch.softmax(output, dim=1).detach().cpu().numpy()

            print(f"y_prob_merge shape {y_prob_merge.shape}")

            y_label = torch.argmax(output, dim=1).detach().cpu().numpy()

            print(f"y_label_head shape {y_label.shape}")

            # merge_output = remove_outliers_sliding_window(
            #     y_label, y_prob_merge, window_size=3, z_threshold=3
            # )

            # print(f"merge_output shape {merge_output.shape}")

            # merge_output = np.argmax(y_prob_merge, axis=1)

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

            prob_avg = np.zeros_like(y_prob_merge[0])
            count = 0

            for prob in y_prob_merge:
                count += 1
                prob_avg += prob

            prob_avg /= count

            final_pred = np.argmax(prob_avg)
            y = torch.sum(yh) // yh.shape[0]

            # print(f"output {output}")
            print(f"prob_avg {prob_avg}")
            print(f"y {y}")
            print(f"final pred {final_pred}")

        y_prob = final_pred
        y_true = y.detach().cpu().numpy()

    return y_prob, y_true


def collaborate_pre_test(
    head_model,
    end_model,
    test_head_loader,
    test_end_loader,
    device,
):
    # metric = Accumulator([[]], 2)
    test_accumulator = 0

    y_prob = []
    y_true = []

    with torch.no_grad():
        for (xh, yh), (xe, ye) in tqdm(
            zip(test_head_loader, test_end_loader), total=len(test_head_loader)
        ):
            xh, xe = xh.to(device), xe.to(device)
            yh, ye = yh.to(device), ye.to(device)
            oh = head_model(xh)
            oe = end_model(xe)

            output = (oh + oe) / 2
            y = (yh + ye) / 2

            test_accumulator = test_accumulator + np.sum(yh.shape[0] + ye.shape[0]) / 2

            y_prob.append(torch.argmax(output, dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            # metric.add(
            #     y.detach().cpu().numpy().tolist(),
            #     torch.argmax(output, dim=-1).detach().cpu().numpy().tolist(),
            # )

    y_prob = np.concatenate(y_prob, axis=0)

    y_true = np.concatenate(y_true, axis=0)

    eval_metrics = [
        metrics.accuracy_score(y_true, y_prob),
        metrics.precision_score(y_true, y_prob, average="macro"),
        metrics.recall_score(y_true, y_prob, average="macro"),
        metrics.f1_score(y_true, y_prob, average="macro"),
        metrics.confusion_matrix(y_true, y_prob),
    ]

    auc = metrics.roc_auc_score(
        label_binarize(y_true, classes=np.arange(5)),
        label_binarize(y_prob, classes=np.arange(5)),
        average="micro",
    )

    print(f"Test Accurancy: {eval_metrics[0]:.4%}")
    print(f"Test Precision: {eval_metrics[1]:.4%}")
    print(f"Test Recall: {eval_metrics[2]:.4%}")
    print(f"Test F1-Score: {eval_metrics[3]:.4%}")
    print(f"Test AUC: {auc:.4f}")

    return test_accumulator, auc, eval_metrics, y_prob, y_true


def real_test(
    model,
    test_data,
    device,
):

    with torch.no_grad():
        test_data = test_data.to(device)
        output = model(
            test_data,
        )
    return output


def collaborate_real_test(
    head_model,
    end_model,
    test_head,
    test_end,
    device,
):

    with torch.no_grad():
        test_head, test_end = test_head.to(device), test_end.to(device)
        oh = head_model(test_head)
        oe = end_model(test_end)

        output = (oh + oe) / 2

    return output


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
            ),  # lp_sg 1000 612 815 1017 1031
        ]
    )
    person_classes = [str(i) for i in range(args.num_class)]
    history_num_train_times = args.num_train_times
    print("person_classes ", person_classes)

    print("start testing")

    # head_data_atten_npz = os.path.join(
    #     PCAP_PATH,
    #     f"datasets/{args.save_date}_head_{args.status}_tp2_hp_dwt_sg.npz",
    # )

    # end_data_hp_dwt_sg_npz = os.path.join(
    #     PCAP_PATH,
    #     f"datasets/{args.save_date}_end_{args.status}_tp2_hp_dwt_sg.npz",
    # )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(f"Total CPU Processes: {mp.cpu_count()}")
    CSI_DICT = defaultdict(list)
    RSSI_DICT = defaultdict(list)
    pred_results = []
    # head_data = []
    # end_data = []
    # head_data_label = []
    # end_data_label = []
    start.record()
    adapt = args.adapt
    i = 1
    while args.passenger != -1:
        try:
            if history_passenger != args.passenger:
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
                history_num_train_times = global_num_train_times
            elif global_num_train_times != args.num_train_times:
                history_num_train_times = args.num_train_times
                global_num_train_times = args.num_train_times
            p = mp.Pool(6)
            if (
                len(CSI_DICT["head"]) == args.predict_duration
                or history_passenger != args.passenger
            ):
                CSI_DICT = defaultdict(list)
                RSSI_DICT = defaultdict(list)
            history_passenger = args.passenger
            result_list = {}
            start_time = time.time()
            sub_start_time = time.time()
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

            # print(f'CSI_DICT["head"] {CSI_DICT["head"]}')

            print(f"len of CSI_DICT {len(CSI_DICT['head'])}")
            print(f"file index {i}")

            i += 1

            # use ground truth label for testing
            CSI_DICT["head_true_label"].append(args.passenger)
            CSI_DICT["end_true_label"].append(args.passenger)

            sub_end_time = time.time()

            # print(
            #     f"data collection time cost {sub_end_time - sub_start_time}, main process, pid {os.getpid()}, Waiting for all processes to finish\n"
            # )

            sub_start_time = time.time()
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

                    # print(f"add ground truth data to memory bank")

                    if history_num_train_times > 0:
                        if history_passenger < args.num_class:
                            merge_model.model.classifier.update_queue(
                                merge_model.model.featurizer(
                                    {
                                        "x1": left_csi_minibatches[0][0],
                                        "x2": right_csi_minibatches[0][0],
                                        "rssi1": left_csi_minibatches[0][2],
                                        "rssi2": right_csi_minibatches[0][2],
                                    }
                                ),
                                left_csi_minibatches[0][1],
                            )
                        else:
                            merge_model.model.classifier.extend_test(
                                merge_model.model.featurizer(
                                    {
                                        "x1": left_csi_minibatches[0][0],
                                        "x2": right_csi_minibatches[0][0],
                                        "rssi1": left_csi_minibatches[0][2],
                                        "rssi2": right_csi_minibatches[0][2],
                                    }
                                ),
                                left_csi_minibatches[0][1],
                            )

                        merge_model.model.extend_classifier.extend_test(
                            merge_model.model.featurizer(
                                {
                                    "x1": left_csi_minibatches[0][0],
                                    "x2": right_csi_minibatches[0][0],
                                    "rssi1": left_csi_minibatches[0][2],
                                    "rssi2": right_csi_minibatches[0][2],
                                }
                            ),
                            left_csi_minibatches[0][1],
                        )

                        merge_model.model.empty_memory.extend_test(
                            merge_model.model.featurizer(
                                {
                                    "x1": left_csi_minibatches[0][0],
                                    "x2": right_csi_minibatches[0][0],
                                    "rssi1": left_csi_minibatches[0][2],
                                    "rssi2": right_csi_minibatches[0][2],
                                }
                            ),
                            left_csi_minibatches[0][1],
                        )

                        history_num_train_times -= 1

                    print(f"num_train_times {history_num_train_times}")

                    # if not paused:
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
                    # else:
                    #     print("Paused, skipping inference")

                    if history_num_train_times == 0:
                        last_pred = pred_results[-1][0]
                        if pred_results[-1][0] != pred_results[-1][1]:
                            predict_tolerance -= 1
                            if predict_tolerance == 0:
                                last_pred = pred_results[-1][1]
                                merge_model.model.empty_memory.update_class_queue(
                                    merge_model.model.featurizer(
                                        {
                                            "x1": left_csi_minibatches[0][0],
                                            "x2": right_csi_minibatches[0][0],
                                            "rssi1": left_csi_minibatches[0][2],
                                            "rssi2": right_csi_minibatches[0][2],
                                        }
                                    ),
                                    pred_results[-1][1],
                                )
                                predict_tolerance = args.predict_tolerance
                        print(f"y_prob {last_pred}")
                    else:
                        last_pred = pred_results[-1][1]
                        print(f"y_true {last_pred}")

                    payload = "{"
                    payload += f'"People":{last_pred}'
                    payload += "}\n"
                    ret = test.publish("v1/devices/me/telemetry", payload)
                    # topic-v1/devices/me/telemetry
                    # print(
                    #     f"ret: {ret}, Please check LATEST TELEMETRY field of the device\n"
                    # )
                    # print(payload)
                    print(f"There are {person_classes[last_pred]} passengers\n")
                    print("Finish model inference")

                    # head_data.extend(CSI_DICT["head"])
                    # end_data.extend(CSI_DICT["end"])
                    # head_data_label.extend(CSI_DICT["head_true_label"])
                    # end_data_label.extend(CSI_DICT["end_true_label"])

                    print(f"passenger {args.passenger}")

            sub_end_time = time.time()

            print(
                f"merge_model.model.empty_memory.memory.shape: {merge_model.model.empty_memory.memory.shape}\n"
            )

            print(
                f"merge_model.model.classifier.memory.shape: {merge_model.model.classifier.memory.shape}\n"
            )

            print(
                f"'merge_model.model.classifier.class_indices' {merge_model.model.classifier.class_indices}"
            )

            print(
                f"'merge_model.model.classifier.class_positions' {merge_model.model.classifier.class_positions}"
            )

            # print(
            #     f"prediction time cost {sub_end_time - sub_start_time}, main process, pid {os.getpid()}, Waiting for all processes to finish\n"
            # )

            p.close()
            p.join()

            end_time = time.time()

            # print(
            #     f"time cost per prediction include data collection: {end_time - start_time}\n"
            # )

            if args.exit:
                break

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
    # print(
    #     f"avg inference time per prediction {infer_time / ((args.end_num_files - args.start_num_files + 1) * (args.passenger - args.passenger + 1))} ms, {infer_time / ((args.end_num_files - args.start_num_files + 1) * (args.passenger - args.passenger + 1)) * 1000 } s"
    # )

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
    # save_atten_npz(
    #     head_data_atten_npz,
    #     end_data_atten_npz,
    #     head_atten,
    #     end_atten,
    #     head_atten_label,
    #     end_atten_label,
    # )

    # pred_results = [res.get() for res in pred_results if res.get() is not None]
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
    threading.Thread(target=pause_input_handler, daemon=True).start()
    main_task()
