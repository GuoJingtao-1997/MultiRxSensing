import numpy as np
import torch
import torchvision.transforms as transforms
from mobileones import mobileone, reparameterize_model
import os
import multiprocessing as mp
from nexcsi import decoder
import subprocess
import time
import paho.mqtt.client as paho
from collections import defaultdict
from scipy import signal

PASSENGER = 0
PCAP_PATH = os.getcwd()
VIEW_LIST = ["head", "end"]
SRC_ADDR_DICT = {"head": "IP_ADDRESS", "end": "IP_ADDRESS"}
NUM_FILES = 400
PACKETS = 300
NULL_SC = True
PILOT_SC = False

ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
broker = "demo.thingsboard.io"  # host name
port = 1883  # data listening port


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


def model_load(checkpoint_path: str, device):
    model = mobileone(num_classes=21, variant="s4", inference_mode=False).to(device)
    checkpoint = torch.load(
        os.path.join(
            os.getcwd(),
            f"613_{checkpoint_path}_mos4.pth",
        ),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint)
    # weight_norm_layer = list(model.children())[-1]
    model = reparameterize_model(model)
    model.eval()
    return model

def dbinv(x):
    return np.power(10, x / 10)

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

def scale_csi_frame(csi, rss):
    subcarrier_count = csi.shape[0]

    rss_pwr = dbinv(rss)

    abs_csi = np.abs(csi)

    csi_mag = np.sum(abs_csi**2, axis=1)

    norm_csi_mag = csi_mag / subcarrier_count

    scale = rss_pwr / norm_csi_mag

    scale = scale[:, np.newaxis]

    return csi * np.sqrt(scale)

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

def realtime_csi_read(pcap_idx: int, view: str):
    print(
        f"Collecting CSI from {SRC_ADDR_DICT[view]}, pid is {os.getpid()}, time is {time.ctime()}"
    )
    start_time = time.time()

    pcap_file = f"{pcap_idx}.pcap"

    subprocess.check_output(
        f"sudo tcpdump -i eth0 dst port 5500 -vv -w {pcap_file} -c {PACKETS} and src {SRC_ADDR_DICT[view]}",
        shell=True,
    )
    device = "raspberrypi"
    sample = decoder(device).read_pcap(pcap_file)
    rssi = sample["rssi"]
    csi = decoder(device).unpack(sample["csi"])
    if NULL_SC:
        csi = np.delete(csi, csi.dtype.metadata["nulls"], axis=1)
    if PILOT_SC:
        csi = np.delete(csi, csi.dtype.metadata["pilots"], axis=1)
    csi_deagc = np.abs(scale_csi_frame(csi, rssi))
    csi_deagc = signal.savgol_filter(csi_deagc, 5, 2, axis=0)
    end_time = time.time()
    print(f"view {view}, file {pcap_idx}.pcap, time taken: {end_time - start_time}")
    return csi_deagc

if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("setup torch device")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("load model")
    head_model = model_load("head", device)
    end_model = model_load("end", device)

    print("start loading data")
    test_transform = transforms.Compose(
        [transforms.Normalize(mean=[0.0005], std=[0.0002])]  # atten 613, 421
    )
    person_classes = [str(i) for i in range(21)]
    print("start testing")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    print(f"Total CPU Processes: {mp.cpu_count()}")
    start_time = time.time()
    y_prob = []
    for i in range(0, NUM_FILES):
        p = mp.Pool(2)
        CSI_DICT = defaultdict(list)
        result_list = {}
        sub_start_time = time.time()
        for view in VIEW_LIST:
            result_list[view] = p.apply_async(
                realtime_csi_read, args=(i, view)
            )
        CSI_DICT["head"].append(result_list["head"].get()[np.newaxis, np.newaxis, :])
        CSI_DICT["end"].append(result_list["end"].get()[np.newaxis, np.newaxis, :])

        sub_end_time = time.time()
        for head_csi, end_csi in zip(CSI_DICT["head"], CSI_DICT["end"]):
            print(f"head csi shape {CSI_DICT["head"][-1].shape}\n")
            print(f"end csi shape {CSI_DICT["end"][-1].shape}\n")

        print(
            f"time cost {sub_end_time - sub_start_time}, main process, pid {os.getpid()}, Waiting for all processes to finish\n"
        )

        y_prob.append(collaborate_real_test(
            head_model,
            end_model,
            CSI_DICT["head"][-1],
            CSI_DICT["end"][-1],
            device,
        ))

        payload = "{"
        payload += f'"People":{y_prob[-1]}'
        payload += "}\n"
        ret = test.publish("v1/devices/me/telemetry", payload)
        # topic-v1/devices/me/telemetry
        print("Please check LATEST TELEMETRY field of the device\n")
        print(payload)
        print(f"There are {person_classes[y_prob[-1]]} passengers\n")
        print("Finish model inference")

        p.close()
        p.join()

        end_time = time.time()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        infer_time = start.elapsed_time(end)

        # print(f"predicted results {}\n")
        print(f"time cost {end_time - start_time}\n")
        print(f"total inference time {infer_time} ms, {infer_time / 1000} s\n")

    print(f"y_prob {y_prob}")
    print(f"y_true {[PASSENGER] * NUM_FILES}")
