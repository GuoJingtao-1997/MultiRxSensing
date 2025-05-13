import numpy as np
from utils import softmax
from nexcsi import decoder
import os
import subprocess
import multiprocessing as mp
import time
from collections import defaultdict
from scipy.signal import butter, filtfilt, savgol_filter
import pywt
from time import perf_counter

PCAP_PATH = os.path.join(os.getcwd(), "../ssd/csi_data")
VIEW_LIST = ["head", "end"]
NUM_FILES = 10
PACKETS = 1000
NULL_SC = True
PILOT_SC = False
CSI_DICT = defaultdict(list)


def dbinv(x):
    return np.power(10, x / 10)


def multi_view_fusion(view_dict: dict):

    "attention-based multi-view fusion"
    scale_factor = view_dict["head"].shape[0] ** -0.5

    env_atten = (
        np.einsum(
            "b i d, b j d -> b i j",
            view_dict["head"],
            view_dict["end"],
            optimize=True,
        )
        * scale_factor
    )

    nonenv_atten = softmax(1 - env_atten, axis=-1)

    head_atten = np.einsum(
        "b j i , b i d -> b j d",
        nonenv_atten,
        view_dict["head"],
        optimize=True,
    )

    end_atten = np.einsum(
        "b i j , b j d -> b i d",
        nonenv_atten,
        view_dict["end"],
        optimize=True,
    )

    view_dict["head_atten"].append(head_atten)
    view_dict["end_atten"].append(end_atten)

    return view_dict


def scale_csi_frame(csi, rss):
    subcarrier_count = csi.shape[0]

    rss_pwr = dbinv(rss)

    abs_csi = np.abs(csi)

    csi_mag = np.sum(abs_csi**2, axis=1)

    norm_csi_mag = csi_mag / subcarrier_count

    scale = rss_pwr / norm_csi_mag

    scale = scale[:, np.newaxis]

    return csi * np.sqrt(scale)


def sqtwolog(x: float) -> float:
    return np.sqrt(2 * np.log(len(x)))


def get_var(cD):
    sig = [abs(s) for s in cD[-1]]
    return np.median(sig) / 0.6745


def denoise(csi_matrix, level=None):
    denoised_csi = np.zeros(csi_matrix.shape)
    frame_count, subcarrier_count = denoised_csi.shape[:2]

    if level is None:
        level = pywt.dwt_max_level(frame_count, "sym3")

    for subcarrier_index in range(subcarrier_count):
        subcarrier_signal = csi_matrix[:, subcarrier_index]

        coefficients = pywt.wavedec(subcarrier_signal, "sym3", level=level)
        coefficients_sln = [coefficients[0]]

        # Adapted from https://github.com/matthewddavis/lombardi/blob/master/processing/NPS-1.3.2/WDen.py
        # Implementing sln scaling.
        rescaling = get_var(coefficients)

        for l in range(level):
            threshold = sqtwolog(coefficients[l + 1] / rescaling)
            threshold *= rescaling

            # Equivalent to wthresh with the "s"=soft setting.
            coefficients_sln.append(
                pywt.threshold(coefficients[l + 1], threshold, mode="soft")
            )

        reconstructed_signal = pywt.waverec(coefficients_sln, "sym3")
        if len(reconstructed_signal) == frame_count:
            denoised_csi[:, subcarrier_index] = reconstructed_signal
        else:
            denoised_csi[:, subcarrier_index] = reconstructed_signal[:-1]

    return denoised_csi


# Sample rate and desired cutoff frequency (in Hz)
fs = 1000.0  # Sample rate, in Hz
cutoff = 30.0  # Desired cutoff frequency of the filter, in Hz

# Normalize the frequency
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

order = 4  # Filter order
b, a = butter(order, normal_cutoff, btype="low", analog=False)


def realtime_csi_read(
    pcap_idx: int,
    passenger: int,
    view: str,
    date: str,
    status: str,
    src_addr_dict: dict,
):
    # print(
    #     f"Collecting CSI from {src_addr_dict[view]}, view {view}, pid is {os.getpid()}, time is {time.ctime()}"
    # )
    passenger_folder = os.path.join(PCAP_PATH, f"{date}_{view}_{status}/{passenger}")
    if not os.path.exists(passenger_folder):
        os.makedirs(passenger_folder)

    pcap_file = os.path.join(passenger_folder, f"{pcap_idx}.pcap")
    start_time = perf_counter()
    subprocess.check_output(
        f"sudo tcpdump -i eth0 dst port 5500 -vv -w {pcap_file} -c {PACKETS} and src {src_addr_dict[view]} 2> /dev/null",
        shell=True,
    )
    end_time = perf_counter()
    print(f"time cost {end_time - start_time:2f} s")
    device = "raspberrypi"
    sample = decoder(device).read_pcap(pcap_file)
    rssi = sample["rssi"]
    csi = decoder(device).unpack(sample["csi"])
    if NULL_SC:
        csi = np.delete(csi, csi.dtype.metadata["nulls"], axis=1)
    if PILOT_SC:
        csi = np.delete(csi, csi.dtype.metadata["pilots"], axis=1)
    csi = filtfilt(b, a, csi, axis=0)
    # csi = np.abs(scale_csi_frame(csi, rssi))
    # csi = denoise(csi)
    csi = savgol_filter(np.abs(csi), 500, 5, axis=0)
    # print(
    #     f"view {view}, file {pcap_idx}.pcap, time taken: {end_time - start_time}, csi shape {csi.shape}, rssi shape {rssi.shape}"
    # )
    return csi, rssi


if __name__ == "__main__":
    print(f"Total CPU Processes: {mp.cpu_count()}")
    p = mp.Pool(2)
    start_time = time.time()
    for i in range(NUM_FILES):
        result_list = {}
        sub_start_time = time.time()
        for view in VIEW_LIST:
            result_list[view] = p.apply_async(realtime_csi_read, args=(i, view))
        CSI_DICT["head"].append(result_list["head"].get())
        CSI_DICT["end"].append(result_list["end"].get())
        sub_end_time = time.time()
        for head_csi, end_csi in zip(CSI_DICT["head"], CSI_DICT["end"]):
            print("head csi shape", head_csi.shape)
            print("end csi shape", end_csi.shape)
        print(
            f"time cost {sub_end_time - sub_start_time}, main process, pid {os.getpid()}, Waiting for all processes to finish"
        )
    p.close()
    p.join()
    end_time = time.time()
    print(f"time cost {end_time - start_time}")
