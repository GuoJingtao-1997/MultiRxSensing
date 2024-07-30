# MultiRxSensing
An implementation of CSI-based collaborative sensing with two receivers and several useful CSI data collection configuration shell scripts.

## Generate Ping Flow
Run generate_traffic.sh to generate ping flow from router

example - generate ping flow with 1000 Hz:
```
. generate_traffic.sh 1000 192.168.0.1
```

## Configure CSI Collection Parameters
Run csiparam_config.sh to configure CSI data collecion parameters

example - collect CSI data derived from Wi-Fi signal in 36 channel with 80 MHz bandwidth:
```
. csiparam_config.sh 36 80 
```

## Setup CSI data forwarding rule 
Run csi_forward.sh to setup CSI data forwarding rule

example - forward CSI data from device with IP address 192.168.3.11 to device with IP address 192.168.3.12:
```
sudo bash csi_forward.sh 192.168.3.11 192.168.3.12
```

## Start CSI Data Collection
Run packetcap_pc.sh to collect CSI data from a certain device and save them into a certain folder

example - collect CSI data under 0-1 people scenarios from a device with IP address 192.168.3.11 and save them into folder name "test". Each scenario will collect 500 pcap files with each of them include 1000 packets:
```
. packetcap_pc.sh test 0 1 1 500 1000 192.168.3.11
```

## Online Test
Run infer_model.py to start online test. Pytorch environment is needed. Note that you need to modify IP_ADDRESS and ACCESS_TOKEN parameters to your own. You may also customize NUM_FILES, and PASSENGER parameters to fit your own setting.

```
python infer_model.py
```


