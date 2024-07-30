# MultiRxSensing
An implementation of CSI-based collaborative sensing with two receivers and several useful CSI data collection configuration shell scripts.

## Generating Ping Flow
Use generate_traffic.sh to generate ping flow from router

example - generate ping flow with 1000 Hz:
```
. generate_traffic.sh 1000 192.168.0.1
```

## Configure CSI Collection Parameters
Use csiparam_config.sh to configure CSI data collecion parameters

example - collect CSI data derived from Wi-Fi signal in 36 channel with 80 MHz bandwidth:
```
. csiparam_config.sh 36 80 
```

## Forward CSI data to another PC/Laptop
Use csi_forward.sh to forward CSI data to another PC/Laptop

example - forward CSI data from device with IP address 192.168.3.11 to device with IP address 192.168.3.12:
```
. csi_forward.sh 192.168.3.11 192.168.3.12
```

## Start CSI Data Collection and save them to a certain folder
Use 

