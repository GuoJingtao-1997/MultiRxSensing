#!/bin/sh
###
 # @Author: Guo Jingtao
 # @Date: 2022-04-02 18:53:40
 # @LastEditTime: 2024-07-30 10:48:24
 # @LastEditors: Guo Jingtao
 # 
### 

#generate makecsiparams
if [ $# -eq 3 ]; then
    csiparam=$(makecsiparams -c $1/$2 -C 1 -N 1 -m $3)
elif [ $# -eq 4 ]; then
    csiparam=$(makecsiparams -c $1/$2 -C 1 -N 1 -m $3 -b $4)
else
    csiparam=$(makecsiparams -c $1/$2 -C 1 -N 1)
fi
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -s500 -b -l34 -v$csiparam
sudo iw phy `iw dev wlan0 info | gawk '/wiphy/ {printf "phy" $2}'` interface add mon0 type monitor
sudo ifconfig mon0 up

