#!/bin/sh
###
 # @Author: Guo Jingtao
 # @Date: 2022-04-03 15:09:07
 # @LastEditTime: 2022-12-18 18:21:19
 # @LastEditors: Guo Jingtao
### 

Fs="$(echo "1 / $1" | bc -l)"

if [ -z "$Fs" ]
then
	Fs=0.01
fi

sudo ping -I wlan0 -f -i "$Fs" $2
