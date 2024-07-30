#!/bin/bash
###
 # @Author: Guo Jingtao
 # @Date: 2022-04-02 18:53:40
 # @LastEditTime: 2023-01-16 20:38:22
 # @LastEditors: Guo Jingtao
### 

if [ ! -d "$1" ];then
	mkdir $1
	else
	echo "folder $1 is existed"
fi

cd $1

for i in $(seq $2 $3)
do
	if [ ! -d "$i" ];then
		mkdir $i
		else
		echo "folder $i is existed"
	fi

	for j in $(seq $4 $5)
	do	# use -Z root to address the "tcpdump: Couldn't change ownership of savefile" problem
		echo $(sudo tcpdump -i eth0 dst port 5500 -vv -w $i/$j.pcap -c $6 and src $7);
	done
	read -n 1 -p "Do you want to continue [Y/N]? " answer
	printf "\n"  #wrap text
	case $answer in
		Y | y)	echo "fine, continue on..."
				continue;;
		N | n | *)	echo "OK, goodbye"
					break 2;;
	esac
	echo "finish capture in $i passenger(s) scenario"
done
cd ..

