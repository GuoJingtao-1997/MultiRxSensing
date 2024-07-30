# Start CSI Forwarding with NF tables
###
 # @Author: Guo Jingtao
 # @Date: 2022-04-02 18:53:40
 # @LastEditTime: 2024-07-30 10:47:15
 # @LastEditors: Guo Jingtao
 # @Description: 
 # @FilePath: /Users/guojingtao/Documents/pi_config_files/csi_forward.sh
 # 
### 

nft add table ip nexmon

nft 'add chain ip nexmon input  { type filter hook input  priority -150; policy accept; }'
nft 'add chain ip nexmon output { type filter hook output priority  150; policy accept; }'

nft add rule ip nexmon input  iifname "wlan0" ip protocol udp ip saddr 10.10.10.10 ip daddr 255.255.255.255 udp sport 5500 udp dport 5500 counter mark set 900 dup to $2 device "eth0"
nft add rule ip nexmon output oifname "eth0"  meta mark 900 counter ip saddr set $1 ip daddr set $2

echo "CSI Forwarding rules added"
