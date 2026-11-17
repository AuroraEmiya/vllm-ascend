#!/bin/bash
HOST_IP=141.1.186.255
export http_proxy=http://$HOST_IP:3128
export https_proxy=http://$HOST_IP:3128
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
