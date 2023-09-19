#!/bin/bash

# Check if the script is running as root (superuser)
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)." 
   exit 1
fi

# Installation - Add if you need to download the dependencies
apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-rtsp

# Define the repository and username
repository="bluenviron/mediamtx"
latest_release=$(curl -s "https://api.github.com/repos/$repository/releases/latest" | grep -oP '"tag_name": "\K.*?(?=")')
file_name="mediamtx_${latest_release}_linux_amd64.tar.gz"
download_url="https://github.com/$repository/releases/download/$latest_release/$file_name"
echo "Downloading $download_url"

# Download the latest release
wget "$download_url"
tar -xvf "$file_name"
rm "$file_name"
