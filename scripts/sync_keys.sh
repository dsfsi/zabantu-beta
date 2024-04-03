#!/bin/bash
set -e

# Script: sync_keys.sh
# Description: Synchronizes keys/secrets from a local machine to a remote server.
# Author: Ndamulelo Nemakhavhani

# Display logo
cat << "EOF"
  _____             _  __               _____ _           
 / ____|           | |/ /              / ____(_)          
| (___  _   _ _ __ | ' / ___ _   _ ___| (___  _ _ __  ___ 
 \___ \| | | | '_ \|  < / _ \ | | / __|\___  \| | '_ \/ __|
 ____) | |_| | | | | . \  __/ |_| \__ \___) | | | | \__ \
|_____/ \__, |_| |_|_|\_\___|\__, |___/_____/|_|_| |_|___/
         __/ |               __/ |                       
        |___/               |___/                        
EOF

usage() {
  echo "Usage: $0 -s <server_ip> -u <username> [-f <env_file>]"
  echo
  echo "  -s <server_ip>   IP address of the remote server"
  echo "  -u <username>    Username for the remote server"
  echo "  -f <env_file>    Path to the .env file (default: .env)"
  echo "  -p <port>        Port number for the remote server (default: 22)"
  echo
}

# Parse command-line arguments
while getopts ":s:u:f:p:" opt; do
  case $opt in
    s) server_ip="$OPTARG";;
    u) username="$OPTARG";;
    f) env_file="$OPTARG";;
    p) port="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage;;
  esac
done

# set defaults
port=${port:-22}
env_file=${env_file:-.env}
project_name="projects/zabantu-beta"

# Check if required arguments are provided
if [ -z "$server_ip" ] || [ -z "$username" ]; then
  echo "Server IP and username are required." >&2
  usage
fi

if [ ! -f "$env_file" ]; then
  echo "Error: .env file '$env_file' not found. Nothing to sync" >&2
  exit 1
else
  echo "Using .env file: $env_file"
  . "$env_file"
fi

echo "Synchronizing keys to remote server..."
# test if the server is reachable
if ! ssh -p "$port" -q "${username}@${server_ip}" exit; then
  echo "Error: Unable to connect to the remote server." >&2
  exit 1
else
  echo "Connection to remote server ok."
fi

# Synchronize keys to the remote server
target_file="/home/${username}/.${username}.secrets.sh"
scp -P "$port" "$env_file" "${username}@${server_ip}:$target_file"


add_secret_load_script_to_remote_bashrc() {
  local remote_bashrc_file="/home/${username}/.bashrc"
  local secret_line=". ${target_file}"

  if ! ssh -p "$port" "${username}@${server_ip}" "grep -q '$secret_line' '$remote_bashrc_file'"; then
    echo "Adding secret execution to remote .bashrc"
    ssh -p "$port" "${username}@${server_ip}" "echo '$secret_line' >> '$remote_bashrc_file'"
    echo "Secret execution added to remote .bashrc"
  else
    echo "Secret execution already exists in remote .bashrc"
  fi
}


setup_google_credentials() {
  if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
      echo "Copying GOOGLE_APPLICATION_CREDENTIALS to remote server..."
      local target_dir="/home/${username}/.myvault"

      ssh -p "$port" "${username}@${server_ip}" "mkdir -p $target_dir"
      scp -P "$port" "$GOOGLE_APPLICATION_CREDENTIALS" "${username}@${server_ip}:${target_dir}/"
      echo "GOOGLE_APPLICATION_CREDENTIALS copied to remote server."

    else
      echo "Error: GOOGLE_APPLICATION_CREDENTIALS file not found." >&2
    fi
  fi
}


copy_env_file_to_remote_server() {
  local project_dir="/home/${username}/${project_name}"
  local target_file="${project_dir}/.env"

  if ssh -p "$port" "${username}@${server_ip}" "[ -d $project_dir ]"; then
    echo "Copying .env file to remote server..."
    scp -P "$port" "$env_file" "${username}@${server_ip}:${target_file}"
    echo ".env file copied to remote server."
  fi
}


if [ $? -eq 0 ]; then
  echo "Key synchronization completed successfully."
  add_secret_load_script_to_remote_bashrc
  setup_google_credentials
  copy_env_file_to_remote_server
else
  echo "Error: Key synchronization failed. See logs for details" >&2
  exit 1
fi