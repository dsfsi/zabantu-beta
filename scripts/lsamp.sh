#!/bin/bash

# ╭━━━╮╱╱╱╱╱╱╱╱╱╱╱╱╭━━━╮
# ┃╭━╮┃╱╱╱╱╱╱╱╱╱╱╱╱┃╭━╮┃
# ┃╰━╯┣━━┳━┳━━┳━━┳━╋╯╭╯┃
# ┃╭━━┫╭╮┃╭┫╭╮┃┃━┫╭╋╮╰╮┃
# ┃┃╱╱┃╰╯┃┃┃╰╯┃┃━┫┃┃╰━╯┃
# ╰╯╱╱╰━━┻╯╰━╮┣━━┻╯╰━━━╯
# ╱╱╱╱╱╱╱╱╱╭━╯┃
# ╱╱╱╱╱╱╱╱╱╰━━╯

# Random Line Sampler
# Utility to sample a specified number of random lines from a given .txt file and saves them to a new directory.
# Example usage: ./lsamp.sh --input data.txt --output samples/newdir/ --lines 1000
# Author: Ndamulelo Nemakhavhani

function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --input <path>        Path to the input .txt file"
    echo "  --output <path>       Path to the output directory"
    echo "  --lines <num>         Number of random lines to sample (default: 10000)"
    echo "  --help                Display this help message"
    echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --input)
            input_file="$2"
            shift 2
            ;;
        --output)
            output_dir="$2"
            shift 2
            ;;
        --lines)
            num_lines="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set default values if not provided
num_lines=${num_lines:-10000}

# Check if the input file is provided
if [ -z "$input_file" ]; then
    echo "Error: Input file not provided."
    usage
    exit 1
fi

# Check if the output directory is provided
if [ -z "$output_dir" ]; then
    echo "Error: Output directory not provided."
    usage
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Get the input file name without the extension
input_file_name=$(basename "$input_file" .txt)

# Sample random lines and save to the output file
output_file="$output_dir/sample.${input_file_name}.txt"
shuf -n "$num_lines" "$input_file" > "$output_file"

echo "Sampled $num_lines random lines from $input_file and saved to $output_file"