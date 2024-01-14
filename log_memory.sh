#!/bin/bash

output_file="memory_usage.log"

echo "Logging memory usage to $output_file..."

while true; do
    # Get memory usage and timestamp
    memory_usage=$(free -m | awk 'NR==2{printf "%sMB", $3}')
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    # Log memory usage to file
    echo "$timestamp $memory_usage" >> "$output_file"

    # Wait for 1 second
    sleep 1
done