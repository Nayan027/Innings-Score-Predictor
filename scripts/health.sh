#!/bin/bash

echo "running health_check.sh ..." >> /home/ubuntu/health_check.log

# Make a curl request to localhost (adjust port or path if needed)
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Health check passed with HTTP 200" >> /home/ubuntu/health_check.log
    exit 0
else
    echo "Health check failed. HTTP status: $HTTP_STATUS" >> /home/ubuntu/health_check.log
    exit 1
fi
