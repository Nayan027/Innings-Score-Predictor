#!/bin/bash

echo "running start.sh ...." >> /home/ubuntu/start.log

cd /home/ubuntu/flaskapp || {
    echo "failed to cd into /home/ubuntu/flaskapp" >> /home/ubuntu/start.log
    exit 1
}


nohup python3 app.py > app.log 2>&1 &

if [ $? -eq 0 ]; then
    echo "flask app started successfully" >> /home/ubuntu/start.log
else
    echo "failed to start app" >> /home/ubuntu/start.log
    exit 1
fi