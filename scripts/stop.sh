#!/bin/bash
echo "running stop.sh ..." >> /home/ubuntu/stop.log
pkill -f app.py || echo "no app.py process found" >> /home/ubuntu/stop.log
echo "stop.sh completed!!!" >> /home/ubuntu/stop.log
exit 0
