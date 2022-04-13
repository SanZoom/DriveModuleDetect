#!/bin/bash

PRO_NAME_2=/home/rock/Python_env/mp_env/code/DriveModuleDetect-main/main.py

while true; do
	NUM_2=`ps aux | grep -w ${PRO_NAME_2} | grep -v grep | wc -l`

	
	if [ "${NUM_2}" -lt "1" ];then
		echo "${PRO_NAME_2} was killed"
		bash /home/rock/Start_Shell/drive_detect.sh
	fi

	sleep 5s
done

exit 0
