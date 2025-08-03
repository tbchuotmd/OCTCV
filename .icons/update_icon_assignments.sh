#!/bin/bash

read -p "Note: please deactivate any conda environments (including base) prior to running this script.  Hit enter if already done, or otherwise Ctrl + C to exit the script."

for dp in $(ls -d ../p?_*/); do

	dirNAME=$(basename $dp)
	pnum=$(echo $dirNAME | cut -d_ -f1)
	
	for i in *.png; do
		
		inum=$(echo $i | cut -d_ -f1)
		if [ $pnum == $inum ]; then
			
			gio set $HOME/PROJECTS/OCTCV/$dirNAME metadata::custom-icon file://$HOME/PROJECTS/OCTCV/.icons/$i
			echo "Assigned $i to $dp"
			
		fi
	
	done
	
done
