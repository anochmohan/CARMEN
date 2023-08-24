#!/bin/bash

for file in /home/ubs6/projects/carmen-project/09232022_2/1691383040.csv 
do
    # Storing fcount and csv file name into variables 
    fcount=$(basename $file .csv)
    in_csv=$(basename $file)
    output_log=CARMEN_logs/$fcount'_log.txt'
    error_log=CARMEN_logs/$fcount'_errorlog.txt'
    
    #Setting up log files
    exec 3>&2 1>$output_log 2>$error_log
    date +"Script executed on - %a %b %e %H:%M:%S %Z %Y"
    trap "echo 'ERROR: An error occured during execution, check log $error_log for details.' >&3" ERR


    # Calling the script
    python 192_96_python_FAM.py --fcount $fcount --chip-dim 192 --in-csv $in_csv \
    --in-layout 192_assignment.xlsx --timepoints 13 --out-dir output


    #Conditional to send email based on log file content
    if [[ -z $(grep '[^[:space:]]' $error_log) ]]; then
        mv $file /home/ubs6/projects/carmen-project/09232022_2/completed_runs/
        #echo 'Should email the html file and send the input files to complete dir' >&3
        echo "$file has been moved to /09232022_2/completed_runs" >&1
    else
        mv $file /home/ubs6/projects/carmen-project/09232022_2/incomplete_runs/
        echo "$file has been moved to /09232022_2/incomplete_runs" >&1
        #mail -a error_log -s "Error on run: $in_csv" ubs6@cdc.gov
    fi

done