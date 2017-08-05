### Brief instructions to run the codes on the remote server

#### Tip: For detailed instructions go through the instructions provided by Colfax Research. Click on the connect tab for instructions on establishing a connection and file transfer. Click on the compute tab to know about job queuing.

After establishing connection with the server using  PuTTY (for windows) or using ssh (for Linux), transfer the required files.
The login node cannot be used for heavy computations. For running the codes, send the job to the compute node using the 'qsub' command using a launch script.

Also, in the script for launching the job, source torch into the path by adding the line: source /opt/torch/bin/torch-activate. Preferably, save the launch script in the home directory and add the path to your code in the script. 
For running a sample file named 'sample.lua' located in the folder 'myFolder', the sample launch script  would have the following lines:

    #PBS -N sampleJob
    cd  myFolder/
    source /opt/torch/bin/torch-activate
    th sample.lua
    
Note: sampleJob is the name assigned to the job.
You may add more lines for monitoring the start and end of code execution.

    #PBS -N sampleJob
    cd  myFolder/
    source /opt/torch/bin/torch-activate
    echo Start
    th sample.lua
    echo End

As instructed in the Colfax Research website, make sure that there is a blank line at the ebd of the script.

For monitoring the current status of the job, use the command 'qstat'

    $ qstat JOBID
    
where, JOBID is the job number assigend to the job when you start the job using 'qsub' command. The status 'R' incidates the the job is running in the compute node, 'E' indicates that these is an error and 'Q' indicates the job is in the queue. After the completion of the job, it will not appear in the list of current jobs you get by running the 'qstat' command. You may then look at the files sampleJob.eJOBID for the errors if any, and sampleJob.oJOBID for the output.

For deleting the job numbered JOBID which is running or is in the queue, used the command 'qdel',

    $ qdel JOBID
