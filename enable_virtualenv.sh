#Use "source job.sh" to run this bash script 
virtualenv --system-site-packages -p python3 ./venv
#!/bin/bash =>> source
source "./venv/bin/activate" #start virtual environment
#Install tensorflow
pip3 install --upgrade tensorflow
pip3 install --upgrade matplotlib
pip3 install tensorflow-cpu
pip3 install tensorflow-gpu
#script output.txt
