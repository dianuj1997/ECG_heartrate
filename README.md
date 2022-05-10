# ECG_heartrate
This repository contain a collection of 15 algorithms for measuring heart rate from ECG signals. 

First the ECG signals are measured from "ecg_measure.ino" using the apparatus described below:

![image](https://user-images.githubusercontent.com/47445756/167529957-fae132fa-aa31-49b2-880f-85b1be81662e.png)


Then the python file reads the serial data and applies the mentioned 15 algorithms over some time (defined by number of loops and samples gathered per loop) and stores the resultant in csv file
