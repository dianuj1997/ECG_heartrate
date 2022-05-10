import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import time
from scipy.signal import butter, lfilter
import neurokit2 as nk
from ecgdetectors import Detectors
import pandas as pd
import heartpy as hp
from heartpy.datautils import rolling_mean
from scipy.interpolate import interp1d
import wfdb
from wfdb import processing
import scipy.signal as sig
import math
import scipy.signal as scs
import peakutils
from scipy.signal import find_peaks




ports=serial.tools.list_ports.comports()
serialInst=serial.Serial()

portVar="COM4"
serialInst.baudrate=9600 
serialInst.port=portVar
serialInst.open()

my_ecg_data=[]
init_t=0
my_time_data=[init_t]
numby=0

hr_1=[]
hr_2=[]
hr_3=[]
hr_4=[]
hr_5=[]
hr_6=[]
hr_7=[]
hr_8=[]
hr_9=[]
hr_10=[]
hr_11=[]
hr_12=[]
hr_13=[]
hr_14=[]
hr_15=[]

iter_loop=0
total_loops=4

while iter_loop<=total_loops:
    
    if serialInst.in_waiting:
        start_time = time.time()
        packet=serialInst.readline()
        dt=time.time() - start_time
        
        packet_reduced=packet.decode("utf-8")
        # print(packet_reduced)
        my_ecg_data.append(float(packet_reduced))
        newtime=my_time_data[-1]+dt
        my_time_data.append(newtime)
        numby+=1
        len_of_data=3000
        if len(my_ecg_data)==len_of_data:
            #------------Measure heart rate (Method #01)--------------------
                #  https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/53983/versions/5/previews/html/ExampleECGBitalino.html


            length=81
            order=5
            length2=101
            order2=5
            ecg=np.array(my_ecg_data,dtype=float)
            timer=np.array(my_time_data,dtype=float)
            timer=timer[0:len_of_data]
            detrended_ecg=signal.detrend(ecg)
            denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
            
            denoised_detrended_ecg2=savgol_filter(detrended_ecg,length2, order2)
            denoised_detrended_ecg_copy=np.copy(denoised_detrended_ecg2)

            maxIndices=argrelextrema(denoised_detrended_ecg, np.greater)
            maxIndices=np.squeeze(maxIndices,axis=0)
            msPerBeat = np.mean(np.diff(timer[maxIndices]))
            rheartRate =1/(msPerBeat)*2
            print("----Heart Rate (Method #01)----: ",rheartRate)
            if rheartRate!="nan" or rheartRate!="inf":
                hr_1.append(rheartRate)

            else:
                hr_1.append(0)

            # plt.plot(timer,ecg)
            # plt.plot(timer,denoised_detrended_ecg)
            # plt.plot(timer[maxIndices],denoised_detrended_ecg[maxIndices],'ro')
            # plt.show()   



            #------------Measure heart rate (Method #02)--------------------
            # https://github.com/neuropsychology/NeuroKit

            
            fs = 1/(np.mean(np.diff(timer)))
            info = nk.signal_findpeaks(denoised_detrended_ecg)
            ecg_rate = nk.signal_rate(peaks=info["Peaks"], desired_length=len(denoised_detrended_ecg)) 
            my_hrater=np.mean(ecg_rate)/100*4/5
            print("----Heart Rate (Method #02)----: ",my_hrater)
            if str(my_hrater)!="nan" or str(my_hrater)!="inf":
                hr_2.append(my_hrater)

            else:
                hr_2.append(0)




            #------------Measure heart rate (Method #03)--------------------

            # https://neurokit2.readthedocs.io/en/latest/functions.html
            # https://pypi.org/project/py-ecg-detectors/


            try:
                framized_ecg = pd.DataFrame(denoised_detrended_ecg, columns = ['ECG'])
                detectors = Detectors(fs)
                r_peaks_pan = detectors.pan_tompkins_detector(framized_ecg.iloc[:,0])
                r_peaks_pan= np.asarray(r_peaks_pan)
                msPerBeat_new = np.mean(np.diff(timer[r_peaks_pan]));
                rheartRate_new = 60*(1/msPerBeat_new)/1.5;
                print("----Heart Rate (Method #03)----: ",rheartRate_new)

                if rheartRate_new!="nan" or rheartRate_new!="inf":

                    hr_3.append(rheartRate_new)

                else:
                    hr_3.append(0)

            except:
                print("----Error in finding R-peaks for Heart Rate (Method #03)")

                hr_3.append(0)

            print("\n")




            #------------Measure heart rate (Method #04)--------------------
            # https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.analysis.html
            try:
            
                wd=hp.peakdetection.detect_peaks(ecg, detrended_ecg, ma_perc = 20, sample_rate = 100.0)
                peaklister=wd['peaklist']
                wd = hp.analysis.calc_rr(peaklister, sample_rate = fs)
                rr_list=wd['RR_list']
                rr_diff = np.diff(rr_list)
                rr_sqdiff = np.power(rr_diff, 2)
                wd, m = hp.analysis.calc_ts_measures(rr_list, rr_diff, rr_sqdiff)
                rheartRate_noob=m['bpm']
                print("----Heart Rate (Method #04)----: ",rheartRate_noob)
                print("\n")
                if str(rheartRate_noob)!="nan" or str(rheartRate_noob)!="inf":
                    hr_4.append(rheartRate_noob)
                else:
                    hr_4.append(0)

            except:
                print("----Error in finding R-peaks for Heart Rate (Method #04)")
                print("\n")
                hr_4.append(0)



            #------------Measure heart rate (Method #05)--------------------

            #https://github.com/maltintas45/Heart-Rate-from-ECG/blob/master/BPM.ipynb

            def calculateBPM(dfA, T=6):
            # as input, dfA is a dataframe with two column; ECG and time(seconds)
            # this function returns the BPMs for seconds
                BPMs=[]
                frequence=dfA.index[dfA['time'] == dfA['time'][0]+1].tolist()[0]-dfA.index[dfA['time'] == dfA['time'][0]].tolist()[0]
                for k in range((dfA.shape[0]/frequence)-T+1):
                    df=dfA[k*frequence:(k+T)*frequence]
                    sig  = df['ECG']
                    widths = np.arange(1, 2)
                    cwtmatr = signal.cwt(sig, signal.ricker, widths)
                    square_cwt_c0=[v**2 for v in cwtmatr[0].tolist()]
                    threshold=np.mean(square_cwt_c0)
                    marks=[]
                    for i,v in enumerate(square_cwt_c0):
                        if v>threshold:
                            marks.append((k*frequence)+i)
                    peaks=[]
                    beats=0
                    #print marks, df['ECG']
                    for i,m in enumerate(marks):
                        #print i
                        if i==0 or i==len(marks)-1:
                            continue
                        if df['ECG'][marks[i-1]]<df['ECG'][m] and df['ECG'][marks[i+1]]<df['ECG'][m]: 
                            peaks.append(m)
                            beats+=1
                    
                    #print 'for ',k,'. second;',str(beats)," beats in ",str(max(df['time'])-min(df['time'])), "seconds" 
                    BPMs.append(beats*(60/T))
                return BPMs
            try:
                df = pd.DataFrame({'time':timer})
                df['ECG'] = ecg
                BPMs = calculateBPM(df, T=20)
                print("----Heart Rate (Method #05)----: ",sum(BPMs)/len(BPMs))

                print("\n")
                if sum(BPMs)/len(BPMs)!="nan" or sum(BPMs)/len(BPMs)!="inf":
                    hr_5.append(sum(BPMs)/len(BPMs))
                else:
                    hr_5.append(0)
            
            except:
                print("----Heart Rate (Method #05) not possible----: ")
                print("\n")
                hr_5.append(0)




            #------------Measure heart rate (Method #06)--------------------
            #https://github.com/johnathanfernandes/ECG-heart-rate-calculator

            try:
            
                Mer=max(ecg);
                scale_factor=1.5/Mer;
                scaled_signal=ecg*scale_factor
                count=0
                for k in range(2,length(scaled_signal)-1):
                    if (scaled_signal[k]> 0.6 & scaled_signal[k]>scaled_signal[k-1] & scaled_signal[k]>scaled_signal[k+1]):
                        count=count+1


                Ner= length(scaled_signal)
                duration=Ner/(fs*60)
                BPMer=count/duration

                print("----Heart Rate (Method #06)----: ",sum(BPMer)/len(BPMer))
                print("\n")
                if sum(BPMer)/len(BPMer)!="nan" or sum(BPMer)/len(BPMer)!="inf":
                    hr_6.append(sum(BPMer)/len(BPMer))
                else:
                    hr_6.append(0)


            except:
                print("----Heart Rate (Method #06) not possible----: ")
                print("\n")
                hr_6.append(0)



            #------------Measure heart rate (Method #07)--------------------
            # https://github.com/MIT-LCP/wfdb-python

            try:

                # Use the gqrs algorithm to detect qrs locations in the first channel
                qrs_inds = processing.gqrs_detect(sig=ecg, fs=fs)

                # Correct the peaks shifting them to local maxima
                min_bpm = 20
                max_bpm = 230
                #min_gap = record.fs * 60 / min_bpm
                # Use the maximum possible bpm as the search radius
                search_radius = int(fs * 60 / max_bpm)
                corrected_peak_inds = processing.correct_peaks(ecg, peak_inds=qrs_inds,
                                                                search_radius=search_radius, smooth_window_size=150)


                hrs = processing.compute_hr(sig_len=ecg.shape[0], qrs_inds=sorted(corrected_peak_inds), fs=fs)

                #print(hrs.size)
                heart_rate_raw = np.array([x for x in hrs if str(x) != 'nan'])
                heart_rater=np.mean(heart_rate_raw)/2.3
                print("----Heart Rate (Method #07)----: ",heart_rater)

                print("\n")
                if str(heart_rater)/2.3!="nan" or str(heart_rater)!="inf":
                    hr_7.append(heart_rater)
                else:
                    hr_7.append(0)

            except:
                print("----Heart Rate (Method #07) not possible----: ")

                print("\n")
                hr_7.append(0)



            #------------Measure heart rate (Method #08)--------------------
            # https://github.com/pantos98/ECG_analysis_python/blob/main/hr_calc_method.py
            # https://github.com/kaseykwong/bme590hrm/blob/master/process_data.py

            try:

                """
                Filter the ecg with a sombrero low pass filter.
                """
                b1 = [-7.757327341237223e-5,  -2.357742589814283e-4, -6.689305101192819e-4, -0.001770119249103, \
                        -0.004364327211358, -0.010013251577232, -0.021344241245400, -0.042182820580118, -0.077080889653194, \
                        -0.129740392318591, -0.200064921294891, -0.280328573340852, -0.352139052257134, -0.386867664739069, \
                            -0.351974030208595, -0.223363323458050, 0, 0.286427448595213, 0.574058766243311, \
                        0.788100265785590, 0.867325070584078, 0.788100265785590, 0.574058766243311, 0.286427448595213, 0, \
                        -0.223363323458050, -0.351974030208595, -0.386867664739069, -0.352139052257134, \
                        -0.280328573340852, -0.200064921294891, -0.129740392318591, -0.077080889653194, -0.042182820580118, \
                        -0.021344241245400, -0.010013251577232, -0.004364327211358, -0.001770119249103, -6.689305101192819e-04, \
                        -2.357742589814283e-04, -7.757327341237223e-05]

                filt_ecg = sig.filtfilt(b1,1,ecg)

                voltage = ecg
        
                hrw = 0.5
                mov_avg = filt_ecg 
                avg_hr = (np.mean(voltage))
                if avg_hr < 0 and abs(np.min(filt_ecg)) - abs(avg_hr) > abs(np.max(filt_ecg)) - abs(avg_hr):
                    voltage = voltage + abs(avg_hr)+abs(np.min(filt_ecg))

                mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
                mov_avg = [(x+abs(avg_hr-abs(np.min(filt_ecg)/2)))*1.2 for x in mov_avg]
                window = []
                peaklist = []
                listpos = 0
                for datapoint in voltage:
                    rollingmean = mov_avg[listpos]
                    if (datapoint < rollingmean) and (len(window) < 1):
                        listpos += 1
                    elif (datapoint > rollingmean):
                        window.append(datapoint)
                        listpos += 1
                        if (listpos >= len(voltage)):
                            beatposition = listpos - len(window) + \
                                        (window.index(max(window)))
                            peaklist.append(beatposition)
                            window = []
                    else:
                        beatposition = listpos - len(window) + \
                                    (window.index(max(window)))
                        peaklist.append(beatposition)
                        window = []
                        listpos += 1


                RR_list = []
                cnt = 0
                while (cnt < (len(peaklist) - 1)):
                    RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
                    ms_dist = ((RR_interval / fs) * 1000.0)
                    RR_list.append(ms_dist)
                    cnt += 1

                

                hrm = 6000 / np.mean(RR_list)


                print("----Heart Rate (Method #08)----: ",hrm)

                print("\n")
                if str(hrm)!="nan" or str(hrm)!="inf":
                    hr_8.append(hrm)
                else:
                    hr_8.append(0)

            except:
                print("----Heart Rate (Method #08) not possible----: ")

                print("\n")
                hr_8.append(0)



            #------------Measure heart rate (Method #09)--------------------
            # https://github.com/ptuduj/ECG-signal-processing/blob/master/ECG_project.ipynb

            try:
                ecg_signal=ecg
                ecg_signal = ecg_signal-np.mean(ecg_signal)
                ecg_signal = ecg_signal / max(abs(ecg_signal)) # normalize to one
                samples_num=len(ecg_signal)
                dcblock = np.zeros(samples_num)
                for index in range(samples_num):
                    if index >= 1:
                        dcblock[index] = ecg_signal[index] - ecg_signal[index-1] + 0.995*dcblock[index-1]

                dcblock = dcblock/ max(abs(dcblock))  # normalize to one

                #low pass filter
                # y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T) 
                lowpass = np.zeros(samples_num)
                for index in range(samples_num):
                    if index >= 1:
                        lowpass[index] += 2*lowpass[index-1]
                    if index >= 2:
                        lowpass[index] -= lowpass[index-2]
                    lowpass[index] += dcblock[index]
                    if index >= 6:
                        lowpass[index] -= 2*dcblock[index-6]
                    if index >=12:
                        lowpass[index] += dcblock[index-12]

                lowpass = lowpass/ max(abs(lowpass))  # normalize to one

                #high pass
                # y(nT) = 32x(nT - 16T) - [y(nT - T) + x(nT) - x(nT - 32T)]
                highpass = np.zeros(samples_num)
                for index in range(samples_num):
                    if index >= 16:
                        highpass[index] += 32 * lowpass[index-16]
                    if index >= 1:
                        highpass[index] -= highpass[index-1]
                    highpass[index] = -lowpass[index]
                    if index >= 32:
                        highpass[index] += lowpass[index-32]

                highpass = highpass/ max(abs(highpass))       # normalize to one

                #derivative
                #y(nT) = 1/8 [−x(nT − 2T) − 2x(nT − T) + 2x(nT + T) + x(nT + 2T)]
                d_signal = np.zeros(samples_num)
                for index in range(samples_num):
                    if index >= 2:
                        d_signal[index] -= highpass[index-2]
                    if index >= 1:
                        d_signal[index] -= 2 * highpass[index-1]

                    if index <= samples_num - 1 - 1:
                        d_signal[index] += 2 * highpass[index+1]
                    if index <= samples_num - 2 - 1:
                        d_signal[index] += highpass[index+2]
                    d_signal[index] /= 8
                    
                d_signal = d_signal/ max(abs(d_signal))  # normalize to one


                # square signal
                s_signal = np.square(d_signal)
                s_signal = s_signal/ max(abs(s_signal))  # normalize to one

                #moving average
                #y(nT) = (1/N)[x(nT - (N - 1)T) + x(nT - (N - 2)T) + ... x(nT)]
                m_signal = (np.convolve(s_signal, np.ones((np.around(0.150 * fs).astype(int))) / np.around(0.15 * fs)))
                m_signal = m_signal/ max(abs(m_signal)); # normalize to one


                peak_locs = scs.find_peaks_cwt(m_signal, widths=np.arange(1, 50), min_length=np.around(0.1 * fs))
                peak_locs_fin = []
                for pl in peak_locs:
                    tmp_sig = m_signal[pl-50:min(pl + 50, len(m_signal))]
                    ind = np.argmax(tmp_sig)
                    peak_locs_fin.append(pl - 50 + ind)

                max_val = np.max(list(map(lambda x: m_signal[x],peak_locs_fin)))
                treshold =0.2* max_val
                peak_final_final = np.array(list(filter(lambda x: m_signal[x] > treshold , peak_locs_fin)))

                #calculate heart rate
                timing =(len(m_signal)-1)/fs # in sec
                r_pick_count = len(peak_final_final)
                hearthrate= r_pick_count/timing 
                hearthrate = hearthrate * 60/2

                print("----Heart Rate (Method #09)----: ",hearthrate)

                print("\n")
                if str(hearthrate)!="nan" or str(hearthrate)!="inf":
                    hr_9.append(hearthrate)

                else:
                    hr_9.append(0)

            except:
                print("----Heart Rate (Method #09) not possible----: ")

                print("\n")
                hr_9.append(0)






            #------------Measure heart rate (Method #10)--------------------

            # https://github.com/haerynkim/bme590hrm/tree/master/functions

            try:

                def find_peak(corr, thres=0.5, min_dist=0.1):
                    cb = np.array(corr)
                    beatidx = peakutils.indexes(cb, thres, min_dist)
                    num_beats = len(beatidx)
                    return beatidx, num_beats


                voltage = denoised_detrended_ecg
                peakidx, numpeak = find_peak(voltage, thres=0.5, min_dist=0.1)
                peaktimes = timer[peakidx]
                rr_int = [t - s for s, t in zip(peaktimes, peaktimes[1:])]
                rr_avg = sum(rr_int) / float(len(rr_int))
                mean_hr_bpm = (60 / rr_avg)/20*1.4/1.76


                print("----Heart Rate (Method #10)----: ",mean_hr_bpm)

                print("\n")
                if mean_hr_bpm!="nan" or mean_hr_bpm!="inf":
                    hr_10.append(mean_hr_bpm)
                else:
                    hr_10.append(0)

            except:
                print("----Heart Rate (Method #10) not possible----: ")
                print("\n")
                hr_10.append(0)



            #------------Measure heart rate (Method #11)--------------------

            # https://github.com/durmuselin/heartratedetectionbyusingmatlab/blob/main/ECGHEARTRATE.m

            try:
                length=81
                order=5
                length2=101
                order2=5
                ecg=np.array(my_ecg_data,dtype=float)
                timer=np.array(my_time_data,dtype=float)
                timer=timer[0:len_of_data]
                detrended_ecg=signal.detrend(ecg)
                denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
                peaks, _ = find_peaks(denoised_detrended_ecg, height=2*np.sqrt(np.mean(denoised_detrended_ecg**2)))
                beatspermin=len(peaks)/(timer[-1]-timer[0])*60
                print("----Heart Rate (Method #11)----: ",beatspermin)
                print("\n")
                if str(beatspermin)!="nan" or str(beatspermin)!="inf":
                    hr_11.append(beatspermin)
                else:
                    hr_11.append(0)

            except:
                print("----Heart Rate (Method #11) not possible----: ")
                print("\n")
                # hr_11.append(0)



             #------------Measure heart rate (Method #12)--------------------

            # https://github.com/mansibm6/heartrate-from-ecg-data/blob/main/heart_rate.m

            try:

                length=81
                order=5
                length2=101
                order2=5
                ecg=np.array(my_ecg_data,dtype=float)
                timer=np.array(my_time_data,dtype=float)
                timer=timer[0:len_of_data]
                detrended_ecg=signal.detrend(ecg)
                denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
                peaks, _ = find_peaks(denoised_detrended_ecg, height=2*np.sqrt(np.mean(denoised_detrended_ecg**2)))
                bmsPerBeat = np.mean(np.diff(timer[peaks]))
                bpmx = (60 / bmsPerBeat)/1.5



                print("----Heart Rate (Method #12)----: ",bpmx)

                print("\n")
                if str(bpmx)!="nan" or str(bpmx)!="inf":
                    hr_12.append(bpmx)
                else:
                    hr_12.append(0)

            except:
                print("----Heart Rate (Method #12) not possible----: ")
                print("\n")
                hr_12.append(0)



             #------------Measure heart rate (Method #13)--------------------

            # https://github.com/Ronald-cons/ECG-HeartRate-Detect/blob/master/hr_detect.py

            try:

                class FIR_filter:
                    def __init__(self, fir_input):
                        self.offset = 0
                        self.p = 0
                        self.coeff = 0
                        self.buffer = np.zeros(number_of_taps)
                        self.input = fir_input

                    def dofilter(self, v):
                        #lms update, get tap input power, buffer 
                        output = 0
                        self.buf_val = self.p + self.offset
                        self.buffer[self.buf_val] = v
                        while self.buf_val >= self.p:
                            output += (self.buffer[self.buf_val] * self.input[self.coeff])
                            self.buf_val = self.buf_val - 1
                            self.coeff = self.coeff + 1

                        self.buf_val = self.p + number_of_taps - 1

                        while self.coeff < number_of_taps:
                            output += (self.buffer[self.buf_val] * self.input[self.coeff])
                            self.buf_val = self.buf_val - 1
                            self.coeff = self.coeff + 1

                        self.offset = self.offset + 1

                        if self.offset >= number_of_taps:
                            self.offset = 0

                        self.coeff = 0
                        return output

                def filterShift(f0,f1,f2):
                    #filter inlcudes band stop (to remove 50hz) and high pass (to remove DC)
                    fir_f_resp = np.ones(number_of_taps)
                    #---------------band stop-------------------------------
                    fir_f_resp[int((f1 / fs) * number_of_taps):int((f2 / fs) * number_of_taps) + 1] = 0
                    fir_f_resp[number_of_taps - int((f2 / fs) * number_of_taps):number_of_taps - int((f1 / fs) * number_of_taps) + 1] = 0 #do mirror
                    #---------------low stop (high pass)-------------------------
                    fir_f_resp[0:int((f0 / fs) * number_of_taps) + 1] = 0
                    fir_f_resp[number_of_taps - int((f0 / fs) * number_of_taps):number_of_taps] = 0 #do mirror

                    fir_t = np.fft.ifft(fir_f_resp) #frequency response to time
                    h_real = np.real(fir_t) #time real ~ casual
                    h_shifted = np.zeros(number_of_taps)
                    h_shifted[0:int(number_of_taps / 2)] = h_real[int(number_of_taps / 2):number_of_taps]
                    h_shifted[int(number_of_taps / 2):number_of_taps] = h_real[0:int(number_of_taps / 2)]
                    return h_shifted

                class matched_filter(FIR_filter):
                    def __init__(self, ecg_input):
                        self.input = ecg_input
                    def Rpeak_detection(self, template, oringin):
                        fir_coeff = template[::-1]
                        detected_array = np.zeros(len(oringin))
                        fir_template = FIR_filter(fir_coeff)
                        for i in range(len(oringin)):
                            detected_array[i] = fir_template.dofilter(self.input[i])
                        detected_output = detected_array * detected_array  # The signal is squared to improve the output
                        return detected_output

                class generateTemplate:
                    def __init__(self):
                        self

                    def mexicanhat(self):
                        t = np.linspace(-250, 250, 500) #in the range of taps
                        temp = (2 / np.sqrt(3 * 35) * np.pi ** (1 / 4)) * \
                                    (1 - (t ** 2 / 35 ** 2)) * np.exp((-t ** 2) / (2 * 35 ** 2))
                        return temp

                    def gaussian1OD(self):
                        t = np.linspace(-250, 250, 500)
                        temp = -t * np.exp((-t ** 2) / 50) / (125 * np.sqrt(2 * np.pi))
                        return temp

                    def gaussian(self):
                        t = np.linspace(-250, 250, 500)
                        temp = np.exp((-t ** 2) / 50) / (5 * np.sqrt(2 * np.pi))
                        return temp

                    def shannon(self):
                        t = np.linspace(-250, 250, 500)
                        temp = np.sqrt(100) * np.sinc(100 * t) * np.exp(2 * 1j * t * np.pi * 4)
                        return temp

                class detectMomentaryHeartRate:
                    def __init__(self, inputlist):
                        self.inputlist = inputlist

                    def detectMomentaryHeartRate(self,Fs):
                        list = self.inputlist  # Output from Matched filter
                        BPM = []  # It will be the array of Peaks
                        counter = 0  # auxiliary counter
                        threshold = max(self.inputlist) * 0.05
                        for i in range(len(list)):
                            if list[i] > threshold:
                                differenceTime = (i - counter)  # difference of time T in second, f = 1/T
                                counter = i
                                bpm = 1 / differenceTime * (60 * Fs) #1min
                                if 200 > bpm > 40:  # Limits for the detection of momentary heart rate
                                    BPM.append(bpm)  # Add this peak to the BPM array
                        BPM = np.delete(BPM, 0) #remove
                        return BPM

                def do_detection(input, Fs, oringin):
                    template = generateTemplate()  # Create the class TemplateMaker

                    # generate the different templates
                    gaussian = template.gaussian()
                    devgaussian = template.gaussian1OD()
                    shannon = template.shannon()
                    mexicanHat = template.mexicanhat()

                    # Matching Filtering the R peak of the signal
                    calculHeartBeat = matched_filter(input)
                    detgaussian = calculHeartBeat.Rpeak_detection(gaussian, oringin)
                    det1ODgaussian = calculHeartBeat.Rpeak_detection(devgaussian, oringin)
                    detshannon = calculHeartBeat.Rpeak_detection(shannon, oringin)
                    detmexicanHat = calculHeartBeat.Rpeak_detection(mexicanHat, oringin)

                    #  calculate its coefficients of the matched filter analytically from a mathematical formula
                    momentary_heart_rate_gaussian = detectMomentaryHeartRate(detgaussian)
                    momentary_heart_rate_gaussian10d = detectMomentaryHeartRate(det1ODgaussian)
                    momentary_heart_rate_shannon = detectMomentaryHeartRate(detshannon)
                    momentary_heart_rate_mexicanhat = detectMomentaryHeartRate(detmexicanHat)

                    # get the result value
                    MHRGaussian = momentary_heart_rate_gaussian.detectMomentaryHeartRate(Fs)
                    MHRGaussian1OD = momentary_heart_rate_gaussian10d.detectMomentaryHeartRate(Fs)
                    MHRShannon = momentary_heart_rate_shannon.detectMomentaryHeartRate(Fs)
                    MHRMexicanHat = momentary_heart_rate_mexicanhat.detectMomentaryHeartRate(Fs)
                    return MHRShannon


                length=81
                order=5
                length2=101
                order2=5
                ecg=np.array(my_ecg_data,dtype=float)
                timer=np.array(my_time_data,dtype=float)
                timer=timer[0:len_of_data]
                detrended_ecg=signal.detrend(ecg)
                denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
                peaks, _ = find_peaks(denoised_detrended_ecg, height=2*np.sqrt(np.mean(denoised_detrended_ecg**2)))
                bmsPerBeat = np.mean(np.diff(timer[peaks]))
                bpmx = 60 / bmsPerBeat*2
                    # Define the frequencies coefficients for the FIR filter
                f0 = 1  #remove DC
                f1 = 45 #Fstop
                f2 = 55 #Fpass
                # Initialise the script
                number_of_taps = 500 #taps

                unfilteredSignal = detrended_ecg  

                Shift = filterShift(f0, f1, f2)
                a = FIR_filter(Shift)
                filteredSignal = np.zeros(len(unfilteredSignal))
                #do filter
                for i in range(len(unfilteredSignal)):
                    filteredSignal[i] = a.dofilter(unfilteredSignal[i])

                moment_heartrate = do_detection(filteredSignal, fs, unfilteredSignal)

                bpmxx=np.mean(moment_heartrate)/1.5



                print("----Heart Rate (Method #13)----: ",bpmxx)

                print("\n")
                
                if bpmxx!="nan" or bpmxx!="inf":
                    hr_13.append(bpmxx)
                else:
                    hr_13.append(0)

            except:
                print("----Heart Rate (Method #13) not possible----: ")
                print("\n")
                hr_13.append(0)


            
             #------------Measure heart rate (Method #14)---------------------------------------
             # https://dadorran.wordpress.com/2014/05/22/heartrate-bpm-example-matlab-code/
            try:
                beat_count = 0
                ecg=np.array(my_ecg_data,dtype=float)
                my_sig=ecg

                for k in range(1,len(my_sig)-1):
                    if my_sig[k] > my_sig[k-1] and my_sig[k]> my_sig[k+1] and my_sig[k]>1:
                        beat_count = beat_count + 1

                N = length(my_sig);
                duration_in_seconds = N/fs
                duration_in_minutes = duration_in_seconds/60
                BPM_avg = beat_count/duration_in_minutes


                print("----Heart Rate (Method #14)----: ",BPM_avg)
                print("\n")
                
                if BPM_avg!="nan" or BPM_avg!="inf":
                    hr_14.append(BPM_avg)
                else:
                    hr_14.append(0)

            except:
                print("----Heart Rate (Method #14) not possible----: ")
                print("\n")
                hr_14.append(0)



            #------------Measure heart rate (Method #15)---------------------------------------------
            # https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/53983/versions/5/previews/html/ExampleECGBitalino.html
            try:
                ecg=np.array(my_ecg_data,dtype=float)
                ecg=ecg/np.max(ecg)*2
                timer=np.array(my_time_data,dtype=float)
                timer=timer[0:len_of_data]
                detrended_ecg=signal.detrend(ecg)
                denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
                _, rlocs = find_peaks(denoised_detrended_ecg, distance=500,height=0.8)
                locs_Qwave = np.zeros(len(rlocs),1);
                locs_Swave = np.zeros(len(rlocs),1);
                locs_Qpre  = np.zeros(len(rlocs),1);
                locs_Spost = np.zeros(len(rlocs),1);
                QRs = np.zeros(len(rlocs),1);

                # Find Q and S waves in the signal
                for ii in range(0,len(rlocs)):
                    window = denoised_detrended_ecg[rlocs[ii]-80:(rlocs(ii)+80)]
                    d_peaks, locs_peaks = find_peaks(-window, distance=40)
                    i=np.argsort(np.array(d_peaks))
                    i=np.flip(i)
                    locs_Qwave[ii] = locs_peaks[i[1]]+(rlocs[ii]-80)
                    locs_Swave[ii] = locs_peaks[i[2]]+(rlocs[ii]-80)
                    d_QRS, locs_QRS= find_peaks(window, height=10)
                    max_i = np.argmax(d_QRS)
                    locs_Q_flat = locs_QRS[max_i-1]
                    locs_S_flat = locs_QRS[max_i+1]
                    locs_Qpre[ii]  = locs_Q_flat+(rlocs[ii]-80)
                    locs_Spost[ii] = locs_S_flat+(rlocs[ii]-80)
                    QRs[ii] = locs_S_flat - locs_Q_flat

                # Calculate the heart rate
                myqrs = np.median(QRs)
                myheartrate = 60/(np.median(np.diff(rlocs))/ 1000)

                print("----Heart Rate (Method #15)----: ",myheartrate)
                print("----QRS (ms) (Method #15)----: ",myqrs)
                print("\n")
                
                if myheartrate!="nan" or myheartrate!="inf":
                    hr_15.append(myheartrate)
                else:
                    hr_15.append(0)

            except:
                print("----Heart Rate (Method #15) not possible----: ")
                print("\n")
                hr_15.append(0)




            #--------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------
            #---------------------------------------------------------------------------------
            iter_loop+=1
            my_ecg_data=[]

# print(hr_11)
df = pd.DataFrame({'Heart Rate (#01)':hr_1})
df['Heart Rate (#02)'] = hr_2
df['Heart Rate (#03)'] = hr_3
df['Heart Rate (#04)'] = hr_4
df['Heart Rate (#05)'] = hr_5
df['Heart Rate (#06)'] = hr_6
df['Heart Rate (#07)'] = hr_7
df['Heart Rate (#08)'] = hr_8
df['Heart Rate (#09)'] = hr_9
df['Heart Rate (#10)'] = hr_10
df['Heart Rate (#11)'] = hr_11
df['Heart Rate (#12)'] = hr_12
df['Heart Rate (#13)'] = hr_13
df['Heart Rate (#14)'] = hr_14
df['Heart Rate (#15)'] = hr_15

df.to_csv('ECG_results.csv')

