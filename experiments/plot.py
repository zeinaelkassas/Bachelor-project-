import librosa
import numpy as np
import matplotlib.pyplot as plt
path="C:/Users/nice/shared/DCASE2021/task2/dev_data/fan/train/section_00_source_train_normal_0000_strength_1_ambient.wav"
y,sr= librosa.load(path,sr=None,mono=True)
#y = librosa.effects.pitch_shift(y,sr=16000,n_steps=3.5)
#y=librosa.effects.preemphasis(y, coef=0.97, return_zf=False)
#y=librosa.effects.time_stretch(y,rate=2)
# plot time domainnnnnnnnnnnnnnnnnnnn
# y=y[12000:16000]
# duration = len(y)/sr
# time = np.arange(0,duration,1/sr)
# plt.plot(time,y)
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Fan after time stretching.wav')
# plt.show()

#plot frequency domainnnnnnnnnnnnnnnn
# X = np.fft.fft(y)
# X_mag = np.absolute(X)
# f = np.linspace(0, sr, len(X_mag))
# left_mag = X_mag[:int(len(X_mag)/2)]
# left_freq = f[:int(len(f)/2)]
# plt.plot(left_freq, left_mag)
# plt.title("Discrete-Fourier Transform", fontdict=dict(size=15))
# plt.xlabel("Frequency", fontdict=dict(size=12))
# plt.ylabel("Magnitude", fontdict=dict(size=12))
# plt.show()