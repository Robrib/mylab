import librosa
import numpy as np
import os

matrix = []
for n in ['kicks', 'snares', 'hihats']:
    filelist = os.listdir('./' + n)
    for file in filelist:
        if file.startswith('.'):
            continue
        y, _ = librosa.core.load('./' + n + '/' + file, sr=44100, duration=0.3708)
        if len(y) < 16352:
            t = np.zeros(16352 - len(y))
            y = np.concatenate((y, t))
        S = librosa.core.stft(y, n_fft=1024, hop_length=32)
        matrix.append(np.array([np.abs(S)]))
    print('n' + 'finished')
        
matrix = np.array(matrix)
np.save('./training_data_512.npy', matrix)
