from keras.models import load_model
import numpy as np
import glob
import librosa
import librosa.feature

model = load_model('dataset_lagu.h5')
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

def set_features_and_labels(file):
    all_features = []
    sound_files = glob.glob(file)
    for f in sound_files:
        features = extract_features_song(f)
        all_features.append(features)

    return np.stack(all_features)

feature = set_features_and_labels('Wali-Orang_Bilang_62.wav')
model.predict(feature)
model.predict_classes(feature)