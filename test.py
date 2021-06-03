from tensorflow.python.keras.models import load_model
import numpy as np
import glob
import librosa
import librosa.feature

model = load_model('dataset_lagu.h5')
def extract_features_song(f):
    y, _ = librosa.load(f)
    mfcc = librosa.feature.mfcc(y)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

def set_features_and_labels(file):
    all_features = []
    sound_files = glob.glob(file)
    for f in sound_files:
        features = extract_features_song(f)
        all_features.append(features)

    return np.stack(all_features)

def predict(filename):    
    feature = set_features_and_labels('wali.wav')
    pred = model.predict(feature)
    result = np.where(pred[0] == np.amax(pred[0]))

    genres = ['wali','betharia_sonata', 'egoist', 'gemie', 'kobayashi', 'mica', 'mizuki', 'mnroid', 'sora', 'tk']
    return genres[result[0][0]]
