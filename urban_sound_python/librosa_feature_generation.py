import librosa
import numpy as np
from tqdm.auto import tqdm
from urban_sound_python.librosa_feature_config import get_column_headers_from_config

def calc_chroma(y, sr):
    '''
    Returns a 2d numpy.ndarray with 12 chroma features (chromagram)
    from a librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.chroma_stft(y=y, sr=sr)

def calc_mfcc(y, sr):
    '''
    Returns numpy.ndarray with mfcc from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.mfcc(y=y, sr=sr)

def calc_zcr(y, sr):
    '''
    Returns numpy.ndarray with zcr from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.zero_crossing_rate(y=y)


def calc_sc(y, sr):
    '''
    Returns numpy.ndarray with sc from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.spectral_centroid(y=y, sr=sr)

def calc_sr(y, sr):
    '''
    Returns numpy.ndarray with sr from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.spectral_rolloff(y=y, sr=sr)

def calc_sb(y, sr):
    '''
    Returns numpy.ndarray with spectral bandwith from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)

def calc_rms(y, sr):
    '''
    Returns numpy.ndarray with rms from a
    librosa audio time series and a sampling rate

        Parameters:
            y (numpy.ndarray): librosa audio time series
            sr (int): sampling rate
    '''
    return librosa.feature.rms(y=y)

# function that creates a numpy array of the features of one input
# sound. The features are calculated according to the config file
def feats_from_sound(filepath, config_file):
    """
    calculates librosa features specified in config_file for a song
    provided via filepath
    """

    y, sr =librosa.core.load(filepath)

    feature_lst = []

    # loop through config file entries
    for k, v in config_file.items():
        feature = v['func'](y, sr)
        for measure in v['measures']:
            feature_lst.append(measure(feature))

    return feature_lst

def feats_from_sounds(audio_paths_list, config_file, disable_progress=True,
                      return_filenames=False):
    """
    calculates librosa features specified in config_file for all
    songs provided in the audio_paths_list
    """

    result_arr = []
    filenames = []
    for path in tqdm(audio_paths_list, disable=disable_progress):
        result_arr.append(feats_from_sound(path, config_file))
        filenames.append(path.split('/')[-1])
    if return_filenames:
        return np.vstack(result_arr), filenames
    else:
        return np.vstack(result_arr)

def generate_original_dataframe(config_file,
                                all_audio_path_list,
                                disable_progress=True):
    """
    Returns a dataframe of generated audio features specified in the config_file
    using the sound file database provided with the all_audio_path_list
    """

    feature_data, _filename_lst = feats_from_sounds(all_audio_path_list,
                                                   config_file,
                                                   return_filenames=True,
                                                   disable_progress=disable_progress)
    column_headers = get_column_headers_from_config(config_file)
    return pd.DataFrame(data=feature_data, columns=column_headers)
