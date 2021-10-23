from urban_sound_python.librosa_feature_generation import *

# provide here which librosa features to calculate & their statistical metrics
librosa_config = {'chroma': {'func':calc_chroma,
                          'measures': [np.std, np.mean]
                         },
                  'mfcc': {'func':calc_mfcc,
                       'measures': [np.std, np.mean]
                         },
                  'zcr': {'func':calc_zcr,
                      'measures': [np.std, np.mean]
                         },
                  'sc': {'func':calc_sc,
                     'measures': [np.std, np.mean]
                         },
                  'sr': {'func':calc_sr,
                     'measures': [np.std, np.mean]
                         },
                  'sb': {'func':calc_sb,
                     'measures': [np.std, np.mean]
                         },
                  'rms': {'func':calc_rms,
                      'measures': [np.std, np.mean]
                         },
                 }

def get_column_headers_from_config(config_file):
    """
    Function to generate header names from config file
    """
    headers = []
    for k,v in config_file.items():
        for measures in v['measures']:
            headers.append(str(k) + '_' + str(measures.__name__))
    return headers
