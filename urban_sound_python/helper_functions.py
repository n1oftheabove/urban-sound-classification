
import os
import pandas as pd

def get_colorpalette(filepath):
    """
    returns a dictionary with the unique class names together
    with their associated colors. To be used for e.g. visualizations.
    filepath: path to meta data file within the repo, e.g.
    './data/UrbanSound8K/metadata/UrbanSound8K.csv'
    """

    df_meta = pd.read_csv(filepath)
    return {cat:'color' for cat in df_meta['class'].unique()}

def get_filepaths_from_dir(directory):
    '''
    returns a list of all file paths stored in a directory
    '''
    files_path = [os.path.join(directory,x) for x in os.listdir(directory)]
    return files_path

def get_filenames_from_dir(directory):
    '''
    returns a list of all file names stored in a directory
    '''
    file_names = [x for x in os.listdir(directory)]
    return file_names

def get_class_from_meta(filepath, meta_df):
    """
    queries the dataframe of the meta csv for a filename, returns the class
    filepath: relative filepath to sound file
    """
    return meta_df[meta_df['slice_file_name']==os.path.basename(filepath)]['class'].values[0]
