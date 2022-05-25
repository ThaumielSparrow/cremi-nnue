from types import ModuleType

import numpy as np
import h5py
import cv2 as cv
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
from tqdm import tqdm
import math
import random


class CustomErrorTypes(Exception):
    # Class that lets me tell you that you did something wrong
    pass

class CREMI:
    """
    Class containing methods to manipulate and process CREMI data. Structure is not extendable beyond datasets at https://cremi.org/data/.

    To initialize, pass a list containing filenames referring to .hdf files.
    """
    def __init__(self, samplefolder, savefolder=None, autocon=False):
        self.samplefolder = samplefolder
        self.savefolder = savefolder
        self.autocon = autocon
    
    def preprocess(self):
        """
        Preprocesses superset CREMI data. Collates all datasets passed together.

        Null function. Creates 2 folders, one containing raw 2D EM data, and the other with processed segmentation.
        """
        true_iter = 0
        if not self.savefolder:
            if not os.path.exists('EM'):
                os.makedirs('EM')
                
            else:
                if not self.autocon:
                    yn_input = input('EM output folder already exists. OK to overwrite its contents? [y/n] ').lower()
                    if yn_input == 'n':
                        sys.exit()
                    else:
                        [f.unlink() for f in Path('EM').glob('*') if f.is_file()] 
                else:
                    [f.unlink() for f in Path('EM').glob('*') if f.is_file()]

            if not os.path.exists('SEG'):
                os.makedirs('SEG')
            else: 
                if not self.autocon:
                    yn_input = input('SEG output folder already exists. OK to overwrite its contents? [y/n] ').lower()
                    if yn_input == 'n':
                        sys.exit()
                    else:
                        [f.unlink() for f in Path('SEG').glob('*') if f.is_file()] 
                else:
                    [f.unlink() for f in Path('EM').glob('*') if f.is_file()]

        else:
            out_em = self.savefolder + '/EM'
            out_seg = self.savefolder + '/SEG'
            Path(out_em).mkdir(exist_ok=True, parents=True)
            Path(out_seg).mkdir(exist_ok=True, parents=True)

        for superset_data in Path(self.samplefolder).rglob('*.hdf'):
            with h5py.File(superset_data, 'r') as dataset:
                z, x, y = dataset['volumes/labels/neuron_ids'].shape

                for Z in tqdm(range(z)):
                    image = dataset['volumes/raw'][Z,:,:]
                    seg = dataset['volumes/labels/neuron_ids'][Z,:,:]
                    if self.savefolder:
                        em_savepath = out_em + f'/EM_{true_iter}.png'
                    else:
                        em_savepath = f'EM/EM_{true_iter}.png'
                    cv.imwrite(em_savepath, image)

                    cnts, heir = cv.findContours(seg, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
                    output = np.zeros([x, y], np.uint8)
                    cv.drawContours(output, cnts, -1, (255,255,255), 3)
                    if self.savefolder:
                        seg_savepath = out_seg + f'/SEG_{true_iter}.png'
                    else:
                        seg_savepath = f'SEG/SEG_{true_iter}.png'
                    cv.imwrite(seg_savepath, output)
                    true_iter += 1
    
    def test_train_split(self, train_folder, test_folder, train_volume=0.5):
        """
        Takes half of training dataset and uses it as evaluation metric. Percentage can be customized by passing
        a frequency to represent the percentage of data that stays as training.
        """
        em_train_folder = train_folder + '/EM'
        seg_train_folder = train_folder + '/SEG'
        em_test_folder = test_folder + '/EM'
        seg_test_folder = test_folder + '/SEG'
        
        try:
            _, _, imgs = next(os.walk(em_train_folder))
        except:
            raise CustomErrorTypes('This function requires that your samplefolder and savefolder contain the folders EM and SEG')

        num_imgs = len(imgs)
        imgs_to_move = num_imgs - math.ceil(num_imgs*train_volume)

        idx_to_remove = random.sample(range(num_imgs), imgs_to_move)

        for i in idx_to_remove:
            Path(f'{em_train_folder}/EM_{i}.png').rename(f'{em_test_folder}/EM_{i}.png')
            Path(f'{seg_train_folder}/SEG_{i}.png').rename(f'{seg_test_folder}/SEG_{i}.png')


if __name__ == "__main__":
    container = CREMI(samplefolder='data/test', savefolder='output/', autocon=True)
    # container.preprocess()
    container.test_train_split(train_volume=0.75, test_folder='./data/test', train_folder='./data/train')
