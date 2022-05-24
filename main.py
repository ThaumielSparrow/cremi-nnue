from preprocess import CREMI
from train import main as train_model

init_train = CREMI(samplefolder='samples/train/', savefolder='data/train/', autocon=True)
init_train.preprocess()

init_test = CREMI(samplefolder='samples/test/', savefolder='data/test/', autocon=True)
init_test.preprocess()

# train_model()