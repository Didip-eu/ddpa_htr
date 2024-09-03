import pytest
import sys
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from handwriting_datasets import monasterium
from functools import partial

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import inferlib

@pytest.fixture(scope="module")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


def test_datapath( data_path ):
    assert isinstance(data_path, Path )


def test_model_init():
    alphabet = inferlib.Alphabet("abcd")
    assert len( inferlib.HTR_Model( alphabet ).alphabet ) == 5


def test_dataloader_file( data_path ):

    ds = monasterium.MonasteriumDataset(task='htr', work_folder=data_path.joinpath('toydataset'), build_items=False, transform=partial(monasterium.MonasteriumDataset.size_fit_transform, max_h=300, max_w=2000))
    # + should keep track of widths and heights for padded images
    dl = DataLoader( list(ds), batch_size=4) 
    assert len(dl) == 5



def dtest_infer_input():
    htr_model = inferlib.HTR_Model( inferlib.Alphabet('abcd') )

    # What is possible
    # - a directory contains both *.png and *.gt transcriptions: settle with this for the moment, for ease of testing
    # - a TSV file path with <img path>, <transcription>
    # - a TSV file path with <img path>, <transcription path>
    #ds = Monasterium( task='htr', shape='bbox', data_path.joinpath('toydataset'))
    #dl = FileDataLoader( data, batch_size=4, shuffle=True ) # what matters is this (an implementation of DataLoader)

    # could be a stream, also
    # data is a TSV file path with <img url>, <transcription>
    #dl = StreamDataLoader( data, batch_size=4, shuffle=True )

    #img_bchw, widths, heights, masks = dataset.to_tensors()

    #htr_model.infer( img_bchw, widths, heights, masks ) 



