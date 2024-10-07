import pytest
import sys
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from didip_handwriting_datasets import monasterium, alphabet
from torchvision.transforms import PILToTensor, ToPILImage, Compose
import numpy as np

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import model_htr

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


def test_dummy( data_path ):
    assert model_htr.dummy()
    assert isinstance(data_path, Path )

@pytest.fixture(scope="session")
def standalone_alphabet():
    alpha = alphabet.Alphabet([' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'],
                              ['B', 'b'], ['C', 'c'], ['D', 'd'], ['E', 'e', 'é'], ['F', 'f'], ['G', 'g'],
                              ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'],
                              ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'],
                              ['S', 's'], ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'], 
                              ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬', '…'])
    return alpha

@pytest.fixture(scope="session")
def bbox_data_set_padded( data_path ):
    return monasterium.MonasteriumDataset(
            task='htr', shape='bbox',
            from_tsv_file=data_path.joinpath('bbox', 'monasterium_ds_train.tsv'),
            transform=Compose([ monasterium.ResizeToMax(128,2048), monasterium.PadToSize(128,2048) ]))


@pytest.fixture(scope="function")
def serialized_model_path( data_path ):
    """ To be used for checking that a file has indeed been created. """
    model_path = data_path.joinpath( "model.mlmodel")
    model_path.unlink( missing_ok=True)
    yield model_path
    #model_path.unlink( missing_ok=True )

def test_model_alphabet_initialization( standalone_alphabet ):

    assert len( model_htr.HTR_Model( standalone_alphabet ).alphabet ) == 42


@pytest.fixture(scope="session")
def data_loader_1_size( bbox_data_set_padded ):

    # + should keep track of widths and heights for padded images
    dl = DataLoader( bbox_data_set_padded, batch_size=1) 
    return dl

@pytest.fixture(scope="session")
def data_loader_4_size( bbox_data_set_padded ):

    # + should keep track of widths and heights for padded images
    dl = DataLoader( bbox_data_set_padded, batch_size=4) 
    return dl

def test_inference_batch_breakup( data_loader_4_size, standalone_alphabet ):
    model = model_htr.HTR_Model( standalone_alphabet )

    b = next(iter(data_loader_4_size))
    assert model.inference_task( b['img'], b['width'], b['mask']).shape == (4,42,256)


def test_data_loader_4_size_batch_structure_img( data_loader_4_size, standalone_alphabet ):
    model = model_htr.HTR_Model( standalone_alphabet )
    b = next(iter(data_loader_4_size))

    assert [ isinstance(i, Tensor) for i in b['img'] ] == [ True ] * 4

def test_data_loader_4_size_batch_structure_height_width( data_loader_4_size, standalone_alphabet ):
    model = model_htr.HTR_Model( standalone_alphabet )
    b = next(iter(data_loader_4_size))

    assert isinstance(b['height'], Tensor) and len(b['height'])==4
    assert isinstance(b['width'], Tensor) and len(b['width'])==4

def test_data_loader_4_size_batch_structure_transcription( data_loader_4_size, standalone_alphabet ):
    model = model_htr.HTR_Model( standalone_alphabet )
    b = next(iter(data_loader_4_size))

    assert [ type(t) for t in b['transcription'] ] == [ str ] * 4

def test_data_loader_4_size_batch_structure_mask( data_loader_4_size, standalone_alphabet ):
    model = model_htr.HTR_Model( standalone_alphabet )
    b = next(iter(data_loader_4_size))

    assert [ isinstance(i, Tensor) for i in b['mask'] ] == [ True ] * 4
    assert [ i.dtype for i in b['mask'] ] == [ torch.bool ] * 4


def test_model_init_nn_type( standalone_alphabet):
    """
    Model initialization constructs a basic torch Module
    """
    model_spec = '[4,128,2048,3 Cr3,13,32]'
    vgsl_model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec, add_output_layer=False ).nn

    assert isinstance(vgsl_model.nn, torch.nn.Module)
    #                           N C   H    W
    assert vgsl_model.input == (4,3,128,2048)

def test_model_forward_default_length_convolutions( data_loader_1_size, standalone_alphabet ):
    """
    Sanity testing on a few layers of convolution + maxpool
    """
    # testing with 2048-wide images
    model_spec = '[1,128,0,3 Cr3,13,32 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2]'
    model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec, add_output_layer=False )
    
    # shape: [1,3,128,2048]
    b = next(iter(data_loader_1_size))
    outputs, _ = model.forward( b['img'] )
    #                        N  C  H   W
    assert outputs.shape == (1,64,16,256)

    
def test_model_forward_default_length_lstm( data_loader_1_size, standalone_alphabet ):
    """
    Sanity testing on a layers of convolutions/maxpool + LSTMs
    """

    # input 3 x 128 x 2048 line images
    model_spec = '[1,128,0,3 Cr3,13,32 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'
    model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec, add_output_layer=False )
    
    b = next(iter(data_loader_1_size))
    outputs, _ = model.forward( b['img'] )
    #                        N   C   W
    assert outputs.shape == (1,400,256) # the 1-H dimension produced by the reshaping has been squeezed


def test_model_forward_default_length_output_layer( data_loader_1_size, standalone_alphabet ):
    """
    Sanity testing on a layers of convolutions/maxpool + LSTMs
    """

    # input 3 x 128 x 2048 line images
    model_spec = '[1,128,0,3 Cr3,13,32 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'
    model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec )
    
    b = next(iter(data_loader_1_size))
    outputs, _ = model.forward( b['img'] )
    #                        N   C   W
    print(outputs)
    assert outputs.shape == (1,standalone_alphabet.maxcode+1,256) # the 1-H dimension produced by the reshaping has been squeezed


def test_model_forward_with_lengths( data_loader_4_size, standalone_alphabet ):
    """
    Sanity testing on a layers of convolutions/maxpool + LSTMs, with widths as an extra parameter
    """
    # input 3 x 128 x 2048 line images
    model_spec = '[4,128,0,3 Cr3,13,32 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'
    model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec, add_output_layer=False )
    
    b = next(iter(data_loader_4_size))
    #print("In test: seq_len=", b['width'])
    outputs, lengths = model.forward( b['img'], b['width'] )
    #                        N   C   W
    assert outputs.shape == (4,400,256) # the 1-H dimension produced by the reshaping has been squeezed
    assert np.array_equal(lengths, np.array([244.0, 243.0, 240.0, 240.0]))


def test_model_forward_with_lengths_and_output_layers( data_loader_4_size, standalone_alphabet ):
    """
    Sanity testing on a layers of convolutions/maxpool + LSTMs + output layer, with widths as an extra parameter
    """
    # input 3 x 128 x 2048 line images
    model_spec = '[4,128,0,3 Cr3,13,32 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'
    model = model_htr.HTR_Model( standalone_alphabet, model_spec=model_spec )
    
    b = next(iter(data_loader_4_size))
    #print("In test: seq_len=", b['width'])
    outputs, lengths = model.forward( b['img'], b['width'] )
    #                        N                             C   W
    assert outputs.shape == (4,standalone_alphabet.maxcode+1,256) # the 1-H dimension produced by the reshaping has been squeezed
    assert np.array_equal(lengths, np.array([244.0, 243.0, 240.0, 240.0]))


def test_model_decode_greedy_no_lengths():
    network_output = array([[2.1577506, 2.4033802, 3.1110797, 1.9402646, 2.0998254, 2.63057  ,
        2.2995133, 3.003648 , 2.8692765, 2.5694218],
       [2.5967445, 3.49689  , 2.9399962, 3.1172242, 2.7738907, 3.9912455,
        3.5913873, 3.3765142, 2.2684784, 3.2620287],
       [3.4117193, 3.3247042, 4.0848   , 3.5203876, 3.2172518, 2.5073898,
        2.889554 , 3.3295877, 2.746986 , 3.1198492],
       [1.6278616, 1.2983577, 1.5978011, 2.3003535, 1.5130877, 1.695713 ,
        1.9486556, 1.5438657, 2.1168454, 1.5031735],
       [3.1382027, 3.3764248, 2.105332 , 2.8138309, 2.826167 , 3.27912  ,
        2.5161686, 2.6481047, 2.8865762, 2.5390067],
       [2.018828 , 2.0418115, 1.4159603, 2.1193213, 1.7868809, 2.30061  ,
        1.326269 , 1.565183 , 1.5320983, 1.962064 ],
       [1.8722607, 2.3883576, 2.2617729, 2.855318 , 2.1102426, 2.038532 ,
        2.4088457, 2.5124578, 2.6301372, 2.0920002],
       [3.3453794, 3.2000327, 2.7664754, 3.5710936, 3.0939007, 3.635675 ,
        3.4741216, 3.2699413, 4.17768  , 2.8730369],
       [2.542821 , 1.7844386, 2.5300303, 2.026345 , 2.4741833, 1.8136001,
        2.2097301, 2.4627254, 1.7341547, 2.1094844],
       [1.9897654, 2.2448957, 2.6281688, 1.256404 , 2.5597622, 1.676508 ,
        2.3310556, 1.6996291, 2.1247885, 2.3254204]], dtype=float32)


    assert model_htr.HTR_Model.decode_greedy( network_outputs ) == [(2, 3.4117193),
 (1, 3.49689),
 (2, 4.0848),
 (7, 3.5710936),
 (2, 3.2172518),
 (1, 3.9912455),
 (1, 3.5913873),
 (1, 3.3765142),
 (7, 4.17768),
 (1, 3.2620287)]


def test_model_save( standalone_alphabet, serialized_model_path):
    """
    Model initialization constructs a torch Module
    """
    model = model_htr.HTR_Model( standalone_alphabet )
    model.save( str( serialized_model_path ))

    assert serialized_model_path.exists()


def test_inference_task( data_loader_4_size, standalone_alphabet ):

    model = model_htr.HTR_Model( standalone_alphabet )
    b = next(iter(data_loader_4_size))
    assert model.inference_task( b['img'], b['width'], b['mask'] ).shape == (4,42,256)

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

    #model_htr.inference_task( img_bchw, widths, heights, masks ) 



