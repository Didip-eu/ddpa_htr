from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from kraken.lib.vgsl import TorchVGSLModel
import numpy as np
import re

from typing import Union,Tuple,List
import warnings



class HTR_Model():
    """
    Note: by convention, VGSL specifies the dimensions in NHWC order, while Torch uses NCHW. The example
    below uses a NHWC = (1, 128, 2048, 3) image as an input.

    +-------------+------------------------------------------+---------------------------------------------+
    | VGSL        | DESCRIPTION                              | Output size (NHWC)     | Output size (NCHW) |
    +=============+==========================================+=============================================+
    | Cr3,13,32   | kernel filter 3x13, 32 activations relu  | 1, 128, 2048, 32       | 1, 32, 128, 2048   |
    +-------------+------------------------------------------+---------------------------------------------+
    | Do0.1,2     | dropout prob 0.1 dims 2                  | -                      | -                  |
    +-------------+------------------------------------------+---------------------------------------------+
    | Mp2,2       | Max Pool kernel 2x2 stride 2x2           | 1, 64, 1024, 32        | 1, 32, 64, 1024    | 
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         | (same)                                   | 1, 32,  512, 32        | 1, 32, 32, 512     |
    | Cr3,9,64    | kernel filter 3x9, 64 activations relu   | 1, 32,  512, 64        | 1, 64, 32, 512     |
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         |                                          |                        |                    |
    | Mp2,2       |                                          | 1, 16, 256, 64         | 1, 64, 16, 256     |
    +-------------+------------------------------------------+---------------------------------------------+
    | Cr3,9,64    |                                          | 1, 16, 256, 64         | 1, 64, 16, 256     |
    | Do0.1,2     |                                          |                        |                    |
    +-------------+------------------------------------------+---------------------------------------------+
    | S1(1x0)1,3  | reshape (N,H,W,C) into N, 1, W,C*H       | 1, 1, 256, 64x16=1024  | 1, 1024, 1, 256    |
    +-------------+------------------------------------------+---------------------------------------------+
    | Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400         | 1, 400, 1, 256     |
    |             | with 200 output channels                 | (either forward (f) or |                    |
    |             |                                          |reverse (r) would yield |                    |
    |             |                                          | 200-sized output)      |                    |
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         | (same)                                   |                        |                    |
    | Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400         | 1, 400, 1, 256     |
    +-------------+------------------------------------------+---------------------------------------------+

    """
    default_model_spec = '[4,128,0,3 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'

    def __init__(self, alphabet: 'Alphabet', model=None, model_spec=default_model_spec, decoder=None, add_output_layer=True):

        # initialize self.nn = torch Module
        if not model:
            # In kraken, TorchVGSLModel is does not work as model factory (parse() method)
            # but as a wrapper-class for a 'nn:Module' property, to which a number
            # of module-specific method calls (to()...) are forwarded. 
            # what is the added value of this complexity?
            # + model save() functionality
            # + model modification (append)
            # + handling hyper-parameters
            # +  train(), eval() switches

            # insert output layer if not already defined
            if re.search(r'O\S+ ?\]$', model_spec) is None and add_output_layer:
                model_spec = '[{} O1c{}]'.format( model_spec[1:-1], alphabet.maxcode + 1)

            self.nn = TorchVGSLModel( model_spec )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn.to( self.device )

        # encoder
        self.alphabet = alphabet
        # decoder
        self.decoder = decoder if decoder else self.decode

    def forward(self, img_nchw: Tensor, widths: Tensor=None):
        """
        The internal logics is entirely delegated to the layers wrapped 
        into the VGSL-defined module: by defaut, an instance of 
        `kraken.lib.layers.MultiParamSequential`

        Args:
            img_nchw (Tensor): a batch of line images
            widths (Tensor): sequence of image lengths

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with (N,C,W) array and
            final output sequence lengths.
        """
        if self.device:
            img_nchw = img_nchw.to( self.device )
        # note the dereferencing: the actual NN is a property of the TorchVGSL object
        o, owidths = self.nn.nn(img_nchw, widths)
        outputs_ncw = o.detach().squeeze(2).float().cpu().numpy()
        if owidths is not None:
            owidths = owidths.cpu().numpy()
        return (outputs_ncw, owidths)


        
    def train_task( self, img_nchw: Tensor, widths: Tensor=None, masks: Tensor=None, transcriptions: Tensor=None):
        pass



    def decode( self, outputs_ncw: np.ndarray ):
        pass



    def inference_task( self, img_nchw: Tensor, widths: Tensor=None, masks: Tensor=None):
       
        assert isinstance( img_nchw, Tensor )
        assert isinstance( widths, Tensor)

        outputs_ncw, output_widths = self.forward( img_nchw, widths ) 

        return outputs_ncw

    
    def save(self, file_path: str):
        self.nn.save_model( file_path )


    def resume( self, path: PathLike):
        pass


    def __repr__( self ):
        return "HTR_Model()"



def dummy():
    return True
