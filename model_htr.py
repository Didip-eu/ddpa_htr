from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from kraken.lib.vgsl import TorchVGSLModel

from typing import Union,Tuple,List
import warnings



class HTR_Model():

    default_model_spec = '[4,300,1300,3 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'

    def __init__(self, alphabet: 'Alphabet', model=None, model_spec=default_model_spec):

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
            self.nn = TorchVGSLModel( model_spec )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn.to( self.device )

        # encoder
        self.alphabet = alphabet

    def forward(self, img_nchw: Tensor, widths: Tensor=None):
        if self.device:
            line = line.to( self.device )
        # note the dereferencing: the actual NN is a property of the TorchVGSL object
        outputs, _ = self.nn.nn(img_nchw, widths)
        self.outputs = o.detach().squeeze(2).float().cpu().numpy()
        return outputs


        
    def train_task( self, img_nchw: Tensor, heights: Tensor=None, widths: Tensor=None, masks: Tensor=None, transcriptions: Tensor=None):
        pass





    def inference_task( self, img_nchw: Tensor, heights: Tensor=None, widths: Tensor=None, masks: Tensor=None, transcriptions: Tensor=None):
       
        assert isinstance( img_nchw, Tensor ) #and img_nchw.dim() == 4
        assert isinstance( heights, Tensor)
        assert isinstance( widths, Tensor)
        assert isinstance( masks, Tensor)
        assert all(isinstance( trsc, str) for gt in transcriptions)
        

        return img_nchw.size()
    
    def save(self, file_path: str):
        self.save_model( file_path )


    def resume( self, path: PathLike):
        pass

    def transcribe( self ):
        pass

    def __repr__( self ):
        return "HTR_Model()"



def dummy():
    return True
