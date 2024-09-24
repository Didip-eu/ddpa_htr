from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from kraken.lib import vgsl

from typing import Union,Tuple,List
import warnings



class HTR_Model( vgsl.TorchVGSLModel ):

    default_model_spec = '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'

    def __init__(self, alphabet: 'Alphabet', model=None, model_spec=default_model_spec):

        super().__init__( model_spec )

        # build the network (default: use kraken's VGSL lib)

        self.alphabet = alphabet

    def train( self ):
        pass


    def infer( self, img_bchw: Tensor, heights: Tensor=None, widths: Tensor=None, masks: Tensor=None, gts: Tensor=None):
       
        assert isinstance( img_bchw, Tensor ) #and img_bchw.dim() == 4
        assert isinstance( heights, Tensor)
        assert isinstance( widths, Tensor)
        assert isinstance( masks, Tensor)
        assert all(isinstance( gt, str) for gt in gts)
        
        # forward()

        return img_bchw.size()
    
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
