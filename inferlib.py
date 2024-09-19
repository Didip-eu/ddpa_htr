from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Union,Tuple,List
import warnings



class HTR_Model:


    def __init__(self, alphabet: 'Alphabet'):
        self.alphabet = alphabet


    def infer( self, img_bchw: Tensor, heights: Tensor=None, widths: Tensor=None, masks: Tensor=None):
        print("infer(): img_bchw=", img_bchw )
        #print(type (img_bchw ))
        assert isinstance( img_bchw, Tensor ) #and img_bchw.dim() == 4
        #assert isinstance( heights, Tensor)
        #assert isinstance( widths, Tensor)
        #assert all( [ isinstance( gt, str) for gt in b['transcription'] ] )
        #assert all( [ isinstance( m, Tensor) for m in masks ] )
        
        return True


    def save( self ):
        pass


    def resume( self, path: PathLike):
        pass

    def transcribe( self ):
        pass

    def __repr__( self ):
        return "HTR_Model()"



        

def dummy():
    return True
