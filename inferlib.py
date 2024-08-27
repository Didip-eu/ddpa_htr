from os import PathLike
from torch import Tensor
from typing import Union,Tuple,List


class HTR_Model:


    def __init__(self, alphabet: list):
        self._alphabet = alphabet

    @property
    def alphabet( self ):
        return self._alphabet

    def infer( self, img_bcwh: Tensor, widths: Tensor, heights: Tensor, masks: Tensor):
        pass


    def save():
        pass


    def resume(path: PathLike):
        pass

    def transcribe( self ):
        pass

    def __repr__( self ):
        return "HTR_Model()"



class Alphabet:

    """
    Internally stored as a dictionary.
    """

    def __init__( self, alpha: str):
        self._alphabet = self.from_string( alpha )

    @staticmethod
    def from_string( stg: str):
        alphadict = { c:s for (c,s) in enumerate(sorted(set( [ s for s in stg if not s.isspace() ])), start=1) }
        alphadict[0]=' '
        return alphadict


    def __len__( self ):
        return len( self._alphabet )

    def __repr__( self ) -> str:
        """ A TSV representation of the alphabet
        """
        return '\n'.join( [ f'{c}\t{s}' for (c,s) in sorted( self._alphabet.items()) ] )

    
    def __eq__( self, other ):
        return self._alphabet == other._alphabet 

    def get_symbol( self, code ) -> str:
        return self._alphabet[ code ]

    def get_code( self, symbol ) -> int:
        for (c,s) in self._alphabet.items():
            if s==symbol:
                return c
        return None

    def get_item( self, i: Union[int,str]) -> Tuple[int,str]:
        for (c,s) in self._alphabet.items():
            if (type(i) is str and i==s) or (type(i) is int and i==c):
                return (c,s)
        return None
    
    def encode(self, sample: List[str]) -> List[int]:
        return [ self.get_code( s ) for s in sample ]

    def decode(self, sample: List[int] ) -> List[str]:
        return [self.get_symbol( c ) for c in sample ] 


        

