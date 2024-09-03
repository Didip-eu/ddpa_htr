from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Union,Tuple,List
import warnings



class HTR_Model:


    def __init__(self, alphabet: 'Alphabet'):
        self.alphabet = alphabet


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
    Internally stored as 2 (synchronized) dictionaries
    - one with int code as key and utf char as value.
    - one with utf char as key and int code as value.

    Features:
        - loads from tsv or string
        - white space char (U0020=32)
        - null character (U2205)
        - default character for encoding (when dealing with unknown char)
        - 
    """
    nullchar = '\u2205'

    def __init__( self, alpha: str):
        self._code_2_utf = self.from_string( alpha )
        self._utf_2_code = { s:c for (c,s) in self._code_2_utf.items() }
        self.default_symbol = '.'
        self.default_code = self.maxcode

    @classmethod
    def from_string(cls, stg: str ):
        alphadict = { c:s for (c,s) in enumerate(sorted(set( [ s for s in stg if not s.isspace() or s==' ' ])), start=1) }
        alphadict[ max( alphadict.keys() )+1 ] = cls.nullchar
        return alphadict

    def __len__( self ):
        return len( self._code_2_utf )

    def __str__( self ) -> str:
        """ A TSV representation of the alphabet
        """
        one_symbol_per_line = '\n'.join( [ f'{c}\t{s}' for (c,s) in sorted( self._code_2_utf.items()) ] )
        return one_symbol_per_line.replace( self.nullchar, '<nul>' )

    def __repr__( self ) -> str:
        return repr( self._code_2_utf )


    @property
    def maxcode( self ):
        return max([ c for (c,s) in self._code_2_utf.items() if s!=self.nullchar ])
    
    def __eq__( self, other ):
        return self._code_2_utf == other._code_2_utf

    def __contains__(self, v ):
        if type(v) is str:
            return (v in self._utf_2_code)
        if type(v) is int:
            return (v in self._code_2_utf)
        return False

    def __getitem__( self, i: Union[int,str]) -> Union[int,str]:
        if type(i) is str:
            return self._utf_2_code[i]
        if type(i) is int:
            return self._code_2_utf[i]

    def get_symbol( self, code ) -> str:
        return self._code_2_utf[ code ] if code in self._code_2_utf else self.default_symbol

    def get_code( self, symbol ) -> int:
        return self._utf_2_code[ symbol ] if symbol in self._utf_2_code else self.default_code
    
    def encode(self, sample_s: str) -> Tensor:
        """ 
        Encode a message string with integers. 

        Input:
            sample_s (str): message string; assume clean sample: no newlines nor tabs.

        Output:
            Tensor: a tensor of integers; symbols that are not in the alphabet yield
                    a default code (=max index) while generating a user warning.
        """
        if [ s for s in sample_s if s in '\n\t' ]:
            raise ValueError("Sample contains illegal symbols: check for tabs and newlines chars.")
        missing = [ s for s in sample_s if s not in self ]
        if missing:
                warnings.warn('The following chars are not in the alphabet: {}'\
                          ' →  code defaults to {}'.format( missing, self.default_code ))
        return torch.tensor([ self.get_code( s ) for s in sample_s ], dtype=torch.int64 )

    def encode_one_hot( self, sample_s: List[str]) -> Tensor:
        """ 
        One-hot encoding of a message string.
        """
        encode_int = self.encode( sample_s )
        return torch.tensor([[ 0 if i+1!=c else 1 for i in range(len(self)) ] for c in encode_int ],
                dtype=torch.bool)

    def encode_batch(self, samples_s: List[str] ) -> Tuple[Tensor, Tensor]:
        """
        Encode a batch of messages.

        Input:
            samples_s (list): a list of strings

        Output:
            tuple( Tensor, Tensor ): a pair of tensors, with encoded batch as first element
                                     and lengths as second element.
        """
        lengths = [ len(s) for s in samples_s ] 
        batch_bw = torch.zeros( [len(samples_s), max(lengths)] )
        for r,s in enumerate(samples_s):
            batch_bw[r,:len(s)] = self.encode( s )
        return (batch_bw, torch.tensor( lengths ))


    def decode(self, sample_t: Tensor, length: int=-1 ) -> str:
        """ 
        Decode an integer-encoded sample.

        Input:
            sample_t (Tensor): a tensor of integers.
            length (int): sample's length; if -1 (default), all symbols are decoded.
        Output:
            str: string of symbols.
        """
        length = len(sample_t) if length < 0 else length
        return "".join( [self.get_symbol( c ) for c in sample_t.tolist()[:length] ] )


    def decode_batch(self, samples_bw: Tensor, lengths: Tensor=None ) -> List[ str ]:
        """
        Decode a batch of integer-encoded samples.

        Input:
            sample_bw (Tensor): each row of integer encodes a string.
            lengths (int): length to be decoded in each sample; the default
                           is full-length decoding.

        Output:
            list: a sequence of strings.
        """
        if lengths == None:
            sample_count, max_length = samples_bw.shape
            lengths = torch.full( (sample_count,), max_length )
        return [ self.decode( s, lgth ) for (s,lgth) in zip( samples_bw, lengths ) ]
        

def dummy():
    return True
