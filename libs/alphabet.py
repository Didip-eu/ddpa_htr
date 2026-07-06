from __future__ import annotations # to allow for type hints to reference the enclosing class

# stdlib
from typing import Union,Tuple,List,Dict  #,Self (>= 3.11)
import re
from pathlib import Path
import itertools
import warnings
from collections import Counter
import unicodedata
import sys

# 3rd party
import torch
from torch import Tensor
import numpy as np

# local
sys.path.append(str(Path(__file__).parents[0]))
import list_utils as lu
import alphabet_compatible
from pylelemmatize import LemmatizerBMP


class Alphabet:
    """Creating and handling alphabets: a thin wrapper around a PyLelemmatizer's mapper object, that provides the
    core functionalities, with a few added hooks for HTR training in torch.

    + filtering out irrelevant characters and extra spaces
    + CTC-compliant labeling
    + encoding/decoding routines
    + save/restore

    Eg.

        >>> mapper=LemmatizerBMP.from_alphabet_mapping( ll.charsets.mufibmp+' ', ll.charsets.ascii_lowercase+' ' )
        >>> alphabet.Alphabet( mapper )

    Design choices:
  
    + the mapper's job is to ensure a consistent transformation of the input string for training and evaluation
      purpose: if some characters should be ignored altogether (eg. '✳' or similar junk in Koenigsfelden, they should be filtered 
      out during the data curation stage, or at the latest before loading time: first because it may yield empty target 
      strings, second because the clean-up is very costly.
    + a mapping is for those characters that are deemed relevant in the GT string; a relevant char maps to
      a specific character or, by default, to the unknown character
    + mapping may be interpretable by a human (eg. 'ꝑ'→ 'p') or not (eg. 'Ꝯ'→ 'c', 'ꝝ'→ 'a'): this is not
      a problem, since the generated string is for training use

    """
    # torch.nn.CTCLoss already has default blank=0
    null_symbol = '\u03f5'
    null_value = 0
    # Unused so far 
    start_of_seq_symbol = '\u21A6' # '↦' i.e. '|->'
    end_of_seq_symbol = '\u21E5' # '⇥' i.e. '->|'

    def __init__(self, mapper: LemmatizerBMP, override_map={} ):
        """
        Construct a new HTR alphabet.

        Args:
            mapper (LemmatizerBMP): a character mapping object.
            override_map (dict): additional mappings for lemmatizer initialization (to add space mapping, f.i.).
        """
        self._mapper = mapper 
        if len(override_map):
            self._mapper = LemmatizerBMP.from_alphabet_mapping( self._mapper.src_alphabet_str, self._mapper.dst_alphabet_str, override_map = override_map )
        self.unknown_symbol = self._mapper.unknown_chr

        self._utf2lbl={ self.null_symbol: self.null_value }
        self._lbl2utf={ self.null_value: self.null_symbol }
        for lbl, utf in enumerate( self._mapper.dst_alphabet_str, start=1):
            self._utf2lbl[utf]=lbl
            self._lbl2utf[lbl]=utf


    def __repr__( self ):
        return f"Alphabet( {repr(self._mapper)} )"

    @staticmethod
    def load( alpha_repr: dict ):
        """
        
        Alphabet instance from serialization object, i.e. a dictionary of the form:

            {'mapping_dict': ..., 'unknown_chr': ... }
        """
        return eval( alpha_repr )

    def __len__(self):
        return len(self._mapper)+1 # add null char to alphabet length.


    def alphabet_tsv(self):
        lines = self._mapper.alphabet_tsv.split('\n')
        header = f"CTC-label\t{lines[0]}"
        ctc_labels = [ self._utf2lbl[chr(int(l.split('\t')[2]))] for l in lines[1:] ]
        return '\n'.join([header, f"{self.null_value}\t-\tBLANK CHARACTER\t-\t{ord(self.null_symbol)}\t'{self.null_symbol}'"] + [ f"{ctc_labels[n]}\t"+l for n,l in enumerate(lines[1:]) ] )


    def mapping_tsv(self):
        return self._mapper.mapping_tsv


    def reduce(self, sample_s: str) -> str:
        """Transform a string, for training and validation purpose.
        Rewrite a string in 2 steps:
        
        1. normalize spaces
        2. map all chars of a charset to their representative.

            >>> Alphabet( LemmatizerBMP({'c':'a', 'b':'b', 'a':'c'})).reduce('a   b  c')
            'c b a'

        Args:
            sample_s (str): message string.

        Returns:
            str: the message, where all members of a given charset have been replaced by their 
                representative.
        """
        return self._mapper( re.sub(r'\s+', ' ', sample_s ))


    def encode(self, sample_s: str) -> Tensor:
        """Encode a message string with integers. Input string is assumed 
        toh have been reduced first.

        Args:
            sample_s (str): message string.

        Returns:
            Tensor: a tensor of integers; 
        """
        return torch.tensor( [ self._utf2lbl[char] for char in sample_s ] )


    def decode(self, sample_t: Union[Tensor,np.ndarray], length: int=-1 ) -> str:
        """Decode an integer-encoded sample.
        
        Args:
            sample_t (Tensor): a tensor of integers (W,).
            length (int): sample's length; if -1 (default), all symbols are decoded.

        Returns:
             str: a string of symbols
        """
        length = len(sample_t) if length < 0 else length
        return  "".join(self._lbl2utf[i] for i in sample_t.tolist()[:length] )


    def encode_batch(self, samples_s: List[str], padded=True) -> Tuple[Tensor, Tensor]:
        """Encode a batch of messages.

        Args:
            samples_s (List[str]): a list of strings
            padded (bool): if True (default), return a tensor of size (N,S) where S is the maximum
               length of a sample mesg; otherwise, return an unpadded 1D-sequence of labels.

        Returns:
            Tuple[Tensor, Tensor]: a pair of tensors, with encoded batch as first element
                and lengths as second element.
        """
        encoded_samples = [ self.encode( s ) for s in samples_s ]
        lengths = [ len(s) for s in encoded_samples ] 

        if padded:
            batch_bw = torch.zeros( [len(samples_s), max(lengths)], dtype=torch.int64 )
            for r,s in enumerate(encoded_samples):
                batch_bw[r,:len(s)] = encoded_samples[r]
            return (batch_bw, torch.tensor( lengths ))

        return ( torch.cat( encoded_samples ), torch.tensor(lengths))


    def decode_batch(self, samples_nw: Tensor, lengths: Tensor=None ) -> List[ str ]:
        """Decode a batch of integer-encoded samples.

        Args:
            sample_nw (Tensor): each row of integers encodes a string.
            lengths (Tensor): length to be decoded in each sample; the default is full-length decoding.
        Returns:
            list: a sequence of strings.
        """
        if lengths == None:
            sample_count, max_length = samples_nw.shape
            lengths = torch.full( (sample_count,), max_length )
        return [ self.decode( s, lgth ) for (s,lgth) in zip( samples_nw, lengths ) ]


    def decode_ctc(self, msg: np.ndarray ):
        """Decode the output labels of a CTC-trained network into a human-readable string. Eg.::

            >>> Alphabet(LemmatizerBMP.from_alphabet_mapping('Hello')).decode_ctc(np.array([1,1,0,2,2,2,0,0,3,3,0,3,0,4]))
            'Hello'

        Args:
            msg (np.ndarray): a sequence of labels, possibly with duplicates and null values.

        Returns:
               str: a string of characters.
        """
        # keep track of positions to keep
        keep_idx = np.zeros( msg.shape, dtype='bool') 
        if msg.size == 0:
            return ''
        # quick removal of duplicated values
        keep_idx[0] = msg[0] != self.null_value 
        keep_idx[1:] = msg[1:] != msg[:-1] 
        # removal of null chars
        keep_idx = np.logical_and( keep_idx, msg != self.null_value )

        return self.decode( msg[ keep_idx ] )


    def to_sets( self, exclude: list=[] )-> dict[str,set]:
        """Return a set representation of the alphabet (for easy checking).
        Virtual symbols (EoS, SoS, null, unknown) are not included.

        Args:
            exclude (List[str]): list of symbols that should not be included into the resulting list. Eg::

        Returns:
             dict[str,set]: a mapping of target characters to sets of source characters.
        """
        dest_to_start = {}
        for (s,d) in self._mapper.mapping_dict.items():
            if s in (self.start_of_seq_symbol, self.end_of_seq_symbol, self.null_symbol, self.unknown_symbol) or s in exclude:
                continue
            if d in dest_to_start:
                dest_to_start[d].add( s )
            else:
                dest_to_start[d]=set([s])
        return { k:sorted(list(dest_to_start[k])) for k in sorted(dest_to_start.keys()) }

    @staticmethod
    def load_old( alphabet_as_list: list[list] ):
        """
        DEPRECATED: Load an old-style serialization, where the alphabet is stored as a list of list.
        (For backward compatibility only.)
        """
        mapping_dict = {}
        for elt in alphabet_as_list:
            if type(elt) is str:
                mapping_dict[elt]=elt
            elif type(elt) is list:
                for c in elt:
                    mapping_dict[c] = elt[0] 
        alphabet = Alphabet( LemmatizerBMP(mapping_dict = mapping_dict ))

        # Use old alphabet library to retrieve char 2 code maps
        alpha_compat = alphabet_compatible.Alphabet( alphabet_as_list )
        alphabet._utf2lbl = alpha_compat._utf_2_code
        alphabet._lbl2utf = alpha_compat._code_2_utf
        return alphabet

    
