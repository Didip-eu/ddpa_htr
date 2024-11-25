from __future__ import annotations # to allow for type hints to reference the enclosing class

# stdlib
from typing import Union,Tuple,List,Dict  #,Self (>= 3.11)
import re
from pathlib import Path
import itertools
import warnings
from collections import Counter

# 3rd party
import torch
from torch import Tensor
import numpy as np

# local
from . import list_utils

class Alphabet:
    """Creating and handling alphabets.

    + one-to-one or many-to-one alphabet, with deterministic mapping either way;
    + prototyping from reasonable subsets of characters to be grouped
    + a choice of input/output sources: TSV, nested lists, mappings.

    """
    null_symbol = '\u03f5'
    null_value = 0
    start_of_seq_symbol = '\u21A6' # '↦' i.e. '|->'
    end_of_seq_symbol = '\u21E5' # '⇥' i.e. '->|'
    unknown_symbol = '?' 
    unknown_value = 1

    def __init__( self, alpha_repr: Union[str,list]='', tokenizer=None ) -> None:
        """Initialize a new Alphabet object. The special characters are added automatically.

            From a nested list::

                >>> alphabet.Alphabet([['a','A'],'b','c'])
                {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

            From a string of characters (one-to-one)::

                >>> alphabet.Alphabet('aAbc ')
                {' ': 1, 'A': 2, 'a': 3, 'b': 4, 'c': 5, 'ϵ': 0, '↦': 6, '⇥': 7}

            Returns:
                alpha_repr (Union[str, list]): the input source--it may be a nested list, or a plain string.
        """
        self._utf_2_code = {}

        if type(alpha_repr) is str:
            self._utf_2_code = self._dict_from_string( alpha_repr )
        elif type(alpha_repr) is list:
            self._utf_2_code = self._dict_from_list( alpha_repr )

        self._finalize()

        # crude, character-splitting function makes do for now
        # TODO: a proper tokenizer that splits along the given alphabet
        self.tokenize = self._tokenize_crude if tokenizer is None else tokenizer

    @property
    def many_to_one( self ):
        return not all(i==1 for i in Counter(self._utf_2_code.values()).values())

    def _finalize( self ) -> None:
        """Finalize the alphabet's data:

        * Add virtual symbols: EOS, SOS, null symbol, default symbol
        * compute the reverse dictionary
        """
        self._utf_2_code[ self.null_symbol ] = self.null_value     # default, null value = characters that are never to be encoded
        self._utf_2_code[ self.unknown_symbol ] = self.unknown_value # unknown symbol (for out-of-alphabet characters) that are encoded as unknown

        for s in (self.start_of_seq_symbol, self.end_of_seq_symbol):
            if s not in self._utf_2_code:
                self._utf_2_code[ s ] = self.maxcode+1
        
        if self.many_to_one:
            # ensure that a many-to-one code maps to first character in a sublist 
            self._code_2_utf = { c:s for (s,c) in reversed(self._utf_2_code.items()) }
        else:
            self._code_2_utf = { c:s for (s,c) in self._utf_2_code.items() }
            

    def to_tsv( self, filename: Union[str,Path]) -> None:
        """Dump to TSV file.

        Args:
            filename (Union[str,Path]): path to TSV file
        """
        with open( filename, 'w') as of:
            print(self, file=of)


    def to_list( self, exclude: list=[] )-> List[Union[str,list]]:
        """Return a list representation of the alphabet.

        Virtual symbols (EoS, SoS, null, unknown) are not included, so that it can be fed back
        to the initialization method.

        Args:
            exclude (List[str]): list of symbols that should not be included into the resulting list. Eg::

                >>> alphabet.Alphabet([['a','A'],'b','c']).to_list(['a','b'])
                ['A', 'c']

        Returns:
             List[Union[str,list]]: a list of lists or strings.
        """
        code_2_utfs = {}
        for (s,c) in self._utf_2_code.items():
            if s in (self.start_of_seq_symbol, self.end_of_seq_symbol, self.null_symbol, self.unknown_symbol) or s in exclude:
                continue
            if c in code_2_utfs:
                code_2_utfs[c].add( s )
            else:
                code_2_utfs[c]=set([s])
        return sorted([ sorted(list(l)) if len(l)>1 else list(l)[0] for l in code_2_utfs.values() ], key=lambda x: x[0])
        

        
    def __len__( self ):
        return len( self._code_2_utf )

    def __str__( self ) -> str:
        """A summary"""
        one_symbol_per_line = '\n'.join( [ f'{s}\t{c}' for (s,c) in  sorted(self._utf_2_code.items()) ] )
        return one_symbol_per_line.replace( self.null_symbol, '\u03f5' )

    def __repr__( self ) -> str:
        return repr( self.to_list() )


    @property
    def maxcode( self ):
        return max( list(self._utf_2_code.values()) )
    
    def __eq__( self, other ):
        return self._utf_2_code == other._utf_2_code

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

    def get_symbol( self, code, all=False ) -> Union[str, List[str]]:
        """Return the class representative (default) or all symbols that map on the given code.

        Args:
            code (int): a integer code.
            all (bool): if True, returns all symbols that map to the given code; if False (default),
                returns the class representative.

        Returns:
            Union[str, List[str]]: the default symbol for this code, or the list of matching symbols.
        """
        if all:
            return [ s for (s,c) in self._utf_2_code.items() if c==code ]
        return self._code_2_utf[ code ] if code in self._code_2_utf else self.default_symbol

    def get_code( self, symbol ) -> int:
        """Return the code on which the given symbol maps.

        For symbols that are not in the alphabet, the default code (1) is returned.

        Args:
            symbol (str): a character.

        Returns:
            int: an integer code
        """
        return self._utf_2_code[ symbol ] if symbol in self._utf_2_code else self.unknown_value


    def stats( self ) -> dict:
        """Basic statistics."""
        return { 'symbols': len(set(self._utf_2_code.values()))-3,
                 'codes': len(set(self._utf_2_code.keys()))-3,
               }


    def symbol_intersection( self, alph: Self )->set:
        """Returns a set of those symbols that can be encoded in both alphabets.

        Args:
            alph (Alphabet): an Alphabet object.

        Returns:
            set: a set of symbols.
        """
        return set( self._utf_2_code.keys()).intersection( set( alph._utf_2_code.keys()))

    def symbol_differences( self, alph: Self ) -> Tuple[set,set]:
        """Compute the difference of two alphabets.

        Args:
            alph (Alphabet): an Alphabet object.

        Returns:
            Tuple[set, set]: a tuple with two sets - those symbols that can be encoded with the first alphabet, but
                 not the second one; and conversely.
        """
        return ( set(self._utf_2_code.keys()).difference( set( alph._utf_2_code.keys())),
                 set(alph._utf_2_code.keys()).difference( set( self._utf_2_code.keys())))


    def encode(self, sample_s: str) -> Tensor:
        """Encode a message string with integers: the string is segmented first.

        Args:
            sample_s (str): message string, clean or not.

        Returns:
            Tensor: a list of integers; 
        """
        sample_s = self.normalize_spaces( sample_s )
        return torch.tensor( [ self.get_code( t ) for t in self.tokenize( sample_s ) ], dtype=torch.int64)


    def encode_one_hot( self, sample_s: List[str]) -> Tensor:
        """One-hot encoding of a message string."""
        encode_int = self.encode( sample_s )
        return torch.tensor([[ 0 if i!=c else 1 for i in range(len(self)) ] for c in encode_int ],
                dtype=torch.bool)

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


    def decode(self, sample_t: Tensor, length: int=-1 ) -> str:
        """Decode an integer-encoded sample.
        
        Args:
            sample_t (Tensor): a tensor of integers (W,).
            length (int): sample's length; if -1 (default), all symbols are decoded.

        Returns:
             str: a string of symbols
        """
        length = len(sample_t) if length < 0 else length
        return "".join( [self.get_symbol( c ) for c in sample_t.tolist()[:length] ] )


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

            >>> alphabet.Alphabet('Hello').decode_ctc(np.array([1,1,0,2,2,2,0,0,3,3,0,3,0,4]))
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

        return ''.join( self.get_symbol( c ) for c in msg[ keep_idx ] )
        

    @staticmethod
    def normalize_spaces(mesg: str) -> str:
        """Normalize the spaces:

        * remove trailing spaces
        * all spaces mapped to standard space (`' '=\\u0020`)
        * duplicate spaces removed

        Eg.::

           >>> normalize_spaces('\\t \\u000Ba\\u000C\\u000Db\\u0085c\\u00A0\\u2000\\u2001d\\u2008\\u2009e')
           ['a b c d e']

        Args:
            mesg (str): a string

        Returns:
               str: a string
        """
        return re.sub( r'\s+', ' ', mesg.strip())


    @classmethod
    def prototype_from_data_paths(cls, 
                                std_charsets: List[str],
                                paths: List[str], 
                                merge:List[str]=[],
                                many_to_one:bool=True,) -> Tuple[ Alphabet, Dict[str,str]]:
        """Given a list of GT transcription file paths, return an alphabet.

        Args:
            std_charsets (List[str]): a list of charsets (strings of chars), considered 
                "standard" for the dataset. Eg.::
            
                    [' ', '1', '2', ..., '9', 'AÁÂÃÄÅÆĂĄÀ', 'aáâãäåæāăąàæ', ..., 'zźżž']

            paths (List[str]): a list of file paths (wildards accepted).

            merge (List[str]): for each of the provided subsequences, merge those output sublists that
                contain the characters in it. Eg. `merge=['ij']` will merge the `'i'` sublist
                (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)

            many_to_one (bool): if True (default), builds a many-to-one alphabet, based on the
                Alphabet class' character classes.

        Returns:
            Tuple[Alphabet, Dict[str,str]]: a pair with 
                * an Alphabet object 
                * a dictionary `{ symbol: [filepath, ... ]}` that assigns to each non-standard symbol
                  all the files in which it appears.
        """
        assert type(paths) is list
        charset = set()
        file_paths = []
        for p in paths:
            path = Path(p)
            if '*' in path.name:
                file_paths.extend( path.parent.glob( path.name ))
            elif path.exists():
                file_paths.append( path )

        # for diagnosis on unexpected, non-standard symbols: populate 
        # a symbol-to-file(s) dictionary
        char_to_file = {}
        for fp in file_paths:
            with open(fp, 'r') as infile:
                chars_in_this_file = set( char for line in infile for char in list(line.strip())  ).difference()
                for c in chars_in_this_file.difference( set(''.join(std_charsets))) :
                    if c in char_to_file:
                        char_to_file[ c ].append( fp.name )
                    else:
                        char_to_file[ c ] = [ fp.name ]
                charset.update( chars_in_this_file )

        return (cls.charset_to_alphabet( charset, std_charsets, merge, many_to_one ), char_to_file )


    @classmethod
    def prototype_from_data_samples(cls, std_charsets: List[str], transcriptions: List[str], 
                                    merge:List[str]=[], many_to_one:bool=True,) -> Alphabet:
        """Given a list of GT transcription strings, return an Alphabet.

        Args:
            std_charsets (List[str]): a list of charsets (strings of chars), considered 
                "standard" for the dataset. Eg.::
            
                    [' ', '1', '2', ..., '9', 'AÁÂÃÄÅÆĂĄÀ', 'aáâãäåæāăąàæ', ..., 'zźżž']

            transcriptions (List[str]): a list of transcriptions.

            merge (List[str]): for each of the provided subsequences, merge those output 
                sublists that contain the characters in it. Eg. `merge=['ij']` will merge the `'i'`
                sublist (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)

            many_to_one (bool): if True (default), builds a many-to-one alphabet, based on
                the Alphabet class' character classes.

        Returns:
                Alphabet: an Alphabet object

        """
        charset = set()

        for tr in transcriptions:
            chars = set( list(tr.strip())  )
            charset.update( chars )

        return cls.charset_to_alphabet( charset, std_charsets, merge, many_to_one )

        
    @classmethod
    def prototype_from_scratch( cls, std_charsets: List[str], merge:List[str]=[],) -> Alphabet:
        """Build a tentative, "universal", alphabet from scratch, without regard to the data: it
        maps every class of characters (charset) to a common code.

        Args:
            std_charsets (List[str]): a list of charsets (strings of chars), considered 
                "standard" for the dataset. Eg.::
            
                    [' ', '1', '2', ..., '9', ['A','Á','Â',...], ['a','á',...], ..., ['z','ź',...]]

            merge (List[str]): for each of the provided subsequences, merge those output sublists
                that contain the characters in it. Eg. `merge=['ij']` will merge the `'i'` sublist
                (`['i','I','î',...]`) with the `'j'` sublist (`['j','J',...]`)

        Returns:
             Alphabet: an Alphabet object
        """

        symbol_list = cls.list_utils.groups_from_groups( std_charsets )
        symbol_list = cls.list_utils.merge_sublists( symbol_list, merge )        

        return cls(symbol_list)


    @classmethod
    def charset_to_alphabet( cls, charset: List[str], std_charsets: List[str], merge:List[str], many_to_one):

        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    

        weird_chars = charset.difference( set(''.join( std_charsets )))
        if weird_chars:
            warnings.warn("The following characters are in the data, but not in the 'standard' charsets used by this prototype: {}".format( weird_chars ))

        symbol_list = cls.list_utils.groups_from_groups(std_charsets, charset) if many_to_one else sorted(charset)
        symbol_list = cls.list_utils.merge_sublists( symbol_list, merge )

        return cls(symbol_list)


    @classmethod
    def _dict_from_list(cls, symbol_list: List[Union[List,str]]) -> Dict[str,int]:
        """Construct a symbol-to-code dictionary from a list of strings or sublists of symbols (for many-to-one alphabets):
        symbols in the same sublist are assigned the same label.
        Works on many-to-one, compound symbols. Eg.::

            >>> from_list( [['A','ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c'] )
            { 'A':2, 'U':3, 'ae':2, 'b':4, 'c':5, 'u':6, 'ue':6, ... }

        Args:
            symbol_list (List[Union[List,str]]): a list of either symbols (possibly with more
                than one characters) or sublists of symbols that should map to the same code.

        Returns:
            Dict[str,int]: a dictionary mapping symbols to codes.
        """
        def flatten( l:list):
            if l == []:
                return l
            if type(l[0]) is not list:
                return [l[0]] + flatten(l[1:])
            return flatten(l[0]) + flatten(l[1:])

        flat_list = flatten( symbol_list )
        if len(flat_list) != len(set(flat_list)):
            duplicates = list( itertools.filterfalse( lambda x: x[1]==1, Counter(sorted(flat_list)).items()))
            raise ValueError(f"Duplicates characters in the input sublists: {duplicates}")

        # if list is not nested (one-to-one)
        if all( type(elt) is str for elt in symbol_list ):
            return {s:c for (c,s) in enumerate( sorted( symbol_list), start=2)}

        # nested list (many-to-one)
        reserved_symbols = (cls.start_of_seq_symbol, cls.end_of_seq_symbol, cls.null_symbol, cls.unknown_symbol)
        def sort_and_label( lol ):
            lol = itertools.filterfalse( lambda x: x in reserved_symbols, lol )
            lol = [ (c,s) for (c,s) in enumerate(sorted([ sub for sub in lol ], key=lambda x: x[0]), start=2)] 
            print("sorted_and_labeled=", lol)
            return lol

        sall = sort_and_label( symbol_list )
        # single-character symbols
        alphadict ={ s:c for (c,s) in sall if type(s) is str and (not s.isspace() or s==' ') }
        # compound symbols
        alphadict.update( {s:c for (c,item) in sall for s in item if type(item) is list }) 
        print("alphadict=",alphadict)
        return alphadict

    @classmethod
    def _dict_from_string(cls, stg: str ) -> Dict[str,int]:
        """Construct a one-to-one alphabet from a single string.

        Args:
            stg (str): a string of characters.
        
        Returns:
            Dict[str,int]: a `{ code: symbol }` mapping.
        """
        alphadict = { s:c for (c,s) in enumerate(sorted(set( [ s for s in stg if not s.isspace() or s==' ' ])), start=2) }
        return alphadict

    def _tokenize_crude( self, mesg: str, quiet=True ) -> List[str]:
        """Tokenize a string into tokens that are consistent with the provided alphabet.
        A very crude splitting, as a provision for a proper tokenizer. Spaces
        are normalized (only standard spaces - `' '=\\u0020`)), with duplicate spaces removed.

        Args:
            mesg (str): a string

        Returns:
            List[str]: a list of characters.
        """

        if not quiet:
            missing = set( s for s in mesg if s not in self )
            if len(missing)>0:
                warnings.warn('The following chars are not in the alphabet: {}'\
                          ' →  code defaults to {}'.format( [ f"'{c}'={ord(c)}" for c in missing ], self.default_code ))

        return list( mesg )
