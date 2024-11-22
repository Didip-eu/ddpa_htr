import pytest
import sys
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
import random

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import alphabet

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def alphabet_one_to_one_tsv(data_path):
    return data_path.joinpath('alphabet_one_to_one_repr_without_nullchar.tsv')

@pytest.fixture(scope="session")
def alphabet_one_to_one_tsv_nullchar(data_path):
    return data_path.joinpath('alphabet_one_to_one_repr_with_nullchar.tsv')

@pytest.fixture(scope="session")
def alphabet_many_to_one_tsv(data_path):
    return data_path.joinpath('alphabet_many_to_one_repr.tsv')

@pytest.fixture(scope="session")
def alphabet_many_to_one_prototype_tsv( data_path ):
    return data_path.joinpath('alphabet_many_to_one_prototype.tsv')

@pytest.fixture(scope="session")
def gt_transcription_samples( data_path ):
    return [ str(data_path.joinpath(t)) for t in ('transcription_sample_1.gt.txt', 'transcription_sample_2.gt.txt') ]

def test_alphabet_dict_from_string():
    """
    Raw dictionary reflects the given string: no less, no more; no virtual chars (null, ⇥, ...)
    """
    # unique symbols, sorted
    assert alphabet.Alphabet.from_string('ßaafdbce →e') == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'ß': 9, '→': 10, }
    # space chars ignored
    assert alphabet.Alphabet.from_string('ßaf \u2009db\n\tce\t→') == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'ß': 9, '→': 10}


def test_alphabet_dict_from_tsv_with_null_char( alphabet_one_to_one_tsv_nullchar ):
    """
    Raw dict contains everything that is in the TSV
    """
    # null char
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_one_to_one_tsv_nullchar) )
    # unique symbols, sorted
    assert alpha == {'ϵ': 0, '?': 1, ' ': 2, ',': 3, 'A': 4, 'J': 11, 'R': 16, 'S': 17, 'V': 18,
                     'b': 21, 'c': 22, 'd': 23, 'o': 33, 'p': 34, 'r': 35, 'w': 40, 
                     'y': 41, 'z': 42, '¬': 43, 'ü': 44}

def test_alphabet_dict_from_tsv_without_null_char( alphabet_one_to_one_tsv ):
    """
    Raw dict contains nothing more than what is in the TSV
    """
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_one_to_one_tsv) )
    # unique symbols, sorted
    assert alpha == {' ': 2, ',': 3, 'A': 4, 'J': 11, 'R': 16, 'S': 17, 'V': 18,
                     'b': 21, 'c': 22, 'd': 23, 'o': 33, 'p': 34, 'r': 35, 'w': 40, 
                     'y': 41, 'z': 42, '¬': 43, 'ü': 44}

def test_alphabet_from_list_one_to_one():
    input_list = ['A', 'a', 'J', 'b', 'ö', 'o', 'O', 'ü', 'U', 'w', 'y', 'z', 'd', 'D']
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'A': 2, 'D': 3, 'J': 4, 'O': 5, 'U': 6, 'a': 7, 'b': 8, 'd': 9, 'o': 10, 'w': 11, 'y': 12, 'z': 13, 'ö': 14, 'ü': 15}

def test_alphabet_from_list_compound_symbols_one_to_one():
    input_list = ['A', 'ae', 'J', 'ü', 'eu', 'w', 'y', 'z', '...', 'D']
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'...': 2, 'A': 3, 'D': 4, 'J': 5, 'ae': 6, 'eu': 7, 'w': 8, 'y': 9, 'z': 10, 'ü': 11} 

def test_alphabet_from_list_many_to_one():
    input_list = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'A': 2, 'D': 3, 'J': 4, 'O': 5, 'U': 6, 'a': 2, 'b': 7, 'd': 3, 'o': 5, 'w': 8, 'y': 9, 'z': 10, 'ö': 5, 'ü': 11}
                    
def test_alphabet_from_list_realistic():
    """ Passing a list with virtual symbols (EoS, SoS) yields a correct mapping 
    """
    input_list = [ ' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'], ['B', 'b'], ['C', 'c'], [    'D', 'd'], ['E', 'e', 'é'], '⇥', ['F', 'f'], ['G', 'g'], ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'], ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'], ['S', 's'], '↦', ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'], ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬','…' ]
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {' ': 2, ',': 3, '-': 4, '.': 5, '1': 6, '2': 7, '4': 8, '5': 9, '6': 10, ':': 11, ';': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, '¬': 39, 'Ö': 27, 'ä': 13, 'é': 17, 'ö': 27, 'ü': 33, 'ÿ': 37, 'ř': 30, 'Ž': 38, '…': 40} 


def test_alphabet_many_to_one_from_tsv( alphabet_many_to_one_tsv ):
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_many_to_one_tsv) )
    # unique symbols, sorted
    assert alpha == {'ϵ': 0, 'A': 1, 'D': 10, 'J': 2, 'O': 4, 'U': 6, 'a': 1, 'b': 3, 
                     'd': 10, 'o': 4, 'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 5}

def test_alphabet_many_to_one_prototype_tsv( alphabet_many_to_one_prototype_tsv ):
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_many_to_one_prototype_tsv), prototype=True)
    assert alpha == {'A': 2, 'D': 3, 'J': 4, 'O': 5, 'U': 6, 'ae': 2, 'b': 7, 'd': 3, 'o': 5, 'w': 8, 'y': 9, 'z': 10, 'ö': 5, 'ü': 11}


def test_alphabet_many_to_one_init( alphabet_many_to_one_tsv ):
    alpha = alphabet.Alphabet( str(alphabet_many_to_one_tsv) )
    # unique symbols, sorted
    assert alpha._utf_2_code == {'A': 1, 'D': 10, 'J': 2, 'O': 4, 'U': 6, 'a': 1, 'b': 3, 'd': 10, 'o': 4, 'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 5, 'ϵ': 0, '?': 1, '↦': 11, '⇥': 12}
    assert alpha._code_2_utf == {12: '⇥', 11: '↦', 0: 'ϵ', 5: 'ü', 4: 'o', 9: 'z', 8: 'y', 7: 'w', 10: 'd', 3: 'b', 1: '?', 6: 'u', 2: 'j'}

def test_alphabet_many_to_one_deterministic_tsv_init(data_path):
    """ Given a code, a many-to-one alphabet from tsv consistently returns the same symbol,
        no matter the order of the items in the input file.
    """
    # initialization from the same input (TSV here) give consistent results
    symbols = set()
    for i in range(10):
        symbols.add( alphabet.Alphabet( str(data_path.joinpath('lol_many_to_one_shuffled_{}'.format(i))) ).get_symbol(2))
    assert len(symbols) == 1


def test_alphabet_many_to_one_deterministic_dict_init():
    """
    Initialization from dictionaries in different orders (but same mapping) gives consistent results
    """
    key_values = [ ('A',1), ('D',10), ('J',2), ('O',4), ('U',6), ('a',1), ('b',3), ('d',10), ('o',4), ('w',7), ('y',8), ('z',9), ('ö',4), ('ü',5) ]
    symbols = set()
    for i in range(10):
        random.shuffle( key_values )
        symbols.add( alphabet.Alphabet( { k:v for (k,v) in key_values } ).get_symbol(2) )
    assert len(symbols) == 1


def test_alphabet_many_to_one_deterministic_list_init():
    """ 
    Initialization from lists in different orders (but same k,v) give consistent results
    """
    list_of_lists = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    symbols = set()
    for i in range(10):
        random.shuffle( list_of_lists )
        symbols.add( alphabet.Alphabet( list_of_lists ).get_symbol(2) )
    assert len(symbols) == 1

def test_alphabet_many_to_one_compound_symbols_deterministic_list_init():
    """
    Initialization from lists in different orders (but same k,v) give consistent results
    (testing with compound symbols)
    """
    list_of_lists = [['A', 'ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c']
    symbols = set()
    for i in range(10):
        random.shuffle( list_of_lists )
        symbols.add( alphabet.Alphabet( list_of_lists ).get_symbol(2) )
    assert len(symbols) == 1


def test_alphabet_init_from_str():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→')
    assert alpha._utf_2_code == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'ß': 9, '→': 10, 'ϵ': 0, '?': 1, '↦': 11, '⇥': 12}
    assert alpha._code_2_utf == {12: '⇥', 11: '↦', 10: '→', 0: 'ϵ', 9: 'ß', 8: 'f', 7: 'e', 6: 'd', 5: 'c', 4: 'b', 3: 'a', 1: '?', 2: ' '} 


def test_alphabet_init_from_tsv( alphabet_one_to_one_tsv ):
    alpha = alphabet.Alphabet( str(alphabet_one_to_one_tsv) )
    assert alpha._utf_2_code == {' ': 2, ',': 3, 'A': 4, 'J': 11, 'R': 16, 'S': 17, 'V': 18, 'b': 21, 'c': 22, 'd': 23, 'o': 33, 'p': 34, 'r': 35, 'w': 40, 'y': 41, 'z': 42, '¬': 43, 'ü': 44, 'ϵ': 0, '?': 1, '↦': 45, '⇥': 46}
    assert alpha._code_2_utf == {46: '⇥', 45: '↦', 0: 'ϵ', 44: 'ü', 43: '¬', 42: 'z', 41: 'y', 40: 'w', 35: 'r', 34: 'p', 33: 'o', 23: 'd', 22: 'c', 21: 'b', 18: 'V', 17: 'S', 16: 'R', 11: 'J', 4: 'A', 1: '?', 3: ',', 2: ' '}
    

def test_alphabet_to_list():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    assert alphabet.Alphabet( list_of_lists ).to_list() == list_of_lists


def test_alphabet_to_list_minus_symbols():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']

    assert alphabet.Alphabet( list_of_lists ).to_list(exclude=['o','w']) == [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'y', 'z', 'ü']


def test_alphabet_len():

    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert len( alpha ) == 13


def test_alphabet_contains_symbol():
    """ 'in' operator """
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert 'a' in alpha
    assert 'z' not in alpha

def test_alphabet_contains_code():

    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert 1 in alpha
    assert 43 not in alpha

def test_alphabet_get_symbol():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert alpha.get_symbol( 8 ) == 'f'

def test_alphabet_get_code():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert alpha.get_code( 'f' ) == 8
    assert alpha.get_code( 'z' ) == 1

def test_alphabet_getitem():
    """ Subscript access """
    alpha = alphabet.Alphabet('ßa fdb\n\tce\t→') 
    assert alpha['f'] == 8
    assert alpha[8] == 'f'

def test_alphabet_eq():
    """ Testing for equality """
    alpha1= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    alpha2= alphabet.Alphabet('ßa fdb\n\tce\t→')
    alpha3= alphabet.Alphabet('ßa db\n\tce\t→')
    assert alpha1 == alpha2
    assert alpha1 != alpha3

def test_normalize_spaces():
    assert alphabet.Alphabet.normalize_spaces(' \t\n\u000Ba\u000C\u000Db\u0085c\u00A0\u2000\u2001d\u2008\u2009e') == 'a b c d e'

def test_encode_clean_sample():
    """ Most common case: no trailing spaces, nor extra spaces."""
    alpha= alphabet.Alphabet('ßa fdbce→') 
    assert alpha.encode('abc ß def') == [3, 4, 5, 2, 9, 2, 6, 7, 8]

def test_encode_normalized_spaces():
    """ Encoding should normalize spaces: strip trailing spaces, homogeneize, merge duplicates. """
    alpha= alphabet.Alphabet('ßa fdbce→') 
    assert alpha.encode('\tabc ß  def ') == [3, 4, 5, 2, 9, 2, 6, 7, 8]

def test_encode_missing_symbols():
    """Unknown symbols generate unknown char (and a warning)."""
    alpha= alphabet.Alphabet('ßa fdbce→') 
    assert alpha.encode('abc z def ') == [3, 4, 5, 2, 1, 2, 6, 7, 8]


def test_encode_one_hot():
    alpha= alphabet.Alphabet('ßa fdbce→') 
    assert alpha.encode_one_hot('abc ß def ').equal( torch.tensor(
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool ))

def test_decode():
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    # full length (default)
    assert alpha.decode( torch.tensor([3, 4, 5, 2, 10, 2, 6, 7, 8, 2], dtype=torch.int64 )) == 'abc → def '
    # explicit length
    assert alpha.decode( torch.tensor([3, 4, 5, 2, 10, 2, 6, 7, 8, 2], dtype=torch.int64 ), 5) == 'abc →'


def test_encode_batch_default():
    """ Batch with clean strings, padded by default """
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaaß' ]
    encoded = alpha.encode_batch( batch_str )

    assert encoded[0].equal( 
            torch.tensor( [[3, 4, 5, 2, 6, 7, 8, 0],
                           [9, 7, 5, 4, 5, 3, 3, 9]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,8], dtype=torch.int64 ))

def test_encode_batch_padded():
    """ Batch with clean strings, padded explicit """
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaaß' ]
    encoded = alpha.encode_batch( batch_str, padded=True )

    assert encoded[0].equal( 
            torch.tensor( [[3, 4, 5, 2, 6, 7, 8, 0],
                           [9, 7, 5, 4, 5, 3, 3, 9]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,8], dtype=torch.int64 ))


def test_encode_batch_unpadded():
    """ Batch with clean strings, unpadded explicit """
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaaß' ]
    encoded = alpha.encode_batch( batch_str, padded=False )

    assert encoded[0].equal( 
            torch.tensor( [3, 4, 5, 2, 6, 7, 8, 9, 7, 5, 4, 5, 3, 3, 9], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,8], dtype=torch.int64 ))


def test_decode_batch():

    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    samples, lengths = (torch.tensor( [[3, 4, 5, 2, 6, 7, 8, 2],
                            [9, 7, 5, 4, 5, 3, 2, 0]], dtype=torch.int64),
             torch.tensor( [8, 7]))
    assert alpha.decode_batch( samples, lengths ) == ["abc def ", "ßecbca "]
    assert alpha.decode_batch( samples, None ) == ["abc def ", "ßecbca ϵ"]



def test_decode_ctc():
    alpha = alphabet.Alphabet([' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'],
                              ['B', 'b'], ['C', 'c'], ['D', 'd'], ['E', 'e', 'é'], ['F', 'f'], ['G', 'g'],
                              ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'],
                              ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'],
                              ['S', 's'], ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'],
                              ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬', '…'])

    decoded = alpha.decode_ctc( np.array([20, 20, 20, 0, 0, 17, 17, 0, 24, 24, 24, 24, 0, 0,
                                24, 24, 0, 27, 27, 3, 0, 2, 35, 35, 27, 30, 30, 30, 
                                24, 16]) )
    assert decoded == 'hello, world'



