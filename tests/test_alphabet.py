import pytest
import sys
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
import random

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from libs import alphabet

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def gt_transcription_samples( data_path ):
    return [ str(data_path.joinpath(t)) for t in ('transcription_sample_1.gt.txt', 'transcription_sample_2.gt.txt') ]

def test_alphabet_dict_from_string():
    """
    Raw dictionary reflects the given string: no less, no more; no virtual chars (null, ⇥, ...)
    """
    # unique symbols, sorted
    assert alphabet.Alphabet._dict_from_string('ßaafdbce →e') == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 's': 9, '→': 10, }
    # space chars ignored
    assert alphabet.Alphabet._dict_from_string('ßaf \u2009db\n\tce\t→') == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 's': 9, '→': 10}


def test_dict_from_list_one_to_one():
    input_list = ['A', 'a', 'J', 'b', 'ö', 'o', 'O', 'ü', 'U', 'w', 'y', 'z', 'd', 'D']
    alpha = alphabet.Alphabet._dict_from_list( input_list )
    assert alpha == {'A': 2, 'D': 3, 'J': 4, 'O': 5, 'U': 6, 'a': 7, 'b': 8, 'd': 9, 'o': 10, 'w': 11, 'y': 12, 'z': 13, 'ö': 14, 'ü': 15}

def test_dict_from_list_compound_symbols_one_to_one():
    input_list = ['A', 'ae', 'J', 'ü', 'eu', 'w', 'y', 'z', '...', 'D']
    alpha = alphabet.Alphabet._dict_from_list( input_list )
    assert alpha == {'...': 2, 'A': 3, 'D': 4, 'J': 5, 'ae': 6, 'eu': 7, 'w': 8, 'y': 9, 'z': 10, 'ü': 11} 

def test_dict_from_list_many_to_one():
    input_list = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    alpha = alphabet.Alphabet._dict_from_list( input_list )
    assert alpha == {'A': 2, 'a': 2, 'D': 3, 'd': 3, 'J': 4, 'O': 5, 'o': 5, 'ö': 5, 'U': 6, 'b': 7, 'w': 8, 'y': 9, 'z': 10, 'ü': 11}
                    
def test_dict_from_list_compound_symbols_many_to_one():
    input_list = ['A', 'ae', 'J', ['ü','U'], 'eu', 'w', 'y', 'z', ['...', '.'], 'D']
    alpha = alphabet.Alphabet._dict_from_list( input_list )
    assert set(alpha.keys()) == set([ 'A', 'ae', 'J','ü','U','eu', 'w', 'y', 'z', '...', '.', 'D'])
    assert alpha == {'A': 3, 'D': 4, 'J': 5, 'ae': 7, 'eu': 8, 'w': 9, 'y': 10, 'z': 11, '.': 2, '...': 2, 'U': 6, 'ü': 6}

def test_dict_from_list_realistic():
    """ Passing a list with virtual symbols (EoS, SoS) yields a correct mapping 
    """
    input_list = [ ' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'], ['B', 'b'], ['C', 'c'], [    'D', 'd'], ['E', 'e', 'é'], '⇥', ['F', 'f'], ['G', 'g'], ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'], ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'], ['S', 's'], '↦', ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'], ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬','…' ]
    alpha = alphabet.Alphabet._dict_from_list( input_list )
    assert alpha == {' ': 2, ',': 3, '-': 4, '.': 5, '1': 6, '2': 7, '4': 8, '5': 9, '6': 10, ':': 11, ';': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, '¬': 39, 'Ö': 27, 'ä': 13, 'é': 17, 'ö': 27, 'ü': 33, 'ÿ': 37, 'ř': 30, 'Ž': 38, '…': 40} 

def test_dict_from_list_duplicated_symbol():
    """ Sublists should be disjoint """
    valid_list = [ ' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'ae', 'ä'], ['B', 'b', 'a']]
    assert alphabet.Alphabet._dict_from_list( valid_list ) == {' ': 2, ',': 3, '-': 4, '.': 5, '1': 6, '2': 7, '4': 8, '5': 9, '6': 10, ':': 11, ';': 12, 'A': 13, 'B': 14, 'a': 14, 'ae': 13, 'b': 14, 'ä': 13}
    # 'a' is in two sublists
    invalid_list = [ ' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ae', 'ä'], ['B', 'b', 'a']]
    with pytest.raises( ValueError ) as e:
        alphabet.Alphabet._dict_from_list( invalid_list )
    

def test_alphabet_many_to_one_deterministic_dict_init():
    """
    Initialization from dictionaries in different orders (but same mapping) gives consistent results
    """
    alpha_list = [ 'A', 'D', 'J', 'O', 'U', 'a', 'b', 'd', 'o', 'w', 'y', 'z', 'ö', 'ü' ]
    symbols = set()
    for i in range(10):
        random.shuffle( alpha_list )
        symbols.add( alphabet.Alphabet( alpha_list ).get_symbol(2) )
    assert len(symbols) == 1

def test_alphabet_many_to_one_consistent_decoding():
    """
    With list-of-lists input, every code assigned to a sublist should map to the first character
    in the sublist.
    """
    alpha = alphabet.Alphabet(['0', '1', '2', ['A', 'À', 'Á', 'Â'], 'B'])
    assert alpha.get_symbol( alpha.get_code('Â')) == 'A'
    assert alpha.get_symbol( alpha.get_code('Á')) == 'A'
    alpha = alphabet.Alphabet(['0', '1', '2', ['C', 'À', 'A', 'Á'], 'B'])
    assert alpha.get_symbol( alpha.get_code('Á')) == 'A'
    assert alpha.get_symbol( alpha.get_code('A')) == 'A'

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
    assert alpha._utf_2_code == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 's': 9, '→': 10, 'ϵ': 0, '?': 1, '↦': 11, '⇥': 12}
    assert alpha._code_2_utf == {12: '⇥', 11: '↦', 10: '→', 0: 'ϵ', 9: 's', 8: 'f', 7: 'e', 6: 'd', 5: 'c', 4: 'b', 3: 'a', 1: '?', 2: ' '} 



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

def test_normalize_string_spaces():
    assert alphabet.Alphabet.normalize_string(' \t\n\u000Ba\u000C\u000Db\u0085c\u00A0\u2000\u2001d\u2008\u2009e') == 'a b c d e'

def test_encode_clean_sample():
    """ Most common case: no trailing spaces, nor extra spaces. """
    alpha= alphabet.Alphabet('ta fdbce→') 
    assert torch.equal( alpha.encode('abc t def'), torch.tensor([3, 4, 5, 2, 9, 2, 6, 7, 8]))

def test_encode_normalized_spaces():
    """ Encoding should normalize spaces: strip trailing spaces, homogeneize, merge duplicates. """
    alpha= alphabet.Alphabet('ta fdbce→') 
    assert torch.equal( alpha.encode('\tabc t  def '), torch.tensor([3, 4, 5, 2, 9, 2, 6, 7, 8]))

def test_encode_normalize_composed_characters():
    """Encoding should normalized 
    + composed characters into their pre-composed equivalent
    + ligatured characters into their 2-char equivalent
    """
    alpha = alphabet.Alphabet('atÅ fdbceso')
    assert torch.equal( alpha.encode('abc t Å def'), torch.tensor( [ 3,  4,  5,  2, 11,  2, 12,  2,  6,  7,  8] ))
    assert torch.equal( alpha.encode('abc ﬆ def'), torch.tensor( [ 3,  4,  5,  2, 10, 11,  2,  6,  7,  8] ))
    assert torch.equal( alpha.encode('abc œdef'), torch.tensor( [3, 4, 5, 2, 9, 7, 6, 7, 8] ))
    assert torch.equal( alpha.encode('abc ædef'), torch.tensor( [3, 4, 5, 2, 3, 7, 6, 7, 8] ))
    assert torch.equal( alpha.encode('abc ßdef'), torch.tensor( [ 3,  4,  5,  2, 10, 10,  6,  7,  8] ))


def test_encode_missing_symbols():
    """Unknown symbols generate unknown char (and a warning)."""
    alpha= alphabet.Alphabet('a fdbce') 
    assert torch.equal(  alpha.encode('abc z def '), torch.tensor( [3, 4, 5, 2, 1, 2, 6, 7, 8]))


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
    alpha= alphabet.Alphabet('sa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaff' ]
    encoded = alpha.encode_batch( batch_str )

    assert encoded[0].equal( 
            torch.tensor( [[3, 4, 5, 2, 6, 7, 8, 0, 0],
                           [9, 9, 7, 5, 4, 5, 3, 8, 8]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,9], dtype=torch.int64 ))

def test_encode_batch_padded():
    """ Batch with clean strings, padded explicit """
    alpha= alphabet.Alphabet('sa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaff' ]
    encoded = alpha.encode_batch( batch_str, padded=True )

    assert encoded[0].equal( 
            torch.tensor( [[3, 4, 5, 2, 6, 7, 8, 0, 0],
                           [9, 9, 7, 5, 4, 5, 3, 8, 8]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,9], dtype=torch.int64 ))


def test_encode_batch_unpadded():
    """ Batch with clean strings, unpadded explicit """
    alpha= alphabet.Alphabet('sa fdb\n\tce\t→') 
    batch_str = [ 'abc def', 'ßecbcaff' ]
    encoded = alpha.encode_batch( batch_str, padded=False )

    assert encoded[0].equal( 
            torch.tensor( [3, 4, 5, 2, 6, 7, 8, 9, 9, 7, 5, 4, 5, 3, 8, 8], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([7,9], dtype=torch.int64 ))


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
    assert decoded == 'HELLO, WORLD'



