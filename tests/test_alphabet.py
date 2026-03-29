import pytest
import sys
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
import random

from pylelemmatize import LemmatizerBMP

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from libs import alphabet


mapper_mufi_ascii_lc = LemmatizerBMP.from_alphabet_mapping( ll.charsets.mufibmp+' ', ll.charsets.ascii+' ')
mapper_iso_ascii = LemmatizerBMP.from_alphabet_mapping( ll.charsets.iso_8859_2+' ', ll.charsets.ascii+' ')

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def gt_transcription_samples( data_path ):
    return [ str(data_path.joinpath(t)) for t in ('transcription_sample_1.gt.txt', 'transcription_sample_2.gt.txt') ]



def test_alphabet_init():
    
    alpha = alphabet.Alphabet( mapper=mapper_mufi_ascii_lc )
    assert alpha.null_value == 0
    assert alpha.
    assert alpha._utf_2_code == {' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 's': 9, '→': 10, 'ϵ': 0, '?': 1, '↦': 11, '⇥': 12}
    assert alpha._code_2_utf == {12: '⇥', 11: '↦', 10: '→', 0: 'ϵ', 9: 's', 8: 'f', 7: 'e', 6: 'd', 5: 'c', 4: 'b', 3: 'a', 1: '?', 2: ' '} 



def test_alphabet_to_list():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    assert alphabet.Alphabet( list_of_lists ).to_list() == list_of_lists


def test_alphabet_len():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert len( alpha ) == 13


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

def test_encode_missing_symbols():
    """Unknown symbols generate unknown char (and a warning)."""
    alpha= alphabet.Alphabet('a fdbce') 
    assert torch.equal(  alpha.encode('abc z def '), torch.tensor( [3, 4, 5, 2, 1, 2, 6, 7, 8]))



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



