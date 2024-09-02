import pytest
import sys
import torch
from torch import Tensor
from pathlib import Path

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import inferlib

@pytest.fixture(scope="module")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


def test_alphabet_from_string():
    """
    Unique characters, Unicode legit, sorted, nullspace, other space chars ignored
    """
    # space char
    alpha = inferlib.Alphabet.from_string('ßafdbce→')
    assert alpha[ max( alpha.keys() ) ] == '\u2205'
    # unique symbols, sorted
    assert inferlib.Alphabet.from_string('ßaafdbce →e') == { 1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→', 10:'∅'}
    # space chars ignored
    assert inferlib.Alphabet.from_string('ßaf \u2009db\n\tce\t→') == { 1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→', 10:'∅'}


def test_alphabet_len():

    alpha = inferlib.Alphabet('ßaf db\n\tce\t→') 
    assert len( alpha ) == 10


def test_alphabet_contains_symbol():
    """ 'in' operator """
    alpha = inferlib.Alphabet('ßaf db\n\tce\t→') 
    assert 'a' in alpha
    assert 'z' not in alpha

def test_alphabet_contains_code():

    alpha = inferlib.Alphabet('ßaf db\n\tce\t→') 
    assert 1 in alpha
    assert 43 not in alpha

def test_alphabet_get_symbol():
    alpha = inferlib.Alphabet('ßaf db\n\tce\t→') 
    assert alpha.get_symbol( 8 ) == 'ß'

def test_alphabet_get_code():
    alpha = inferlib.Alphabet('ßafdb\n\tce\t→') 
    assert alpha.get_code( 'ß' ) == 7
    assert alpha.get_code( 'z' ) == alpha.maxcode

def test_alphabet_getitem():
    """ Subscript access """
    alpha = inferlib.Alphabet('ßa fdb\n\tce\t→') 
    print(alpha)
    assert alpha['ß'] == 8
    assert alpha[8] == 'ß'

def test_alphabet_eq():
    """ Testing for equality """
    alpha1= inferlib.Alphabet('ßa fdb\n\tce\t→') 
    alpha2= inferlib.Alphabet('ßa fdb\n\tce\t→')
    alpha3= inferlib.Alphabet('ßa db\n\tce\t→')
    assert alpha1 == alpha2
    assert alpha1 != alpha3


def test_encode_clean_sample():
    alpha= inferlib.Alphabet('ßa fdbce→') 
    encoded = alpha.encode('abc ß def ')
    assert encoded.equal( torch.tensor([2, 3, 4, 1, 8, 1, 5, 6, 7, 1], dtype=torch.int64))

def test_encode_missing_symbols():
    """Unknown symbols generate a warning."""
    alpha= inferlib.Alphabet('ßa fdbce→') 
    with pytest.warns(UserWarning):
        encoded = alpha.encode('abc z def ')
        assert encoded.equal( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64))

def test_encode_illegal_symbols():
    """Illegal symbols raise an exception."""
    alpha= inferlib.Alphabet('ßa fdbce→') 
    with pytest.raises(ValueError):
        encoded = alpha.encode('abc\n z def ')
    with pytest.raises(ValueError):
        encoded = alpha.encode('abc\t z def ')

def test_encode_one_hot():
    alpha= inferlib.Alphabet('ßa fdbce→') 
    assert alpha.encode_one_hot('abc ß def ').equal( torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool ))

def test_decode():
    alpha= inferlib.Alphabet('ßa fdb\n\tce\t→') 
    # full length (default)
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 )) == 'abc → def '
    # explicit length
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 ), 5) == 'abc →'

def test_encode_batch_1():
    """ Batch with clean strings """
    alpha= inferlib.Alphabet('ßa fdb\n\tce\t→') 
    batch_str = [ 'abc def ', 'ßecbca ' ]
    encoded = alpha.encode_batch( batch_str )

    assert encoded[0].equal( 
            torch.tensor( [[2, 3, 4, 1, 5, 6, 7, 1],
                           [8, 6, 4, 3, 4, 2, 1, 0]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([8,7], dtype=torch.int64 ))

def test_decode_batch():

    alpha= inferlib.Alphabet('ßa fdb\n\tce\t→') 
    samples, lengths = (torch.tensor( [[2, 3, 4, 1, 5, 6, 7, 1],
                            [8, 6, 4, 3, 4, 2, 1, 0]], dtype=torch.int64),
             torch.tensor( [8, 7]))
    assert alpha.decode_batch( samples, lengths ) == ["abc def ", "ßecbca "]
    assert alpha.decode_batch( samples, None ) == ["abc def ", "ßecbca ."]


def test_dummy( data_path):
    """
    A dummy test, as a sanity check for the test framework.
    """
    print(data_path)
    assert inferlib.dummy() == True

