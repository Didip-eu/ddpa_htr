from typing import List, Union, Tuple
import itertools
import pytest
import Levenshtein


def cer_wer_ler( predicted_mesg: List[str], target_mesg: List[str], word_separator: Union[str,int]=' ') -> Tuple[float, float]:
    """ Compute CER, WER, and LER for a batch.

    The sequences to be compared can be either plain strings, or encoded (i.e. integer-labeled)
    sequences.

    Args:
        predicted_mesg (List[str]): list of predicted strings.
        target_mesg (List[str]): list of target strings.
        word_separator (Union[str,int]): value of the separator (default: ' ')

    Returns:
        Tuple[float, float]: a 3-tuple with CER, WER, and LER (line error rate).
    """
    if len(predicted_mesg) != len(target_mesg):
        raise ValueError("Input lists must have the same lengths!")
    #if type(predicted_mesg[0][0]) != type(word_separator):
    #    raise ValueError('Mismatch between sequence type ({}) and separator type ({})'.format( type(predicted_mesg[0]), type(word_separator)))
    batch_cers = [ edit_dist(pred, target)/len(target) for (pred, target) in zip (predicted_mesg, target_mesg ) ]
    batch_cer = sum(batch_cers) / len(target_mesg)
    line_error = len( [ err for err in batch_cers if err > 0 ] ) / len(target_mesg)

    # WER
    pred_split, target_split = [ [ split_generic(seq, word_separator) for seq in mesg ] for mesg in (predicted_mesg, target_mesg) ] 
    pairs = list(zip( pred_split, target_split ))

    batch_wer = 0.0
    for p in pairs:
        make_hashable = lambda x: tuple(x) if type(x) is list else x 
        enc = { make_hashable(w):v for (v,w) in enumerate( p[0] + p[1] ) } 
        enc_pred, enc_target = [ enc[ make_hashable(w)] for w in p[0] ], [enc[make_hashable(w)] for w in p[1] ]
        batch_wer += edit_distance( enc_pred, enc_target ) / len( enc_target)
    batch_wer /= len(pairs)

    return (batch_cer, batch_wer, line_error)

def cer_wer_ler_with_masks( predicted_mesg: List[str], target_mesg: List[str], masks: List[List[Tuple[int,int]]]=[], word_separator: Union[str,int]=' ') -> Tuple[float, float]:
    """ Compute CER, WER, and LER for a batch.

    The sequences to be compared can be either plain strings, or encoded (i.e. integer-labeled)
    sequences.

    Args:
        predicted_mesg (List[str]): list of predicted strings.
        target_mesg (List[str]): list of target strings.
        masks (List[List[Tuple[int,int]]]): a list of masks for each message, i.e. array of 
            pairs (<offset>,<length>)
        word_separator (Union[str,int]): value of the separator (default: ' ')

    Returns:
        Tuple[float, float]: a 4-tuple with 
            + CER
            + WER
            + LER (line error rate)
            + MER (CER, not considering masked bits)
            + contribution of masked bits to the error rate
    """
    if len(predicted_mesg) != len(target_mesg):
        raise ValueError("Input lists must have the same lengths!")

    batch_char_error_counts = [ edit_dist(pred, target) for (pred, target) in zip (predicted_mesg, target_mesg ) ] 
    batch_cers = [ bcec/len(target) for (bcec, target) in zip (batch_char_error_counts, target_mesg ) ]
    batch_cer = sum(batch_cers) / len(target_mesg)
    line_error = len( [ err for err in batch_cers if err > 0 ] ) / len(target_mesg)

    masked_char_error_counts = [ edit_dist_with_mask(pred, target, mask) for (pred, target, mask) in zip (predicted_mesg, target_mesg, masks ) ]
    masked_cers = [ mcec/len(target) for (mcec,target) in zip (masked_char_error_counts, target_mesg) ]
    masked_cer = sum(masked_cers) / len(target_mesg)
    # difference = those errors that are due to masked parts
    masked_contribution = (sum(batch_char_error_counts) - sum(masked_char_error_counts))/sum(batch_char_error_counts)

    # WER
    pred_split, target_split = [ [ split_generic(seq, word_separator) for seq in mesg ] for mesg in (predicted_mesg, target_mesg) ] 
    pairs = list(zip( pred_split, target_split ))

    batch_wer = 0.0
    for p in pairs:
        make_hashable = lambda x: tuple(x) if type(x) is list else x 
        enc = { make_hashable(w):v for (v,w) in enumerate( p[0] + p[1] ) } 
        enc_pred, enc_target = [ enc[ make_hashable(w)] for w in p[0] ], [enc[make_hashable(w)] for w in p[1] ]
        batch_wer += edit_dist( enc_pred, enc_target ) / len( enc_target)
    batch_wer /= len(pairs)

    return (batch_cer, batch_wer, line_error, masked_cer, masked_contribution)


def split_generic( seq: Union[str,list], sep: Union[str,int] ) -> List[list]:
    """ Split a sequence into subsequences, along a separator value.

    Args:
        seq (Union[str,list]): a list of strings or integers
        sep (Union[str,int): a separator value.

    Returns:
        List[list]: a list of subsequences.
    """
#    def split_rec( sq, sp, accum ):
#        """ Split a sequence, functional style,
#        with immutable sequences.
#        """
#        if sq == []:
#            return accum
#        if sq[0] == sp:
#            return (accum + split_rec( sq[1:], sp, tuple() ))
#        accum = accum[:-1] + ( accum[-1] + (sq[0], ),) if accum else ((sq[0],),)
#        return split_rec(sq[1:], sp, accum)

    def split_rec( seq, sep, accum ):
        """ Split a sequence, functional style.
        """
        if seq == []:
            return accum
        if seq[0] == sep:
            return accum + split_rec( seq[1:], sep, [] )
        if accum:
            accum[-1].append( seq[0] )
        else:
            accum.append( [ seq[0] ])
        return split_rec(seq[1:], sep, accum)

    if type(seq) is str and type(sep) is str:
        return seq.split(sep)
    elif type(seq) is list:
        return split_rec( seq, sep, [] )
    return seq


def edit_dist( x, y ):
    """
    Compute edit distance, recursively.

    Args:
        x (str): start string
        y (str): target string

    Returns:
        int: the Levenshtein edit distance, where each insertion, deletion and substitution 
            contributes 1 to the distance.
    """
    if len(x)==0 or len(y)==0:
        return abs(len(x)-len(y))

    memo = [ [-1]*(len(y)+1) for r in range(len(x)+1) ]
    def dist_rec(i, j):
        if memo[i][j]>0:
            return memo[i][j]
        if i<0 or j<0:
            return abs(i-j)
        if x[i] == y[j]:
            d = dist_rec(i-1, j-1)
        else:
            d = min(dist_rec(i-1, j), dist_rec(i, j-1), dist_rec(i-1, j-1)) + 1 
        memo[i][j]=d
        return d

    return dist_rec(len(x)-1, len(y)-1)

def edit_dist_with_mask(x, y, masks=[]):
    """
    Compute edit distance while masking one or more slice of the target string.
    The masked parts do not contribute to the resulting distance.

    Args:
        x (str): start string
        y (str): target string
        masks (List[Tuple[int,int]]): a list of tuples (offset, length)

    Returns:
        int: the Levenshtein edit distance, where each insertion, deletion and substitution 
            contributes for 1 to the distance, if it does not involve the masked parts.
    """
    jays = list( itertools.chain.from_iterable([ range(m[0], m[0]+m[1]) for m in masks ] )) if len(masks)>0 else []

    memo = [ [0]*(len(y)+1) for r in range(len(x)+1) ]
    for i in range(len(x)+1):
        memo[i][0]=i
        for j in range(1,len(y)+1):
            if i==0:
                memo[0][j]=memo[0][j-1] + (0 if j-1 in jays else 1)
            elif x[i-1]==y[j-1]:
                memo[i][j] = memo[i-1][j-1]
            else:
                memo[i][j]=min( memo[i-1][j-1], memo[i-1][j], memo[i][j-1]) + (0 if j-1 in jays else 1)
    #print( ' '*7 +'  '.join(list(y)) + '\n' + '\n'.join( [ f'{x[r-1]}  {row}' for (r,row) in enumerate(memo) ]))
    return memo[-1][-1]


def align_dist(x, y):
    """ Compute an edit-distance-based alignment between strings x and y.

    Args:
        x (str): start string
        y (str): target string

    Returns:
        Tuple[list,list]: a pair of lists storing the aligned indices in x and y, respectively.
    """
    if not x or not y:
        return ()
    memo = [ [0]*(len(y)+1) for r in range(len(x)+1) ]
    for i in range(len(x)+1):
        memo[i][0]=i                # if Y_j=ϵ, distance = i
        for j in range(1,len(y)+1):
            if i==0:
                memo[0][j]=j        # X_i=ϵ, dist = j
            elif x[i-1]==y[j-1]:    
                memo[i][j] = memo[i-1][j-1]
            else:
                memo[i][j]=min( memo[i-1][j-1], memo[i-1][j], memo[i][j-1]) + 1
    
    # ugly, but efficient
    i,j=len(x), len(y)
    pairs = []
    while i>0 and j>0:
        if x[i-1]==y[j-1]:
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif memo[i-1][j-1] <= memo[i-1][j] and memo[i-1][j-1] <= memo[i][j-1]:
            i -= 1
            j -= 1
        elif memo[i-1][j] <= memo[i][j] and memo[i-1][j] <= memo[i][j-1]:
            i -= 1
        elif memo[i][j-1] <= memo[i][j] and memo[i][j-1] <= memo[i-1][j]:
            j -= 1
    
    alignments = tuple(zip(*reversed(pairs))) # unzipping :)
 
    return alignments

def align_lcs(x, y):
    """ Compute a LCS-based alignment between strings x and y.

    Args:
        x (str): start string
        y (str): target string

    Returns:
        Tuple[list,list]: a pair of lists storing the aligned indices in x and y, respectively.
    """
    if not x or not y:
        return ()
    memo = [ [0]*(len(y)+1) for r in range(len(x)+1) ]
    for i in range(len(x)+1):
        memo[i][0]=0
        for j in range(1,len(y)+1):
            if i==0:
                memo[0][j]=0
            elif x[i-1]==y[j-1]:
                memo[i][j] = memo[i-1][j-1]+1
            else:
                memo[i][j]=max( memo[i-1][j], memo[i][j-1]) 
    
    # ugly, but efficient
    i,j=len(x), len(y)
    pairs = []
    while i>0 and j>0:
        if x[i-1]==y[j-1]:
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif memo[i-1][j] >= memo[i][j] and memo[i-1][j] >= memo[i][j-1]:
            i -= 1
        elif memo[i][j-1] >= memo[i][j] and memo[i][j-1] >= memo[i-1][j]:
            j -= 1
    
    alignments = tuple(zip(*reversed(pairs))) # unzipping :)
 
    return alignments









@pytest.mark.parametrize("x, y, alignment", [ ('', '', ()),
                                             ('La route tourne', '', ()),
                                             ('', 'Le roi court', ()),
                                             ('La route tourne', 'Le roi court', ((0,2,3,4,8,10,11,12),(0,2,3,4,6,8,9,10))),
                                             ('Le roi court', 'La route tourne', ((0,2,3,4,6,8,9,10), (0,2,3,4,8,10,11,12))),
                                         ])
def test_alignment(x, y, alignment):
    assert align(x, y) == alignment

@pytest.mark.parametrize("x, y, distance", [ ('', '', 0),
                                             ('La route tourne', '', 15),
                                             ('', 'Le roi court', 12),
                                             ('La route tourne', 'Le roi court', 7),
                                             ('Le roi court', 'La route tourne', 7),
                                         ])
def test_edit_dist_strings(x, y, distance):
    assert edit_dist(x, y) == distance
    assert edit_dist(x, y) == Levenshtein.distance( x, y )

@pytest.mark.parametrize("x, y, distance", [ ('', '', 0),
                                             ('La route tourne', '', 15),
                                             ('', 'Le roi court', 12),
                                             ('La route tourne', 'Le roi court', 7),
                                             ('Le roi court', 'La route tourne', 7),
                                         ])
def test_edit_dist_codes(x, y, distance):
    x_codes, y_codes = [ord(c) for c in x ], [ ord(c) for c in y ]
    assert edit_dist( x_codes, y_codes ) == distance
    assert edit_dist(x, y) == Levenshtein.distance( x_codes, y_codes )

@pytest.mark.parametrize("x, y, mask, distance", [ ('', '', [], 0),
                                                   ('La route tourne', '', [], 15),
                                                   ('', 'Le roi court', [], 12),
                                                   ('La route tourne', 'Le roi court', [], 7),
                                                   ('Le roi court', 'La route tourne', [], 7),
                                                   ('', 'La route tourne', [(3,3)], 12),
                                                   ('', 'La route tourne', [(1,2), (5,3)], 10),
                                                   ('La route tourne', 'Le roi court', [(3,5)], 3),
                                                   ('La route tourne', 'Le roi court', [(0,2)], 6),
                                                   ('La route tourne', 'Le roi court', [(0,2),(3,5),(8,3)], 1), ])
def test_edit_dist_with_mask(x, y, mask, distance):
    assert edit_dist_with_mask(x, y, mask) == distance


