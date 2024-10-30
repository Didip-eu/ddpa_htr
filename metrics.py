from typing import List, Union, Tuple
import Levenshtein


def cer_wer_ler( predicted_mesg: List[str], target_mesg: List[str], word_separator: Union[str,int]=' ') -> Tuple[float, float]:
    """ Compute CER, WER, and LER for a batch.

    The sequences to be compared can be either plain strings, or encoded (i.e. integer-labeled)
    sequences.

    :param predicted_mesg: list of predicted strings.
    :type predicted_mesg: List[str]
    :param target_mesg: list of target strings.
    :type target_mesg: List[str]
    :param word_separator: value of the separator (default: ' ')
    :type word_separator: Union[str,int]

    :returns: a 3-tuple with CER, WER, and LER (line error rate).
    :rtype: Tuple(float, float)
    """
    if len(predicted_mesg) != len(target_mesg):
        raise ValueError("Input lists must have the same lengths!")
    #if type(predicted_mesg[0][0]) != type(word_separator):
    #    raise ValueError('Mismatch between sequence type ({}) and separator type ({})'.format( type(predicted_mesg[0]), type(word_separator)))
    char_errors = [ Levenshtein.distance(pred, target)/len(target) for (pred, target) in zip (predicted_mesg, target_mesg ) ]
    cer = sum(char_errors) / len(target_mesg)
    line_error = len( [ err for err in char_errors if err > 0 ] ) / len(char_errors)

    # WER
    pred_split, target_split = [ [ split_generic(seq, word_separator) for seq in mesg ] for mesg in (predicted_mesg, target_mesg) ] 
    pairs = list(zip( pred_split, target_split ))

    wer = 0.0
    for p in pairs:
        make_hashable = lambda x: tuple(x) if type(x) is list else x 
        enc = { make_hashable(w):v for (v,w) in enumerate( p[0] + p[1] ) } 
        enc_pred, enc_target = [ enc[ make_hashable(w)] for w in p[0] ], [enc[make_hashable(w)] for w in p[1] ]
        wer += Levenshtein.distance( enc_pred, enc_target ) / len( enc_target)
    wer /= len(pairs)

    return (cer, wer, line_error)


def split_generic( seq: Union[str,list], sep: Union[str,int] ) -> List[list]:
    """ Split a sequence into subsequences, along a separator value.

    :param seq: a list of strings or integers
    :type seq: Union[str,list]
    :param sep: a separator value.
    :type sep: Union[str, int]

    :returns: a list of subsequences.
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

