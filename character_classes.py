# classes for characters, for building alphabets

# Each name defines a broad category of symbols, for easy construction of alphabets:
# * all chars in the same sublist map to the same code
# * the char that comes first in a sublist is assumed to be the subset's representative

space_charset  = [ ' ' ]

latin_charset = [ 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ['A','Á','Â','Ã','Ä','Å','Æ','Ă','Ą','À', 'Ā'], ['a','á','â','ã','ä','å','æ','ă','ą','à','ā'], 
    'B', 'b',
    ['C','Ç','Ć','Ĉ','Ċ','Č'], ['c','ç','ć','ĉ','ċ','č'],
    ['D','Ð','Ď','Đ'], ['d','ð','ď','đ'],
    ['E','È','É','Ê','Ë','Ē','Ĕ','Ė','Ę','Ě'], ['e','è','é','ê','ë','ē','ĕ','ė','ę','ě'],
    'F', 'f',
    ['G','Ĝ','Ğ','Ġ','Ģ'], ['g','ĝ','ğ','ġ','ģ','ḡ'],
    ['H','Ĥ','Ħ'], ['h','ĥ','ħ'],
    ['I','Ì','Í','Î','Ï','Ĩ','Ī','Ĭ','Į','İ','Ĳ'], ['i','ì','í','î','ï','ĩ','ī','ĭ','į','ı','ĳ'],
    ['J','Ĵ', 'Ɉ'], ['j','ĵ','ɉ'],
    ['K','Ķ'], ['k','ķ','ĸ'],
    ['L','Ĺ','Ļ','Ľ','Ŀ','Ł','£'], ['l','ĺ','ļ','ľ','ŀ','ł'],
    'M', 'm',
    ['N','Ñ','Ń','Ņ','Ň','Ŋ'], ['n','ñ','ń','ņ','ň','ŉ','ŋ'],
    ['O','Ò','Ó','Ô','Õ','Ö','Ō','Ŏ','Ő','Œ'], ['o','ò','ó','ô','õ','ö','ō','ŏ','ő','œ','°'],
    ['P','Ꝑ','Ꝓ'], ['p','ꝑ','ꝓ'],
    ['Q','Ꝗ','Ꝙ'], ['q','ꝗ','ꝙ'],
    ['R','Ŕ','Ŗ','Ř','Ꝛ','Ꝝ','ꝶ'], ['r','ŕ','ŗ','ř','ˀ','ꝛ','ꝝ','ꝵ'],
    ['S','Ś','Ŝ','Ş','Š','ß'], ['s','ś','ŝ','ş','š','ſ','ẜ','\uf2f7'], # last is German schilling sign
    ['T','Ţ','Ť','Ŧ'], ['t','ţ','ť','ŧ','ꝷ'],
    ['U','Ù','Ú','Û','Ü','Ũ','Ū','Ŭ','Ů','Ű','Ų'], ['u','ù','ú','û','ü','ũ','ū','ŭ','ů','ű','ų'],
    ['V','Ꝟ'], ['v','ꝟ'],
    ['W','Ŵ','Ẅ'], ['w','ŵ','ẅ'],
    'X', 'x',
    ['Y','Ȳ','Ý','Ÿ','Ŷ','Ẏ'], ['y','ȳ','ý','ÿ','ŷ','ẏ','ẙ'],
    ['Z','Ź','Ż','Ž' 'Ẑ', 'Ƶ', 'Ʒ', 'Ȥ'], ['z','ź','ż','ž', 'ẑ', 'ƶ', 'ʒ', 'ȥ'],
]

punctuation_charset = [[ '.','✳'],  [',',';',':'], ['-','¬','—','='], ['¶','§' ]] 

# respectively: acdehimortuvx
superscript_charset = [['\u0363','\u0368','\u0369','\u0364','\u036a','\u0365','\u036b','\u0366','\u036c','\u036d','\u0367','\u036e','\u036f']]

# variety of diacritics (used in combination with other letters, not to
# be confused with accented pre-composed characters). Possible ways to handle them:
#
# * if the combination stands for itself:
#
#   + replace combination with its equivalent pre-composed character (not always possible, nor useful)
#   + discard the diacritic mark from the sample (giving essentially the same result as mapping the accented characters above on their base representative)
#
# * if the combination is an abbreviation = standing for (base letter + string)
#
#   + replace with an abbreviation representative (eg. '*'), if sticking with 'basic' transcription
#   + expand the abbreviation (not always possible, nor relevant)
# 
diacritic_charset = [ ["'","ʼ"] + [ chr(c) for c in range(0x300,0x316) ] + [ '\u0321', '\u0322', '\u0327', '\u0328', '\u1dd0'] ] # 0x0300-0x0316: above, 0x0321-0x1dd0: attached below

parenthesis_charset = [ ['(',')','[',']','{','}','/','\\','|'] ]

# all abbreviation signs that are conventional stand-in for a string
# of characters = more than a 'decorated' letter
abbreviation_charset = [ '*', 'ƺ','Ꝯ','ꝯ','Ꝫ','ꝫ','ȝ','₰','ꝰ','ꝭ','&','₎','Ꜿ','כּ' ]

hebrew_charset = [ chr(c) for c in range(0x0591,0x05f5) ]

greek_charset = [ chr(c) for c in range(0x03b1,0x03e1) ] + [ chr(c) for c in range(0x0391,0x03b0) ]

all_charsets = space_charset \
            + latin_charset \
            + punctuation_charset \
            + superscript_charset \
            + diacritic_charset \
            + parenthesis_charset \
            + abbreviation_charset \
            + hebrew_charset \
            + greek_charset 

# check for duplicates:
from libs import list_utils as lu
import itertools
from collections import Counter

all_chars = lu.flatten( all_charsets )
if len(all_chars) != len(set(all_chars)):
    duplicates = list( itertools.filterfalse( lambda x: x[1]==1, Counter(sorted(all_chars)).items()))
    raise ValueError(f"Duplicate characters in the input sublists: {duplicates}")
