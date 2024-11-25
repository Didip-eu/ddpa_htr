# classes for characters, for building alphabets

# Each name defines a broad category of symbols, for easy construction of alphabets:
# * all chars in the same sublist map to the same code
# * the char that comes first in a sublist is assumed to be the subset's representative

space_charset  = [ ' ' ]

latin_charset = [ 
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
    ['A','Á','Â','Ã','Ä','Å','Æ','Ă','Ą','À'], ['a','á','â','ã','ä','å','ā','ă','ą','à','æ'],
    'B', 'b',
    ['C','Ç','Ć','Ĉ','Ċ','Č'], ['c','ç','ć','ĉ','ċ','č'],
    ['D','Ð','Ď','Đ'], ['d','ð','ď','đ'],
    ['E','È','É','Ê','Ë','Ē','Ĕ','Ė','Ę','Ě'], ['e','è','é','ê','ë','ē','ĕ','ė','ę','ě'],
    'F', 'f',
    ['G','Ĝ','Ğ','Ġ','Ģ'], ['g','ĝ','ğ','ġ','ģ','ḡ'],
    ['H','Ĥ','Ħ'], ['h','ĥ','ħ'],
    ['I','Ì','Í','Î','Ï','Ĩ','Ī','Ĭ','Į','İ','Ĳ'], ['i','ì','í','î','ï','ĩ','ī','ĭ','į','ı','ĳ'],
    ['J','Ĵ'], ['j','ĵ','ɉ'],
    ['K','Ķ'], ['k','ķ','ĸ'],
    ['L','Ĺ','Ļ','Ľ','Ŀ','Ł','£'], ['l','ĺ','ļ','ľ','ŀ','ł'],
    'M', 'm',
    ['N','Ñ','Ń','Ņ','Ň','Ŋ'], ['n','ñ','ń','ņ','ň','ŉ','ŋ'],
    ['O','Ò','Ó','Ô','Õ','Ö','Ō','Ŏ','Ő','Œ'], ['o','ò','ó','ô','õ','ö','ō','ŏ','ő','œ','°'],
    'P', ['p','ꝑ','ꝓ'],
    'Q', ['q','ꝗ','ꝙ'],
    ['R','Ŕ','Ŗ','Ř'], ['r','ŕ','ŗ','ř','ˀ'],
    ['S','Ś','Ŝ','Ş','Š','ß'], ['s','ś','ŝ','ş','š'],
    ['T','Ţ','Ť','Ŧ'], ['t','ţ','ť','ŧ','ꝷ'],
    ['U','Ù','Ú','Û','Ü','Ũ','Ū','Ŭ','Ů','Ű','Ų'], ['u','ù','ú','û','ü','ũ','ū','ŭ','ů','ű','ų'],
    'V', ['v','ꝟ'],
    ['W','Ŵ'], ['w','ŵ'],
    'X', 'x',
    ['Y','Ŷ','Ÿ'], ['y','ý','ÿ','ŷ'],
    ['Z','Ź','Ż','Ž'], ['z','ź','ż','ž'],
]

punctuation_charset = [[ '.','✳'],  [',',';',':'], ['-','¬','—','='], ['¶','§' ]] 

# respectively: acdehimortuvx
subscript_charset = [['\u0363','\u0368','\u0369','\u0364','\u036a','\u0365','\u036b','\u0366','\u036c','\u036d','\u0367','\u036e','\u036f']]

# variety of diacritics (to be used in combination with other letters, not to
# be confused with accented 1-byte symbols)
diacritic_charset = [ ["'","ʼ"] + [ chr(c) for c in range(0x300,0x316) ] ]

parenthesis_charset = [ ['(',')','[',']','/','\\','|'] ]

abbreviation_charset = [ 'ƺ','Ꝯ','ꝯ','ꝫ','ȝ','ꝝ','₰','ꝛ','ꝰ','ꝭ','&','₎','כּ' ]

hebrew_charset = [ chr(c) for c in range(0x0591,0x05f5) ]

greek_charset = [ chr(c) for c in range(0x03b1,0x03e1) ] + [ chr(c) for c in range(0x0391,0x03b0) ]

charsets = [ space_charset, latin_charset, punctuation_charset, subscript_charset, diacritic_charset, parenthesis_charset, abbreviation_charset, hebrew_charset, greek_charset ]


