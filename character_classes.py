# classes for characters, for building alphabets

# each line define a tuple (<class representant>, <charset>) 
# * all chars in the same set (string) map to the same code
# * the char that comes first in a set is assumed to be the subset's representative

space_charset  = [ ' ' ]

latin_charset = [ 
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
    'AÁÂÃÄÅÆĂĄÀ', 'aáâãäåæāăąàæ',
    'B', 'b',
    'CÇĆĈĊČ', 'cçćĉċč',
    'DÐĎĐ', 'dðďđď',
    'EÈÉÊËĒĔĖĘĚ', 'eèéêëēĕėęě',
    'F', 'f',
    'GĜĞĠĢ', 'gĝğġģḡ',
    'HĤĦ', 'hĥħ',
    'IÌÍÎÏĨĪĬĮİĲ', 'iìíîïĩīĭįıĳ',
    'JĴ', 'jĵɉ',
    'KĶ', 'kķĸ',
    'LĹĻĽĿŁ£', 'lĺļľŀł',
    'M', 'm',
    'NÑŃŅŇŊ', 'nñńņňŉŋ',
    'OÒÓÔÕÖŌŎŐŒ', 'oòóôõöōŏőœ°',
    'P', 'pꝑꝓ',
    'Q', 'qꝗꝙ',
    'RŔŖŘ', 'rŕŗřˀ',
    'SŚŜŞŠß', 'sśŝşš',
    'TŢŤŦ', 'tţťŧꝷ',
    'UÙÚÛÜŨŪŬŮŰŲ', 'uùúûüũūŭůűų',
    'V', 'vꝟ',
    'WŴ', 'wŵ',
    'X', 'x',
    'YŶŸ', 'yýÿŷ',
    'ZŹŻŽ', 'zźżž'
]

punctuation_charset = [ '.✳', ',;:', '-¬—=', '¶§' ] 

# respectively= acdehimortuvx
subscript_charset = ['\u0363\u0368\u0369\u0364\u036a\u0365\u036b\u0366\u036c\u036d\u0367\u036e\u036f']

# variety of diacritics (to be used in combination with other letters, not to
# be confused with accented 1-byte symbols)
diacritic_charset = ["'ʼ" + ''.join([ chr(c) for c in range(0x300,0x316) ])]

parenthesis_charset = [ '()[]/\\|' ]

abbreviation_charset = [ 'ƺꝙꝮꝯꝫȝꝝ₰ꝛꝰꝭ&§₎כּ' ]

hebrew_charset = [ chr(c) for c in range(0x0591,0x05f5) ]

greek_charset = [ chr(c) for c in range(0x03b1,0x03e1) ] + [ chr(c) for c in range(0x0391,0x03e0) ]

charsets = [ space_charset, latin_charset, punctuation_charset, subscript_charset, diacritic_charset, parenthesis_charset, abbreviation_charset, hebrew_charset, greek_charset ]
