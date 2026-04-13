#!/usr/bin/env python3 

import fargv
from fargv import FargvStr, FargvInt, FargvChoice, FargvBool, FargvPositional

p = {
        'ordered_list': [[], "Some things"],
        'integer': [2, "Integer"],
        'choice': [('1','2','3'), "Choice"],
    }

args, _ = fargv.parse( p )

print(args)
print(args.ordered_list)
print(args.integer)
print(args.choice)
