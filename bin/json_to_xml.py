#!/usr/bin/env python3
"""
Script for JSON -> PageXML conversion.

To minimize dependencies, this script does not include the '-line_height_factor' option.
See its counterpart in 'ddpa_lines_ng' for that matter.

"""

import sys
import json
import fargv
from pathlib import Path

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib



p = {
    'file_paths': set([]),
    'polygon_key': 'coords',
    'output_format': ('xml', 'stdout'),
    'with_transcription': [1, "Extract line transcription, if it exists"],
    'overwrite_existing': [0, "Overwrite an existing output file."],
    'comment': ['',"A text string to be added to the <Comments> elt."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    for json_path in args.file_paths:
        json_path=Path( json_path )
        xml_path = json_path.with_suffix('.xml')

        with open( json_path, 'r') as json_if:
            segdict = json.load( json_if )
            if args.comment:
                segdict['comment']=args.comment

            if args.output_format == 'stdout':
                seglib.xml_from_segmentation_dict( segdict, '', polygon_key=args.polygon_key, with_text=args.with_transcription )
            else:
                if not args.overwrite_existing and xml_path.exists():
                    print("File {} exists: abort.".format( xml_path ))
                else:
                    seglib.xml_from_segmentation_dict( segdict, xml_path, polygon_key=args.polygon_key, with_text=args.with_transcription )

