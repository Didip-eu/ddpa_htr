#!/usr/bin/env python3
"""
JSON -> JSON conversion

Read a JSON segmentation file, with a choice of options:

+ remove transcription data
+ add a comment
+ merge with content (HTR or segmentation) of another file

Legacy format (with a top-level 'lines' array) is silently converted to 
the nested structure 

    { 'regions': [
        { 'coords': [ ... ], 
          'lines': [{ ... }, ... ] }, ...
      ]
    }

Note: to minimize dependencies, this script does not include the '-line_height_factor' option.
   See its counterpart in 'ddpa_lines_ng'.

"""

import sys
import json
import fargv
from pathlib import Path
from datetime import datetime

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_path': '',
    'polygon_key': 'coords',
    'output_file': ['', "Output file (default: standard output)."],
    'overwrite_existing': [0, "Overwrite an existing output file."],
    'drop_transcription': [0, "Extract line transcription, if it exists"],
    "comment": ['',"A text string to be added to the <Comments> elt."],
    "inject_htr": ['',"Inject the argument file's HTR content into the main file, while keeping the segmentation."],
    "inject_segmentation": ['',"Inject the argument file's segmentation into the main file, while keeping the HTR."],
    "force": [0, "Force injections on mismatched ids (but not on mismatched line counts)."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    segdict = None

    with open( args.file_path, 'r') as json_if:
        segdict = json.load( json_if )

        # always 
        if 'lines' in segdict:
            segdict = seglib.segdict_sink_lines( segdict )

        lines = seglib.line_dicts_from_segmentation_dict( segdict )

        if args.inject_htr or args.inject_segmentation:
            injection_path = Path(args.inject_htr if args.inject_htr else args.inject_segmentation)
            with open( injection_path ) as injection_if:
                injection_dict = json.load( injection_if )
                
                # check number of regions and lines
                if len(segdict['regions']) != len(injection_dict['regions']):
                    print("Region counts in the two files do not seem to match. Please do a manual check.")
                    sys.exit()
                injection_lines = seglib.line_dicts_from_segmentation_dict( injection_dict )
                if len(lines) != len(injection_lines):
                    print("Line counts in the two files do not seem to match. Please do a manual check.")
                    sys.exit()
                # check ids 
                zip_lines = list(zip(lines, injection_lines))
                if not all([ l1['id']==l2['id'] for (l1, l2) in zip_lines ]):
                    if not args.force:
                        answer=input("Line ids in the two files do not match. Do you want to continue? [Yn]")
                        if answer!='' or answer!='y' or answer!='Y':
                            sys.exit()
                # inject line data
                for l1, l2 in zip_lines:
                    if args.inject_htr:
                        if 'text' in l2:
                            l1['text'] = l2['text'] 
                    elif args.inject_segmentation:
                        for k in ('coords', 'baseline', 'centerline'):
                            if k in l2:
                                l1[k] = l2[k]
                # inject region data
                if args.inject_segmentation:
                    zip_regions = list(zip(segdict['regions'], injection_dict['regions']))
                    if not all([ r1['id']==r2['id'] for (r1, r2) in zip_regions ]):
                        if not args.force:
                            answer=input("Regions ids in the two files do not match. Do you want to continue? [Yn]")
                            if answer!='' or answer!='y' or answer!='Y':
                                sys.exit()
                    for r1, r2 in zip_regions:
                        if 'coords' in r2:
                            r1['coords'] = r2['coords']
                

        # remove transcriptions
        if not args.inject_htr and args.drop_transcription:
            for line in lines: 
                if 'text' in line:
                    del line['text']

        # insert metadata at the top
        segdict['metadata'].update( {'created': str(datetime.now()), 'creator': __file__ })
        regions = segdict['regions']
        del segdict['regions']
        if args.comment:
            segdict['metadata']['comments']=args.comment
        segdict['regions']=regions

        # output
        if segdict is not None:
            if args.output_file:
                output_path = Path( output_file )
                if not args.overwrite_existing and output_path.exists():
                    print("File {} exists: abort.".format(args.output_file))
                else:
                    with open( output_path,'w') as of:
                        of.write( json.dumps( segdict, indent=2))
            else:
                print( json.dumps( segdict, indent=2 ))

