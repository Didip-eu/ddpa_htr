#!/usr/bin/env python3
"""
JSON -> JSON conversion, with an emphasis on HTR.

Read a JSON metadata file, with a choice of options:

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

Note: although this script is similar to its counterpart in 'ddpa_lines_ng', 
its features are tailored to HTR tasks:

+ to minimize dependencies, the '-line_height_factor' option is not included,
  nor is the '-promote_regions' one.
+ options to merge segmentation into htr, or conversely
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvTuple

from segtformats import segtformats as sgf

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_paths': FargvPositional(default=[], description="Input file (JSON)."),
    'polygon_key': 'coords',
    'input_suffix': ('.htr.pred.json', "Input file suffix."),
    'output_suffix': ('', "Output file suffix; if empty, write on standard output"),
    'output_file': ('', "Output file (default: standard output)."),
    'overwrite_existing': (False, "Overwrite an existing output file."),
    'drop_transcription': (False, "Extract line transcription, if it exists"),
    "comment": ('',"A text string to be added to the <Comments> elt."),
    "inject_htr_suffix": ('',"Inject HTR content of file with given suffix into the main file, while keeping the segmentation."),
    "inject_segmentation_suffix": ('',"Inject the segmentation data of file with given suffix into the main file, while keeping the HTR."),
    "force": (False, "Force injections on mismatched ids (but not on mismatched line counts)."),
}


if __name__ == '__main__':

    args, _ = fargv.parse( p )

    for file_path in args.file_paths:

        output_file_path = Path( file_path.replace( args.input_suffix, args.output_suffix )) if args.output_suffix else None
        print(f'{file_path} → {output_file_path}')
        if output_file_path and not args.overwrite_existing and output_file_path.exists():
            print(f"Existing {output_file_path}: skipping." )
            continue

        json_path = Path( file_path )

        segdict = None
        with open( args.file_path, 'r') as json_if:
            segdict = json.load( json_if )

            # always 
            if 'lines' in segdict:
                segdict = sgf.segdict_sink_lines( segdict )

            lines = sgf.line_dicts_from_segmentation_dict( segdict )

            # at most one of the two suffixes must be set 
            if bool(args.inject_htr_suffix) != bool(args.inject_segmentation_suffix):

                injection_path = Path(file_path.replace(args.input_suffix, args.inject_htr_suffix)) if args.inject_htr_suffix else Path(file_path.replace( args.input_suffix, args.inject_segmentation_suffix))
                with open( injection_path ) as injection_if:
                    injection_dict = json.load( injection_if )
                    
                    # check number of regions and lines
                    if len(segdict['regions']) != len(injection_dict['regions']):
                        print("Region counts in the two files do not seem to match. Please do a manual check.")
                        sys.exit()
                    injection_lines = sgf.line_dicts_from_segmentation_dict( injection_dict )
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
                        elif args.inject_segmentation_suffix:
                            for k in ('coords', 'baseline', 'centerline'):
                                if k in l2:
                                    l1[k] = l2[k]
                    # inject region data
                    if args.inject_segmentation_suffix:
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
            if not args.inject_htr_suffix and args.drop_transcription:
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

