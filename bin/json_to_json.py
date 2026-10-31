#!/usr/bin/env python3
"""
JSON -> JSON conversion, with an emphasis on HTR.

Read a JSON metadata file, with a choice of options:

+ expand polygon heights wr/ original x-height
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

+ the '--promote_regions' option is absent
+ additional options to merge segmentation into htr, or conversely
"""

import sys
import json
import re
from datetime import datetime
from pathlib import Path
import logging

import fargv
from fargv import FargvPositional, FargvChoice

from segtformats import segtformats as sgf

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_paths': FargvPositional(default=[], description="Input file (JSON)."),
    'out': FargvChoice(['','auto'], description="Set to 'auto' for output to filename <input stem>.<output_suffix>; leave empty for standard output."),
    'line_height_factor': (1.0, "Factor to be applied to the original line strip height."),
    'input_suffix': ('.htr.pred.json', "Input file suffix."),
    'output_suffix': '.json',
    'overwrite_existing': (False, "Overwrite an existing output file."),
    'drop_transcription': (False, "Drop line transcription, if it exists"),
    "comment": ('',"A text string to be added to the <Comments> elt."),
    "inject_htr_suffix": ('',"Inject HTR content of file with given suffix into the main file, while keeping the segmentation."),
    "inject_segmentation_suffix": ('',"Inject the segmentation data of file with given suffix into the main file, while keeping the HTR."),
    "force": (False, "Force injections on mismatched ids (but not on mismatched line counts)."),
    'verbosity': FargvChoice(['2','0','1','3'], description="Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"),
}

logging_format="%(asctime)s - %(levelname)s: %(filename)s: %(funcName)s - %(message)s"
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    args, _ = fargv.parse( p )

    logging.basicConfig(
            level={0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }[int(args.verbosity)],
            format=logging_format,
            force=True )

    for file_path in args.file_paths:

        json_path = Path( file_path )

        segdict = None
        with open( json_path, 'r') as json_if:
            segdict = json.load( json_if )

            # always
            if 'lines' in segdict:
                segdict = sgf.segdict_sink_lines( segdict )

            line_dicts= sgf.line_dicts_from_segmentation_dict( segdict )

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
                    if len(line_dicts) != len(injection_lines):
                        print("Line counts in the two files do not seem to match. Please do a manual check.")
                        sys.exit()
                    # check ids
                    zip_lines = list(zip(line_dicts, injection_lines))
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
                for line in line_dicts:
                    if 'text' in line:
                        del line['text']

            # expand polygons
            if args.line_height_factor != 1.0:
                line_polygons = seglib.line_polygons_from_segmentation_dict( segdict, factor=args.line_height_factor )
                for polyg, line in zip( line_polygons, line_dicts ):
                    line['coords']=polyg

            # insert metadata at the top
            regions = segdict['regions']
            del segdict['regions']
            segdict['metadata'].update( {'created': str(datetime.now().isoformat('T','seconds')), 'creator': __file__ })

            if args.line_height_factor != 1.0:
                segdict['line_height_factor']=args.line_height_factor
            if args.comment:
                segdict['metadata']['comments']=args.comment
            segdict['regions']=regions

            segdict_str = json.dumps( segdict, indent=2 )
            if not args.out:
                print( segdict_str )
                continue

            out_path = ''
            if args.out == 'auto':
                if not re.search( r'{}$'.format(args.input_suffix), json_path.name):
                    logger.warning(f"Input file path '{json_path.name}' does not match input suffix '{args.input_suffix}': output aborted.")
                    continue
                out_path = re.sub(r'{}$'.format( args.input_suffix ), args.output_suffix, json_path )
            logger.debug(f"Output file = {out_path}")

            if not args.overwrite_existing and Path(out_path).exists():
                logger.info("File {} exists: skipping.".format( out_path ))
                continue
            with open(out_path, 'w') as json_outf:
                logger.info(f"W: {json_path.name} → {Path(out_path).name}")
                json_outf.write( segdict_str )

