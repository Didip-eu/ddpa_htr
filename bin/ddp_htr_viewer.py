#!/usr/bin/env python3

"""
HTR viewer on page, with HTR results provided.
"""

# stdlib
from pathlib import Path
import sys
import fargv
import re
import glob
from typing import List, Tuple, Callable, Union
import json
import logging

# 3rd party
from PIL import Image

# local
root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 
from libs import list_utils as lu
from libs import visuals as vz


logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "ddpa_htr_viewer",
    "img_paths": set([]),
    "charter_dirs": set(["./"]),
    "htr_file_suffix": "htr.pred.json", # under each image dir, suffix of the subfolder that contains the transcriptions
    "output_format": [ ("txt", "plt", "png"), "Output format: 'txt' for ASCII plot (default); 'plt' for PyPlot; 'png' for on-disk image;"],
}


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    all_img_paths = list(sorted(args.img_paths))

    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir )
        logger.debug(f"Charter Dir: {charter_dir}")
        if charter_dir_path.is_dir() and charter_dir_path.joinpath("CH.cei.xml").exists():
            charter_images = [str(f) for f in charter_dir_path.glob("*.img.*")]
            print(charter_images)
            all_img_paths += charter_images
        else:
            logger.error("Skipping directory {}: check that it is an existing charter directory and that it contains a {} file.".format( charter_dir_path, "CH.cei.xml"))

        args.img_paths = list(all_img_paths)

    logger.debug(args)

    for img_path in list( args.img_paths):

        img_path = Path( img_path )
        
        print(img_path)

        # remove any dot-prefixed substring from the file name
        # (Path.suffix() only removes the last suffix)
        stem = re.sub(r'\..+', '',  img_path.name )

        htr_dir = img_path.parent

        htr_file_path = htr_dir.joinpath(f'{stem}.{args.htr_file_suffix}')

        print(htr_file_path)

        with open( htr_file_path, 'r') as htrf:

            htr_dict = eval(json.load( htrf ))

            strings, scores = ( [ l[k] for l in htr_dict ] for k in ('transcription', 'scores'))

            vz.predictions_over_scores( strings, scores )


            


