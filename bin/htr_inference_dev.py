#!/usr/bin/env python3

"""
HTR inference: with various options, for development and testing purpose:

+ flexible input: FSDB or not, page-scope inference (segmentation required) or line image inference.
+ TODO: choice of models
"""

# stdlib
from pathlib import Path
import sys
import re
from typing import Callable, Union
import json
import logging
from datetime import datetime

# 3rd party
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor, Compose
from torchvision.datasets import VisionDataset
import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvTuple
from segtformats import segtformats as sgf

# local
root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 
from libs.htr_model import HTR_Model
from libs import seglib, transforms as tsf
from libs import list_utils as lu
from libs.charter_htr_datasets import CharterInferenceDataset, LineInferenceDataset


logging_format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s"
logging_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }
logging.basicConfig( level=logging.INFO, format=logging_format, force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "htr_dev",
    "model_path": "./best.mlmodel", 
    "device": FargvChoice(["cpu","cuda", "gpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], description="Computing device"),
    "decoder": FargvChoice(['greedy','beam-search'], description="Decoding layer: greedy or beam-search."),
    "img_paths": FargvPositional(default=[]),
    "charter_dirs": [],
    "line_scope": (False, "Inference on line images."),
    "segmentation_suffix": (".lines.pred.json", "Suffix of line segmentation label."),
    "output_dir": ('', 'Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image.'),
    "img_suffix": ".img.jpg",
    "msk_suffix": ".bool.npy.gz",
    "with_mask": True,
    "htr_suffix": '', 
    "output_format": FargvChoice(["stdout", "json", "tsv", "xml"], description="Output formats; 'stdout' and 'tsv' = 3-column output '<index>\t<line id>\t<prediction>', on console and file, respectively, with optional GT and scores columns (see relevant option); 'json' and 'xml' = page-wide segmentation file."),
    "output_data": (["pred"], "By default, the application yields only character predictions; for standard or TSV output, additional data can be chosen: 'scores', 'gt', 'metadata' (see below)."),
    "overwrite_existing": (True, "Write over existing output file (default)."),
    "line_padding_style": FargvChoice(['median', 'noise', 'zero', 'none'], description="How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding"),
    "line_height_factor": FargvFloat(1.0, description="Factor to be applied to the original line strip height."),
    "verbosity": (2, "Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO, default), 3 (DEBUG)"),
}


def pack_fsdb_inputs_outputs( args:dict, segmentation_suffix:str ) -> list[tuple]:
    """
    Compile image files and/or charter paths in the CLI arguments.
    No existence check on the dependency (segmentation path).

    Args:
        dict: the parsed arguments.
        segmentation_suffix (str): suffix of the expected segmentation file.
    Returns:
        list[tuple]: a list of triplets (<img file path>, <segmentation file path>, <output file path>)
    """
    all_img_paths = set([ Path(p) for p in args.img_paths ])

    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir )
        if charter_dir_path.is_dir() and charter_dir_path.joinpath("CH.cei.xml").exists():
            new_imgs = charter_dir_path.glob("*{}".format(args.img_suffix))
            all_img_paths = all_img_paths.union( charter_dir_path.glob("*{}".format(args.img_suffix)))
    path_triplets = []
    for img_path in all_img_paths:
        img_stem = re.sub(r'{}$'.format( args.img_suffix), '', img_path.name )
        if img_stem == img_path.name:
            logger.warning(f"Suffix '{args.img_suffix}' not found in image name {img_path.name}: is the value for the --img_suffix option correct?")
        segfile_path = Path( re.sub(r'{}$'.format( args.img_suffix), segmentation_suffix, str(img_path) ))
        output_dir = img_path.parent if not args.output_dir else Path(args.output_dir)
        out_file_path = f'{img_stem}.{args.appname}.pred{args.htr_suffix}.{args.output_format}'
        path_triplets.append( ( img_path, segfile_path, output_dir.joinpath( out_file_path )))
    #return path_triplets
    return sorted( path_triplets, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.parse( p )
    logger.debug(args)

    if args.device=='cuda' or args.device=='gpu':
        args.device='cuda:0'
    model = HTR_Model.load( args.model_path, device=args.device if args.device!='cpu' else 'cpu' )
    if args.decoder=='beam-search': # this overrides whatever decoding function has been used during training
        model.decoder = HTR_Model.decode_beam_search

    img_height, img_width, padding_style = 128, 2048, args.line_padding_style
    if 'img_height' in model.image_specs:
        img_height = model.image_specs['img_height']
    if 'img_width' in model.image_specs:
        img_width = model.image_specs['img_width']
    if 'padding_style' in model.image_specs:
        padding_style = model.image_specs['padding_style']

    # no pages, only line images (and optional masks)
    if args.line_scope:
        dataset = LineInferenceDataset(
                        args.img_paths, 
                        img_suffix=args.img_suffix, 
                        msk_suffix=args.msk_suffix, 
                        with_mask=args.with_mask,
                        transform = Compose([ 
                            tsf.ResizeToHeight( img_height, img_width ),
                            tsf.PadToWidth( img_width ) ]),
                        padding_style=line_padding_style,)
        
        if not dataset.ok:
            logger.warning("Could not build a proper dataset. Aborting.")
            sys.exit()

        line_dicts = []
        for sample in DataLoader(dataset, batch_size=1):
            try:
                # strings, np.ndarray
                predicted_string, line_scores = model.inference( sample['img'], sample['width'] )
                # since batch is 1, flattening batch values
                line_id = sample['id'][0] # for some reason, the transform wraps the id into an array
                line_dicts.append( { 'id': line_id, 'text': predicted_string[0], 'scores': lu.flatten(line_scores.tolist()) })
            except Exception as e:
                logger.warning("Inference failed on line {} (image file {}): {}".format( line_id, sample['img_path'], e))
                continue
        
        if args.output_format in ('json', 'xml'):
            logger.warning("No option for JSON or XML output format → falling back to standard output.")
            args.output_format = 'stdout'
        if args.output_format in ('tsv', 'stdout'):
            header_row = ['Index', 'Id', 'Prediction']
            if 'gt' in args.output_data:
                header_row.append( 'GT' )
            if 'scores' in args.output_data:
                header_row.append( 'Scores')
            output_rows=[ '\t'.join( header_row ) ]
            for idx, line_dict in enumerate(line_dicts):
                logger.debug( line_dict )
                output_row = [ str(idx), line_dict['id'], line_dict['text'] ]
                if 'gt' in args.output_data and 'gt' in line_dict:
                    output_row.append( line_dict['gt'] )
                if 'scores' in args.output_data and 'scores' in line_dict:
                    output_row.append( str(line_dict['scores']) )

                output_rows.append( '\t'.join( output_row ) )
            if args.output_format == 'stdout':
                print('\n'.join(output_rows))
            else:
                with open( output_file_path, 'w') as htr_outfile:
                    htr_outfile.write( '\n'.join( output_rows) )
                    htr_outfile.write( '\n')

    # pages (+ segmentation data)
    else:
        for img_idx, img_triplet in enumerate( pack_fsdb_inputs_outputs( args, args.segmentation_suffix )):

            img_path, segmentation_file_path, output_file_path = img_triplet
            if segmentation_file_path.suffix != '.json' and args.line_height_factor != 1.0:
                logger.info("-args.line_height_factor={} not applicable to XML segmentation data: ignored.")
            logger.debug( "File path={}".format( img_triplet[0]))
            if not args.overwrite_existing and output_file_path.exists():
                logger.debug("Found {}: exiting.".format( output_file_path ))
                continue

            if not segmentation_file_path.exists():
                logger.info("Skipping image {}: no segmentation file {} found.".format( img_path, segmentation_file_path ))
                continue
        
            dataset = CharterInferenceDataset( 
                                        img_path, segmentation_file_path,
                                        transform = Compose([ 
                                            tsf.ResizeToHeight( img_height, img_width ), 
                                            tsf.PadToWidth( img_width ) ]),
                                        padding_style=padding_style,
                                        line_height_factor=args.line_height_factor,)
            if not dataset.ok:
                logger.warning("Could not build a proper dataset. Aborting.")
                continue
             
            # 2. HTR inference

            # Idea: the live page dictionary is updated with all the info that may be of interest:
            # depending on the output format chosen, some of it gets deleted later.
            for line, sample in enumerate(DataLoader(dataset, batch_size=1)):
                try:
                    # strings, np.ndarray
                    predicted_string, line_scores = model.inference( sample['img'], sample['width'] )
                    print(predicted_string)
                    # since batch is 1, flattening batch values
                    line_id = sample['id'][0] # for some reason, the transform wraps the id into an array
                    line_dict = { 'id': line_id, 'text': predicted_string[0], 'scores': lu.flatten(line_scores.tolist()) }
                    dataset.update_pagedict_line( line_id, line_dict, keep_gt=('gt' in args.output_data) )
                except Exception as e:
                    logger.warning("Inference failed on line {} in file {}: {}".format( line, img_path, e))
                    continue

            # 3. Output
            if args.output_format in ('json', 'xml') and ('gt' in args.output_data or 'scores' in args.output_data):
                logger.warning("Skipping output data fields ({}): choose either 'stdout' or 'tsv' to include them in the output.".format(args.output_data))

            # stdout and tsv for extra data
            if args.output_format in ('stdout', 'tsv'):
                header_row = ['Index', 'Id', 'Prediction']
                if 'gt' in args.output_data:
                    header_row.append( 'GT' )
                if 'scores' in args.output_data:
                    header_row.append( 'Scores')
                if 'metadata' in args.output_data and 'metadata' in dataset.page_dict:
                    header_row.extend( [str.capitalize(k) for k in dataset.page_dict['metadata'].keys()] )
                output_rows=[ '\t'.join( header_row ) ]
                for idx, line_dict in enumerate(dataset.page_dict['lines']):
                    logger.debug( line_dict )
                    output_row = [ str(idx), line_dict['id'], line_dict['text'] if 'text' in line_dict else '-']
                    if 'gt' in args.output_data and 'gt' in line_dict:
                        output_row.append( line_dict['gt'] )
                    if 'scores' in args.output_data and 'scores' in line_dict:
                        output_row.append( str(line_dict['scores']) )
                    if 'metadata' in args.output_data and 'metadata' in dataset.page_dict:
                        output_row.extend([ str(elt) for elt in dataset.page_dict['metadata'].values() ])
                    output_rows.append( '\t'.join( output_row ) )
                if args.output_format == 'stdout':
                    print('\n'.join(output_rows))
                else:
                    with open( output_file_path, 'w') as htr_outfile:
                        htr_outfile.write( '\n'.join( output_rows) )
                        htr_outfile.write( '\n')

            # Json and Xml for standard page annotation
            elif args.output_format in ('json', 'xml'):
                for line in dataset.page_dict['lines']:
                    if 'scores' in line:
                        del line['scores']
                    if 'gt' in line:
                        del line['gt']
                # deleting top-level 'lines' reference
                del dataset.page_dict['lines']
                dataset.page_dict.update({
                    'created': str(datetime.now()), 'creator': __file__,    
                })
                if args.output_format == 'json':
                    with open( output_file_path, 'w') as htr_outfile:
                        htr_outfile.write(json.dumps( dataset.page_dict, indent=2))
                elif args.output_format == 'xml':
                    sgf.page_xml_from_segmentation_dict( dataset.page_dict, output_file=output_file_path )
            if output_file_path.exists():
                logger.info(f"HTR output saved in {output_file_path}")
                

