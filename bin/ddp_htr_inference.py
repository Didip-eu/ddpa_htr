#!/usr/bin/env python3

"""
HTR inference on page, with segmentation provided.
"""

# stdlib
from pathlib import Path
import sys
import fargv
import re
import glob
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

# local
root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 
from model_htr import HTR_Model
from libs import seglib, transforms as tsf
from libs import list_utils as lu

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "htr",
    "model_path": "./best.mlmodel", # gdown https://drive.google.com/uc?id=1GOKgGWvhO7ugWw0tevzXhQa2cVx09iLu 
    "decoder": [('greedy','beam-search'), "Decoding layer: greedy or beam-search."],
    "img_paths": set([]),
    #"charter_dirs": set(["./"]),
    "charter_dirs": set([]),
    "segmentation_suffix": ".lines.pred.json", # under each image dir, suffix of the subfolder that contains the segmentation data 
    "output_dir": ['', 'Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image.'],
    "img_suffix": ".img.jpg",
    "htr_suffix": ".htr.pred", # under each image dir, suffix of the subfolder that contains the transcriptions
    "output_format": [ ("stdout", "json", "tsv", "xml"), "Output formats; 'stdout' and 'tsv' = 3-column output '<index>\t<line id>\t<prediction>', on console and file, respectively, with optional GT and scores columns (see relevant option); 'json' and 'xml' = page-wide segmentation file."],
    "output_data": [ set(["pred"]), "By default, the application yields only character predictions; for standard or TSV output, additional data can be chosen: 'scores', 'gt', 'metadata' (see below)."],
    'overwrite_existing': [1, "Write over existing output file (default)."],
    "line_padding_style": [ ('median', 'noise', 'zero', 'none'), "How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding"],
}


class InferenceDataset( VisionDataset ):

    def __init__(self, img_path: Union[str,Path],
                 segmentation_data: Union[str,Path], 
                 transform: Callable=None,
                 padding_style=None) -> None:
        """ A minimal dataset class for inference on a single charter (no transcription in the sample).
        Allow for keeping the segmentation meta-data along with the about-to-be generated HTR.

        Args:
            img_path (Union[Path,str]): charter image path
            segmentation_data (Union[Path, str]): segmentation metadata (XML or JSON)
            transform (Callable): Image transform.
            padding_style (str): How to pad the bounding box around the polygons, when 
                building the initial, raw dataset (before applying any transform):
                + 'median'= polygon's median value,
                + 'noise' = random noise,
                + 'zero'= 0-padding, 
                + None (default) = no padding, i.e. raw bounding box
        """

        trf = v2.Compose( [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        if transform is not None: 
            trf = v2.Compose( [trf, transform] )
        super().__init__(root, transform=trf )

        img_path = Path( img_path ) if type(img_path) is str else img_path
        segmentation_data = Path( segmentation_data ) 

        # extract line images: functions line_images_from_img_* return a pair (<seg_dict>, <sequence of tuples (<line_img_hwc>: np.ndarray, <mask_hwc>: np.ndarray)>)
        line_extraction_func = seglib.line_images_from_img_json_files if segmentation_data.suffix == '.json' else seglib.line_images_from_img_xml_files

        line_padding_func = lambda x, m, channel_dim=2: x # by default, identity function
        if padding_style == 'noise':
            line_padding_func = tsf.bbox_noise_pad
        elif padding_style == 'median':
            line_padding_func = tsf.bbox_median_pad
        elif padding_style == 'zero':
            line_padding_func = tsf.bbox_zero_pad

        self.data = []
        try:
            # This creates a page dict with a convenient top-level 'lines' array, raised from 
            # its containing region(s): allow for easy update of all line objects - this top-level 
            # reference to the line array is later deleted, before serializing the ouput.
            self.page_dict = line_extraction_func( img_path, segmentation_data, as_dictionary=True )
            for img_hwc, mask_hwc, line_dict in self.page_dict['lines']:
                mask_hw = mask_hwc[:,:,0]
                self.data.append( { 'img': line_padding_func( img_hwc, mask_hw, channel_dim=2 ), 
                                    'height':img_hwc.shape[0],
                                    'width': img_hwc.shape[1],
                                    'id': str(line_dict['id']),
                                    'img_filename': str(img_path),
                                   } )
            # at this point, we don't need the image data anymore: restoring original line dictionaries into the page data
            self.page_dict['lines'] = [ triplet[2] for triplet in self.page_dict['lines'] ]
            self.line_id_to_index = { str(lrecord['id']): idx for idx, lrecord in enumerate( self.page_dict['lines']) }
        except Exception as e:
            logger.warning("Error when creating the line dataset: {}".format( e ))
        self.ok = len(self.data) > 0

    def update_pagedict_line(self, line_id:str, kv: dict, keep_gt=0 ):
        """ Update a given line dictionary with prediction data, whatever they are."""
        this_line = self.page_dict['lines'][ self.line_id_to_index[ line_id ]]
        if keep_gt:
            this_line['gt']=this_line['text']
        this_line.update( kv )

    def __getitem__(self, index: int):
        sample = self.data[index]
        sample['img']=sample['img'].copy() # Torch warning otherwise
        logger.debug(f"type(sample['img'])={type(sample['img'])} with shape= {sample['img'].shape}" )
        return self.transform( sample )

    def __len__(self):
        return len(self.data)


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
        segfile_path = Path( re.sub(r'{}$'.format( args.img_suffix), segmentation_suffix, str(img_path) ))
        output_dir = img_path.parent if not args.output_dir else Path(args.output_dir)
        path_triplets.append( ( img_path, segfile_path, output_dir.joinpath( f'{img_stem}.{args.appname}.pred.{args.output_format}')))
    #return path_triplets
    return sorted( path_triplets, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug(args)

    model = HTR_Model.load( args.model_path )
    if args.decoder=='beam-search': # this overrides whatever decoding function has been used during training
        model.decoder = HTR_Model.decode_beam_search

    for img_idx, img_triplet in enumerate( pack_fsdb_inputs_outputs( args, args.segmentation_suffix )):

        img_path, segmentation_file_path, output_file_path = img_triplet
        logger.debug( "File path={}".format( img_triplet[0]))
        if not args.overwrite_existing and output_file_path.exists():
            continue

        if not segmentation_file_path.exists():
            logger.info("Skipping image {}: no segmentation file {} found.".format( img_path, segmentation_file_path ))
            continue
    
        dataset = InferenceDataset( img_path, segmentation_file_path,
                                    transform = Compose([ tsf.ResizeToHeight(128,2048), tsf.PadToWidth(2048),]),)
        if not dataset.ok:
            logger.warning("Could not build a proper dataset. Aborting.")
            continue
         
        # 2. HTR inference

        # Idea: the live page dictionary is updated with all the info that may be of interest:
        # depending on the output format chosen, some of it gets deleted later.
        for line, sample in enumerate(DataLoader(dataset, batch_size=1)):
            try:
                # strings, np.ndarray
                predicted_string, line_scores = model.inference_task( sample['img'], sample['width'] )
                # since batch is 1, flattening batch values
                line_id = sample['id'][0] # for some reason, the transform wraps the id into an array
                line_dict = { 'id': line_id, 'text': predicted_string[0], 'scores': lu.flatten(line_scores.tolist()) }
                dataset.update_pagedict_line( line_id, line_dict, keep_gt=('gt' in args.output_data) )
            except Exception as e:
                logger.warning("Inference failed on line {} in file {}".format( line, img_path))
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
                output_row = [ str(idx), line_dict['id'], line_dict['text'] ]
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
                seglib.xml_from_segmentation_dict( dataset.page_dict, output_file_path )
        if output_file_path.exists():
            logger.info(f"HTR output saved in {output_file_path}")
            pass

            


