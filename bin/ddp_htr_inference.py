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
from typing import List, Tuple, Callable, Union
import json
import logging

# 3rd party
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToTensor, Compose
from torchvision.datasets import VisionDataset

# local
root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 
from model_htr import HTR_Model
from libs import seglib, transforms as tsf
from libs import list_utils as lu

logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "ddpa_htr_inference",
    "model_path": "/tmp/model_monasterium-2024-10-28.mlmodel", # gdown https://drive.google.com/uc?id=1GOKgGWvhO7ugWw0tevzXhQa2cVx09iLu 
    "img_paths": set([]),
    "charter_dirs": set(["./"]),
    "segmentation_dir": ['', 'Alternate location to search for the image segmentation data files (for testing).'], # for testing purpose
    "segmentation_file_suffix": "lines.pred.json", # under each image dir, suffix of the subfolder that contains the segmentation data 
    "output_dir": ['', 'Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image.'],
    "htr_file_suffix": "htr.pred", # under each image dir, suffix of the subfolder that contains the transcriptions
    "output_format": [ ("json", "stdout", "tsv"), "Output format: 'stdout' for sending decoded lines on the standard output; 'json' and 'tsv' create JSON and TSV files, respectively."],
    #"output_data": [ ("pred", "logits"), "By default, the application yields character predictions; 'logits' have it returns logits instead."],
    "padding_style": [ ('median', 'noise', 'zero', 'none'), "How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding"],
}


class InferenceDataset( VisionDataset ):

    def __init__(self, img_path: Union[str,Path],
                 segmentation_data: Union[str,Path], 
                 transform: Callable=None,
                 padding_style=None) -> None:
        """ A minimal dataset class for inference on a single charter (no transcription in the sample).

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

        trf = transform if transform else ToTensor()
        super().__init__(root, transform=trf )

        img_path = Path( img_path ) if type(img_path) is str else img_path
        segmentation_data = Path( segmentation_data ) if type(segmentation_data) is str else segmentation_data

        # extract line images: functions line_images_from_img_* return tuples (<line_img_hwc>: np.ndarray, <mask_hwc>: np.ndarray)
        line_extraction_func = seglib.line_images_from_img_json_files if segmentation_data.suffix == '.json' else seglib.line_images_from_img_xml_files

        line_padding_func = lambda x, m, channel_dim=2: x # by default, identity function
        if padding_style == 'noise':
            line_padding_func = tsf.bbox_noise_pad
        elif padding_style == 'median':
            line_padding_func = tsf.bbox_median_pad
        elif padding_style == 'zero':
            line_padding_func = tsf.bbox_zero_pad
        print(line_padding_func)

        self.data = []

        for (img_hwc, mask_hwc) in line_extraction_func( img_path, segmentation_data ):
            mask_hw = mask_hwc[:,:,0]
            self.data.append( { 'img': line_padding_func( img_hwc, mask_hw, channel_dim=2 ), #tsf.bbox_median_pad( img_hwc, mask_hw, channel_dim=2 ), 
                                'height':img_hwc.shape[0],
                                'width': img_hwc.shape[1],
                               } )

    def __getitem__(self, index: int):
        sample = self.data[index].copy()
        logger.debug(f"type(sample['img'])={type(sample['img'])} with shape= {sample['img'].shape}" )
        return self.transform( sample )

    def __len__(self):
        return len(self.data)


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

        # remove any dot-prefixed substring from the file name
        # (Path.suffix() only removes the last suffix)
        stem = re.sub(r'\..+', '',  img_path.name )

        segmentation_dir = img_path.parent

        if args.segmentation_dir != "":
            if Path( args.segmentation_dir ).exists():
                segmentation_dir = Path( args.segmentation_dir )
            else:
                raise FileNotFoundError(f"Provided segmentation directory {args.segmentation_dir} does not exists!")

        segmentation_file_path = segmentation_dir.joinpath(f'{stem}.{args.segmentation_file_suffix}')

        dataset = None
        if not segmentation_file_path.exists():
            logger.info("Skipping image {}: no segmentation file {} found.".format( img_path, segmentation_file_path ))
            continue
        dataset = InferenceDataset( img_path, segmentation_file_path,
                                    transform = Compose([ ToTensor(),
                                                          tsf.ResizeToHeight(128,2048),
                                                          tsf.PadToWidth(2048),]),
                                    padding_style=args.padding_style)
        logger.info("Charter mini-dataset: " + str(dataset))
        
        if dataset is None:
            raise FileNotFoundError("Could not build a proper dataset. Aborting.")
         
        # 2. HTR inference

        model = HTR_Model.load( args.model_path )
        predictions = []

        for line, sample in enumerate(DataLoader(dataset, batch_size=1)):
            # strings, np.ndarray
            predicted_string, line_scores = model.inference_task( sample['img'], sample['width'] )
            # since batch is 1, flattening batch values
            predictions.append( {"line_id": line, "transcription": predicted_string[0], 'scores': lu.flatten(line_scores.tolist()) } )

        # 3. Output
        if args.output_format == 'stdout':
            print( '\n'.join( str(p) for p in predictions) )

        elif args.output_format in ('json', 'tsv'):

            output_dir = Path( img_path ).parent
            output_file_name = output_dir.joinpath(f'{stem}.{args.htr_file_suffix}.{args.output_format}')
            with open( output_file_name, 'w') as htr_outfile:
                if args.output_format == 'json':
                    json.dump( str(predictions), htr_outfile, indent=4)
                elif args.output_format == 'tsv':
                    print( '\n'.join( [ f'{line_dict["line_id"]}\t{line_dict["transcription"]}' for line_dict in predictions ] ), file=htr_outfile )
                logger.info(f"Output transcriptions in file {output_file_name}")

            


