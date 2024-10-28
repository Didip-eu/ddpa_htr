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

# didip
from didip_handwriting_datasets import monasterium as mom, alphabet 

# local
root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 

from model_htr import HTR_Model
import seglib

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "ddpa_htr",
    "model_path": str(Path( root, 'models', 'htr', 'default.mlmodel' )),
    "img_paths": set(glob.glob( str(Path.home().joinpath("tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/*/*.jpg")))),
    "segmentation_dir": ['', 'Alternate location to search for the image segmentation data files (for testing).'], # for testing purpose
    "segmentation_file_suffix": "lines.pred", # under each image dir, suffix of the subfolder that contains the segmentation data 
    "output_dir": ['', 'Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image.'],
    "htr_file_suffix": "htr.pred", # under each image dir, suffix of the subfolder that contains the transcriptions
    "output_format": [ ("stdout", "json", "tsv"), "Output format: 'stdout' for sending decoded lines on the standard output; 'json' and 'tsv' create JSON and TSV files, respectively."],
    "padding_style": [ ('median', 'noise', 'zero', 'none'), "How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding"],
}


class InferenceDataset( VisionDataset ):

    def __init__(self, img_path: Union[str,Path],
                 segmentation_data: Union[str,Path], 
                 transform: Callable=None,
                 padding_style=None) -> None:
        """ A minimal dataset class for inference on a single charter.

        + transcription not included in the sample

        :param img_path: charter image path
        :type img_path: Union[Path,str]

        :param segmentation_data: segmentation metadata (XML or JSON)
        :type segmentation_data: Union[Path, str]

        :param padding_style: 
            How to pad the bounding box around the polygons, when building the initial, raw dataset (before applying any transform):
            + 'median'= polygon's median value,
            + 'noise' = random noise,
            + 'zero'= 0-padding, 
            + None (default) = no padding, i.e. raw bounding box
        :type padding_style: str
        """

        trf = transform if transform else ToTensor()
        super().__init__(root, transform=trf )

        # str -> path conversion
        img_path = Path( img_path ) if type(img_path) is str else img_path
        segmentation_data = Path( segmentation_data ) if type(segmentation_data) is str else segmentation_data

        # extract line images: functions line_images_from_img_* return tuples (<line_img_hwc>: np.ndarray, <mask_hwc>: np.ndarray)
        line_extraction_func = seglib.line_images_from_img_segmentation_dict if segmentation_data.suffix == '.json' else seglib.line_images_from_img_xml_files

        line_padding_func = lambda x, m, c: x # by default, identity function
        if padding_style == 'noise':
            line_padding_func = mom.MonasteriumDataset.bbox_noise_pad
        elif padding_style == 'median':
            line_padding_func = mom.MonasteriumDataset.bbox_median_pad
        elif padding_style == 'zero':
            line_padding_func = mom.MonasteriumDataset.bbox_zero_pad


        self.data = []

        for (img_hwc, mask_hwc) in line_extraction_func( img_path, segmentation_data ):
            mask_hw = mask_hwc[:,:,0]
            self.data.append( { 'img': line_padding_func( img_hwc, mask_hw, channel_dim=2 ), #mom.MonasteriumDataset.bbox_median_pad( img_hwc, mask_hw, channel_dim=2 ), 
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

    for img_path in list( args.img_paths):

        # remove any dot-prefixed substring from the file name
        # (Path.suffix() only removes the last suffix)
        stem = re.sub(r'\..+', '',  Path( img_path ).name )

        segmentation_dir = Path( img_path ).parent

        if args.segmentation_dir != "":
            if Path( args.segmentation_dir ).exists():
                segmentation_dir = Path( args.segmentation_dir )
            else:
                raise FileNotFoundError(f"Provided segmentation directory {args.segmentation_dir} does not exists!")

        # Look for existing segmentation file (pattern: <image file stem>.lines.pred.{xml,json})
        segmentation_file_path_candidates = [ segmentation_dir.joinpath(f'{stem}.{args.segmentation_file_suffix}.{format_suffix}') for format_suffix in ('xml','json') ]

        dataset = None
        for segmentation_file_path in segmentation_file_path_candidates:
            if not segmentation_file_path.exists():
                continue
            dataset = InferenceDataset( img_path, 
                                        segmentation_file_path,
                                        transform = Compose([ ToTensor(),
                                                              mom.ResizeToHeight(128,3200),
                                                              mom.PadToWidth(3200),]))
            logger.info(dataset)
            break
        if dataset is None:
            raise FileNotFoundError("Could not build a proper dataset. Aborting.")
         
        # 2. HTR inference

        model = HTR_Model( mom.MonasteriumDataset.get_default_alphabet(), net=args.model_path  )

        predictions = []

        for line, sample in enumerate(DataLoader(dataset, batch_size=1)):
            predictions.append( {"line_id": line, "transcription": model.inference_task( sample['img'], sample['width'] ) } )

        # 3. Output
        if args.output_format == 'stdout':
            print( predictions )

        elif args.output_format in ('json', 'tsv'):

            output_dir = Path( img_path ).parent
            output_file_name = output_dir.joinpath(f'{stem}.{args.htr_file_suffix}.{args.output_format}')
            with open( output_file_name, 'w') as htr_outfile:

                if args.output_format == 'json':
                    print( json.dumps( predictions ), file=htr_outfile)
                elif args.output_format == 'tsv':
                    print( '\n'.join( [ f'{line_dict["line_id"]}\t{line_dict["transcription"]}' for line_dict in predictions ] ), file=htr_outfile )

                print(f"Output transcriptions in file {output_file_name}")

            


