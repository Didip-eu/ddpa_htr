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

# 3rd party
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# didip
from didip_handwriting_datasets import monasterium as mom, alphabet

# local


root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 

import seglib

p = {
    "appname": "htr",
    "model_path": str(Path( root, 'models', 'htr', 'default.mlmodel' )),
    "img_paths": set(glob.glob( str(Path.home().joinpath("tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*.jpg")))),
    "segmentation_dir": ['', 'Alternate location to search for the image segmentation data files (for testing).'], # for testing purpose
    "segmentation_file_suffix": "lines.pred", # under each image dir, suffix of the subfolder that contains the segmentation data 
    "no_legacy_polygons": [True, "Enforce newer polygon extraction method, no matter how the model was trained."]
}


class InferenceDataset( Dataset ):

    def __init__(self, img_path: Union[str,Path], segmentation_data: Union[str,Path], alphabet: alphabet.Alphabet, transform: Callable=None ) -> None:
        """ A minimal dataset class for inference.

        + transcription not included in the sample

        :param img_path: charter image path
        :type img_path: Union[Path,str]
        :param segmentation_data: segmentation metadata (XML or JSON)
        :type segmentation_spec: Union[Path, str]
        """
        # str -> path conversion
        img_path = Path( img_path ) if type(img_path) is str else img_path
        segmentation_data = Path( segmentation_data ) if type(segmentation_data) is str else segmentation_data

        # extract line images
        line_extraction_func = seglib.line_images_from_img_segmentation_dict if segmentation_data.suffix == '.json' else seglib.line_images_from_img_xml_files
        self.data = []

        for (img_hwc, mask_hwc) in line_extraction_func( img_path, segmentation_data ):
            # raw lines are median-padded anyway
            print("img.shape =", img_hwc.shape, "mask.shape =", mask_hwc.shape )
            mask_hw = mask_hwc[:,:,0]
            self.data.append( { 'img': mom.MonasteriumDataset.bbox_median_pad( img_hwc, mask_hw, channel_dim=2 ), 
                                'height':img_hwc.shape[0],
                                'width': img_hwc.shape[1],
                               } )
        self.transform = ToTensor() if transform is None else transform
        self.alphabet = alphabet

    def __getitem__(self, index: int):
        sample = self.data[index]
        sample['img'] = self.transform(sample['img'])
        return sample

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    for path in list( args.img_paths):
        print(path)


        stem = re.sub(r'\..+', '',  Path( path ).name )

        # 1. Load segmentation object from PageXML (better: JSON?)
        segmentation_dir = Path( path ).parent

        if args.segmentation_dir != "":
            if Path( args.segmentation_dir ).exists():
                segmentation_dir = Path( args.segmentation_dir )
            else:
                print("Provided segmentation directory {args.segmentation_dir} does not exists!")

        segmentation_object = None

        # Look for existing segmentation file (pattern: <image file stem>.lines.pred.{xml,json})
        segmentation_file_path_candidates = [ segmentation_dir.joinpath(f'{stem}.{args.segmentation_file_suffix}.{format_suffix}') for format_suffix in ('xml',) ]

        # 1. From source (XML, or better: segmentation map), compile ds/dl of tensors
        #    + get line images BBs + polygon masks
        # 
        # 2. Inference requiert:
        #    1. alphabet (taken from Monasterium, passed to the model)
        #    2. image transform (taken from Monasterium)
        #    3. model spec
        #    (1) and (2) suppose building a small test/validate dataset,
        #    with features (alphabet, transforms) taken from Monasterium
        #   + build dataset from bare image samples
        #   + quick-and-dirt getitem()

        for segmentation_file_path in segmentation_file_path_candidates:
            if not segmentation_file_path.exists():
                continue
            segmentation_object = xml.XMLPage( segmentation_file_path ).to_container()
            break
        if segmentation_object is None:
            print("Could not find a proper segmentation data file. Aborting.")
            sys.exit()
        #else:
        #    print("Create segmentation record {} from segmentation data {}".format( segmentation_object, segmentation_file_path ))
         

        # 2. HTR inference
        line_ids =  [ line.id for line in segmentation_object.lines ]

        htr_model = models.load_any( args.model_path )
        pred_it = rpred.rpred( 
                        network = htr_model, 
                        im = Image.open( path ), 
                        bounds = segmentation_object,
                        no_legacy_polygons = args.no_legacy_polygons)

        for line, line_id, record in zip(range(1, len(line_ids)+1), line_ids, pred_it):
            print(f'{line}\t{line_id}\t{record}')


