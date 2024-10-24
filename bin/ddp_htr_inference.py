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
from PIL import Image

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


class InferenceDataSet( Dataset ):

    def __init__(self, transform:Callable=None, img_path: Path, segmentation_spec: Path, alphabet: alphabet.Alphabet ) -> None:
        """ A minimal dataset class for inference.

        + transcription not included in the sample

        :param img_path: charter image path
        :type img_path: Path
        :param segmentation_spec: segmentation metadata (XML or JSON)
        :type segmentation_spec: Path
        """
        self.data = []
        for (img_path, mask_hw) in seglib.line_images_from_img_xml_files( img_path, page_xml ):
            img_
            # raw lines are median-padded anyway
            self.data.append( { 'img': Monasterium.bbox_median_pad( img_chw, mask_hw ), 
                                'height':img_chw.shape[1],
                                'width': img_chw.shape[-1],
                               } )
        self.transform = None
        self.alphabet = alphabet

    def __getitem__(self, index: int):
        img_chw, mask_hw = self.data[index]

        return self.transform( sample )




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


