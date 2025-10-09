
# stdlib
import sys
import warnings
import random
import json
import shutil
import re
import os
from pathlib import *
from typing import *

# 3rd-party
from tqdm import tqdm
from PIL import Image, ImagePath
import skimage as ski
import gzip

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms


torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2

class DataException( Exception ):
    """"""
    pass


"""
Utility classes to manage charter data.

"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


class HTRDataset(VisionDataset):
    """A generic dataset class for HTR tasks, with minimal functionalities for accessing 
    ready-made line samples.
    + if the folder contains a TSV, the names of the files to be included are read from it.
    + if the folder has no TSV, all present files are assumed to be in the dataset
    This class does not generate subsets (it is the responsibility of the training class).
    """

    def __init__(self,
                from_line_tsv_file: str='',
                from_work_folder: str='dataset',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = lambda x: x,
                expansion_masks = False,
                channel_func: Callable[[np.ndarray, np.ndarray],np.ndarray]= None,
                channel_suffix: str='',
                line_padding_style: str = None,
                ) -> None:
        """Initialize a dataset instance.

        Args:
            from_line_tsv_file (str): if set, the data are to be loaded from the given file
                (containing folder is assumed to be the work folder, superceding the
                from_work_folder option).
            from_work_folder (str): if set, the samples are to be loaded from the
                given directory, without prior processing.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground
                truth at loading time.
            expansion_masks (bool): if True (default), add transcription expansion offsets
                to the sample if it is present in the XML source line annotations.
            channel_func (Callable): function that takes image and binary polygon mask as inputs,
                and generates an additional channel in the sample. Default: None.
            channel_suffix (str): when loading items from a work folder, which suffix
                to read for the channel file. Default: '' (=ignore channel file).
            line_padding_style (str): When extracting line bounding boxes, padding to be 
                used around the polygon: 'median'=median value of the polygon; 'noise'=random;
                'zero'=0s. The polygon boolean mask is automatically saved on/retrieved from the disk;
                Default is None.
        """

        data = []
        if from_line_tsv_file:
            tsv_path = Path( from_line_tsv_file )
            if tsv_path.exists():
                self.work_folder_path = tsv_path.parent
                # paths are assumed to be absolute
                self.data = self.load_from_tsv( tsv_path, expansion_masks )
                logger.debug("data={}".format( data[:6]))
                #logger.debug("height: {} type={}".format( data[0]['height'], type(data[0]['height'])))
            else:
                raise FileNotFoundError(f'File {tsv_path} does not exist!')

        elif from_work_folder:
            self.work_folder_path = Path(from_work_folder)
            logger.info("Building samples from existing images and transcription files in {}".format(self.work_folder_path))
            self.data = self.load_line_items_from_dir( self.work_folder_path, channel_suffix )

        if not self.data:
            raise DataException("No data found. from_tsv_file={}, from_work_folder={}".format(from_tsv_file, from_work_folder))

        trf = v2.Compose( [ v2.ToDtype(torch.float32, scale=True) ])  
        if transform is not None:
            trf = v2.Compose( [ trf, transform ] ) 
        super().__init__(root=self.work_folder_path, transform=trf, target_transform=target_transform ) # if target_transform else self.filter_transcription)

        if line_padding_style and line_padding_style not in ['noise', 'zero', 'median', 'none']:
            raise ValueError(f"Incorrect padding style: '{line_padding_style}'. Valid styles: 'noise', 'zero', or 'median'.")

        # bbox or polygons and/or masks
        self.config = {
                'from_line_tsv_file': from_line_tsv_file,
                'from_work_folder': from_work_folder,
                'channel_func': channel_func,
                'channel_suffix': channel_suffix,
                'line_padding_style': line_padding_style,
                'expansion_masks': expansion_masks,
        }


    @staticmethod
    def load_line_items_from_dir( work_folder_path: Union[Path,str], channel_suffix:str='' ) -> list[dict]:
        """Construct a list of samples from a directory that has been populated with
        line images and line transcriptions

        Args:
            work_folder_path (Union[Path,str]): a folder containing images (`*.png`), transcription 
                files (`*.gt.txt`) and optional extra channel.
            channel_suffix (str): default suffix for the extra channel ('*.channel.npy.gz')

        Returns:
            list[dict]: a list of samples.
        """
        samples = []
        if type(work_folder_path) is str:
            work_folder_path = Path( work_folder_path )
        for img_file_path in work_folder_path.glob('*.png'):
            sample=dict()
            logger.debug(img_file_path)            
            gt_file_name = img_file_path.with_suffix('.gt.txt')
            sample['img']=img_file_path
            with Image.open( img_file_path, 'r') as img:
                sample['width'], sample['height'] = img.size
            
            with open(gt_file_name, 'r') as gt_if:
                transcription=gt_if.read().rstrip()
                expansion_masks_match = re.search(r'^(.+)<([^>]+)>$', transcription)
                if expansion_masks_match is not None:
                    sample['transcription']=expansion_masks_match.group(1)
                    sample['expansion_masks']=eval(expansion_masks_match.group(2))
                else:
                    sample['transcription']=transcription

            # optional mask
            channel_file_path = img_file_path.with_suffix( channel_suffix )
            if channel_file_path.exists():
                sample['img_channel']=channel_file_path

            samples.append( sample )

        logger.debug(f"Loaded {len(samples)} samples from {work_folder_path}")
        return samples
                

    @staticmethod
    def load_from_tsv(file_path: Path, expansion_masks=False) -> list[dict]:
        """Load samples (as dictionaries) from an existing TSV file. Each input line is a tuple::

            <img file path> <transcription text> <height> <width> [<polygon points>]

        Args:
            file_path (Path): A file path.
            expansion_masks (bool): Load expansion mask field.

        Returns:
            list[dict]: A list of dictionaries of the form::

            {'img': <img file path>,
             'transcription': <transcription text>,
             'height': <original height>,
             'width': <original width>,
            ['img_channel': <2D extra channel> ]
            }

        """
        work_folder_path = file_path.parent
        samples=[]
        logger.debug("work_folder_path={}".format(work_folder_path))
        logger.debug("tsv file={}".format(file_path))
        with open( file_path, 'r') as infile:
            # Detection: 
            # - is the transcription passed as a filepath or as text?
            first_line = next( infile )[:-1]
            img_path, file_or_text, height, width = first_line.split('\t')[:4]
            inline_transcription = False if Path(file_or_text).exists() else True
            # - Is there a field for an extra channel
            has_channel = len(first_line.split('\t')) > 4
            infile.seek(0)

            for tsv_line in infile:
                fields = tsv_line[:-1].split('\t')
                img_file, gt_field, height, width = fields[:4]
                binary_mask_file = work_folder_path.joinpath( img_file ).with_suffix('.bool.npy.gz')

                expansion_masks_match = re.search(r'^(.+)<([^>]+)>$', gt_field)
                if not inline_transcription:
                    with open( work_folder_path.joinpath( gt_field ), 'r') as igt:
                        gt_field = '\n'.join( igt.readlines() )
                elif expansion_masks_match is not None:
                    gt_field = expansion_masks_match.group(1)

                spl = { 'img': work_folder_path.joinpath( img_file ), 'transcription': gt_field,
                        'height': int(height), 'width': int(width) }
                if has_channel:
                    spl['img_channel']=work_folder_path.joinpath(fields[4])
                if expansion_masks and expansion_masks_match is not None:
                    spl['expansion_masks']=eval( expansion_masks_match.group(2))
                if binary_mask_file.exists():
                    spl['binary_mask']=binary_mask_file

                samples.append( spl )
                               
        return samples


    @staticmethod
    def dataset_stats( samples: list[dict] ) -> str:
        """Compute basic stats about sample sets.

        + avg, median, min, max on image heights and widths
        + avg, median, min, max on transcriptions

        Args:
            samples (list[dict]): a list of samples.

        Returns:
            str: a string.
        """
        heights = np.array([ s['height'] for s in samples  ], dtype=int)
        widths = np.array([ s['width'] for s in samples  ], dtype=int)
        gt_lengths = np.array([ len(s['transcription']) for s in samples  ], dtype=int)

        height_stats = [ int(s) for s in(np.mean( heights ), np.median(heights), np.min(heights), np.max(heights))]
        width_stats = [int(s) for s in (np.mean( widths ), np.median(widths), np.min(widths), np.max(widths))]
        gt_length_stats = [int(s) for s in (np.mean( gt_lengths ), np.median(gt_lengths), np.min(gt_lengths), np.max(gt_lengths))]

        stat_list = ('Mean', 'Median', 'Min', 'Max')
        row_format = "{:>10}" * (len(stat_list) + 1)
        return '\n'.join([
            row_format.format("", *stat_list),
            row_format.format("Img height", *height_stats),
            row_format.format("Img width", *width_stats),
            row_format.format("GT length", *gt_length_stats),
        ])


    def _generate_readme( self, filename: str, params: dict )->None:
        """Create a metadata file in the work directory.

        Args:
            filename (str): a filepath.
            params (dict): dictionary of parameters passed to the dataset task builder.

        Returns:
            None
        """
        filepath = Path(self.work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n\n\t+ ' + 
                  '\n+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)
            print( repr(self), file=of)


    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        """Callback function for the iterator. Assumption: the raw sample always contains
        the bounding box image + binary polygon mask. Any combined image (ex. noise-background)
        is constructed from those, _before_ any transform that is passed to the DS constructor.

        Args:
            index (int): item index.

        Returns:
            dict[str,Union[Tensor,int,str]]: a sample dictionary
        """
        img_path = self.data[index]['img']
        
        assert isinstance(img_path, Path) or isinstance(img_path, str)

        sample = self.data[index].copy()
        sample['transcription']=self.target_transform( sample['transcription'] )

        img_array_hwc = ski.io.imread( img_path ) # img path --> img ndarray

        if self.config['line_padding_style'] is not None:
            assert 'binary_mask' in sample and sample['binary_mask'].exists()
            with gzip.GzipFile(sample['binary_mask'], 'r') as mask_in:
                binary_mask_hw = np.load( mask_in )
                padding_func = lambda x, m, channel_dim=2: x
                if self.config['line_padding_style']=='noise':
                    padding_func = self.bbox_noise_pad
                elif self.config['line_padding_style']=='zero':
                    padding_func = self.bbox_zero_pad
                elif self.config['line_padding_style']=='median':
                    padding_func = self.bbox_median_pad
                img_array_hwc = padding_func( img_array_hwc, binary_mask_hw, channel_dim=2 )
                if len(img_array_hwc.shape) == 2: # for ToImage() transf. to work in older torchvision
                    img_array_hwc=img_array_hwc[:,:,None]
        del sample['binary_mask']

        # img ndarray --> tensor
        sample['img']=v2.Compose( [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img_array_hwc)
        logger.debug("Before transform: sample['img'].dtype={}".format( sample['img'].dtype))

        if 'img_channel' in self.data[index]:
            channel_t = None
            if self.data[index]['img_channel'].suffix == '.gz':
                with gzip.GzipFile(self.data[index]['img_channel'], 'r') as channel_in:
                    channel_t = torch.from_numpy( np.load( channel_in ) )/255
            else:
                channel_t = np.load(self.data[index]['img_channel'])/255
            sample['img']=torch.cat( [sample['img'], channel_t[None,:,:]] )

        sample = self.transform( sample )
        sample['id'] = Path(img_path).name

        logger.debug("After transform: sample['img'] has shape {} and type {}".format( sample['img'].shape, sample['img'].dtype))
        return sample

    def __getitems__(self, indexes: list ) -> list[dict]:
        """To help with batching.

        Args:
            indexes (list): a list of indexes.

        Returns:
            list[dict]: a list of samples.
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of data points.
        """
        return len( self.data )


    def __repr__(self) -> str:

        summary = '\n'.join([
                    f"Work folder:\t{self.work_folder_path}",
                    f"Data points:\t{len(self.data)}",
                    "Stats:",
                    f"{self.dataset_stats(self.data)}" if self.data else 'No data',])
        if self.config['from_line_tsv_file']:
             summary += "\nBuilt from TSV input:\t{}".format( self.config['from_line_tsv_file'] )
        
        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")


    @staticmethod
    def bbox_median_pad(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
        """Pad a polygon BBox with the median value of the polygon. Used by
        the line extraction method.

        Args:
            img_chw (np.ndarray): an array (C,H,W). Optionally: (H,W,C)
            mask_hw (np.ndarray): a 2D Boolean mask (H,W).
            param channel_dim (int): the channel dimension: 2 for (H,W,C) images. Default is 0.
        
        Returns:
            np.ndarray: the padded image, with same shape as input.
        """
        img = img_chw.transpose(2,0,1) if (channel_dim == 2 and len(img_chw.shape) > 2) else img_chw
        
        if len(img.shape)==2:
            img = img[None]
        padding_bg = np.zeros( img.shape, dtype=img.dtype)

        for ch in range( img.shape[0] ):
            med = np.median( img[ch][mask_hw] ).astype( img.dtype )
            padding_bg[ch] += np.logical_not(mask_hw) * med
            padding_bg[ch] += img[ch] * mask_hw
        return padding_bg.transpose(1,2,0) if (channel_dim==2 and len(img_chw.shape)>2) else padding_bg[0]

    @staticmethod
    def bbox_noise_pad(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
        """Pad a polygon BBox with noise. Used by the line extraction method.

        Args:
            img_chw (np.ndarray): an array (C,H,W). Optionally: (H,W,C) or (H,W)
            mask_hw (np.ndarray): a 2D Boolean mask (H,W).
            channel_dim (int): the channel dimension: 2 for (H,W,C) images. Default is 0.

        Returns:
            np.ndarray: the padded image, with same shape as input.
        """
        img = img_chw.transpose(2,0,1) if (channel_dim == 2 and len(img_chw.shape) > 2) else img_chw
        padding_bg = np.random.randint(0, 255, img.shape, dtype=img_chw.dtype)
        
        padding_bg *= np.logical_not(mask_hw) 
        if len(img.shape)>2:
            mask_hw = np.stack( [ mask_hw, mask_hw, mask_hw ] )

        padding_bg += img * mask_hw
        return padding_bg.transpose(1,2,0) if (channel_dim==2 and len(img.shape) > 2) else padding_bg

    @staticmethod
    def bbox_zero_pad(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
        """Pad a polygon BBox with zeros. Used by the line extraction method.

        Args:
            img_chw (np.ndarray): an array (C,H,W). Optionally: (H,W,C)
            mask_hw (np.ndarray): a 2D Boolean mask (H,W).
            channel_dim (int): the channel dimension: 2 for (H,W,C) images. Default is 0.

        Returns:
            np.ndarray: the padded image, with same shape as input.
        """
        img = img_chw.transpose(2,0,1) if (channel_dim == 2 and len(img_chw.shape) > 2) else img_chw
        if len(img.shape)>2:
            mask_hw = np.stack( [ mask_hw, mask_hw, mask_hw ] )
        img_out = img * mask_hw
        return img_out.transpose(1,2,0) if (channel_dim==2 and len(img.shape) > 2) else img_out


class PadToWidth():
    """Pad an image to desired length."""

    def __init__( self, max_w ):
        self.max_w = max_w

    def __call__(self, sample: dict) -> dict:
        """Transform a sample: only the image is modified, not the nominal height and width.
        """
        t_chw, w = [ sample[k] for k in ('img', 'width' ) ]
        if w > self.max_w:
            warnings.warn("Cannot pad an image that is wider ({}) than the padding size ({})".format( w, self.max_w))
            return sample
        new_t_chw = torch.zeros( t_chw.shape[:2] + (self.max_w,))
        new_t_chw[:,:,:w] = t_chw

        transformed_sample = sample.copy()
        transformed_sample.update( {'img': new_t_chw } )
        return transformed_sample



class ResizeToHeight():
    """Resize an image with fixed height, preserving aspect ratio as long as the resulting width
    does not exceed the specified max. width. If that is the case, the image is horizontally
    squeezed to fix this.

    """

    def __init__( self, target_height, max_width ):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, sample: dict) -> dict:
        """Transform a sample

           + resize 'img' value to desired height
           + modify 'height' and 'width' accordingly

        """
        t_chw, h, w = [ sample[k] for k in ('img', 'height', 'width') ]
        # freak case (marginal annotations): original height is the larger
        # dimension -> specify the width too
        if h > w:
            t_chw = v2.Resize( size=(self.target_height, int(w*self.target_height/h) ), antialias=True)( t_chw )
        # default case: original height is the smaller dimension and
        # gets picked up by Resize()
        else:
            t_chw = v2.Resize(size=self.target_height, antialias=True)( t_chw )
            
        if t_chw.shape[-1] > self.max_width:
            t_chw = v2.Resize(size=(self.target_height, self.max_width), antialias=True)( t_chw )
        h_new, w_new = [ int(d) for d in t_chw.shape[1:] ]

        transformed_sample = sample.copy()
        transformed_sample.update( {'img': t_chw, 'height': h_new, 'width': w_new } )

        return transformed_sample
        


def dummy():
    """"""
    return True
