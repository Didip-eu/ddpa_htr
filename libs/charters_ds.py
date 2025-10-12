
# stdlib
import sys
import warnings
import random
import tarfile
import json
import shutil
import re
import os
from pathlib import *
from typing import *
import time

# 3rd-party
from tqdm import tqdm
from PIL import Image
import skimage as ski
import gzip

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms as transforms

from . import download_utils as du
from . import seglib

#from . import alphabet, character_classes.py

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2

class DataException( Exception ):
    """"""
    pass


"""
Utility classes to manage charter data, provided as PageXML.

"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)



class PageDataset(VisionDataset):
    """A generic dataset class for charters, equipped with page-wide functionalities:

        * page augmentation functionalities
        * region and line/transcription extraction methods (from original page images and XML metadata)

        File-management logic:

        - a <root> folder where archives are saved and decompressed (set at initialization time, with a reasonable default in the current location)
        - a page work folder
            - defaults to the root directory of the tarball, in last resort
            - OR: implied from <from_page_dir> option, when loading data from existing pages
            - OR: specified at initialization time through the <page_work_folder> parameter - archive is checked; after extraction, page files
                  are copied in the work folder
        - [optional] a cache for augmented pages
        - a line work folder where line samples extracted from the page are to be saved

        Depending on the options passed at initialization time, the page work folder may be under <root> or not.


        Attributes:
            dataset_resource (dict): meta-data (URL, archive name, type of repository).

    """

    dataset_resource = None

    def __init__( self,
                root: str='./data',
                page_work_folder: str = '',
                line_work_folder: str = './dataset/htr_line_dataset', 
                from_page_dir: str = '',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = lambda x: x,
                extract_pages: bool = False,
                dry_run: bool = False,
                channel_func: Callable[[np.ndarray, np.ndarray],np.ndarray]= None,
                channel_suffix: str='',
                count: int = 0,
                line_padding_style: str = None,
                resume_task: bool = False,
                lbl_suffix: str = '.xml',
                img_suffix: str = '.jpg',
                polygon_key: str = 'coords',
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data' in this project's directory.
            page_work_folder (str): Where page images and XML annotations are to be extracted.
            line_work_folder (str): Where line images and ground truth transcriptions 
                are to be created; default: './dataset/htr_line_dataset';
            from_page_dir (str): if set, the samples have to be extracted from the
                raw page data contained in the given directory. GT metadata are either
                JSON files or PageXML.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground
                truth at loading time.
            extract_pages (bool): if True, extract the archive's content into the base
                folder no matter what; otherwise (default), check first for a file tree 
                with matching name and checksum.
            channel_func (Callable): function that takes image and binary polygon mask as inputs,
                and generates an additional channel in the sample. Default: None.
            channel_suffix (str): when loading items from a work folder, which suffix
                to read for the channel file. Default: '' (=ignore channel file).
            dry_run (bool): if True (default), compute all paths (root, page_work_folder, line_work_folder)
                but does not write anything.
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)
            lbl_suffix (str): '.xml' for PageXML (default) or valid, unique suffix of JSON file.
                Ex. '.htr.gt.json'
            img_suffix (str): image suffix. Default: '.jpg'

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_dir or from_line_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_dir, -from_work_folder, -from_line_tsv_file")

        self._transforms = transform if transform is not None else v2.ToImage()
        self.polygon_key = polygon_key
        self.data = []

        self.root_path = Path(root) 
        self.archive_root_folder_path = self.root_path.joinpath( self.dataset_resource['tarball_root_name'] )
        self.page_work_folder_path = self.archive_root_folder_path
        if from_page_dir:
            self.page_work_folder_path = Path(from_page_dir) 
        elif page_work_folder:
            self.page_work_folder_path = Path(page_work_folder)
        self.line_work_folder_path = Path(line_work_folder)

        logger.info(f"""
        Dataset creation (dry run):
            root (prefix of archive tree): {self.root_path}
            archive path: {self.root_path.joinpath(self.dataset_resource['tarball_filename'])}
            archive_root_folder: {self.archive_root_folder_path}
            page_work_folder (pages): {self.page_work_folder_path} (exists: {self.page_work_folder_path.exists()})
            line_work_folder (lines): {self.line_work_folder_path} (exists: {self.line_work_folder_path.exists()})
            """)
        if dry_run:
            return

        # Folder creation, when needed
        if from_page_dir != '' and not self.page_work_folder_path.exists():
           raise FileNotFoundError(f"Work folder {self.page_work_folder_path} does not exist. Abort.")
        else:
            self.root_path.mkdir( parents=True, exist_ok=True )
            self.page_work_folder_path.mkdir( parents=True, exist_ok=True)

        page_paths = []
        # Assume pages are already there = ignore archive
        if from_page_dir:
            page_paths = sorted( self.page_work_folder_path.glob('*{}'.format(lbl_suffix)))
        else:
            self.download_and_extract( self.root_path, self.dataset_resource, extract=extract_pages )
            # copy files from archive folder to page_work_folder
            if self.archive_root_folder_path != self.page_work_folder_path:
                logger.debug("Copying files...")
                self._purge( self.page_work_folder_path )
                for page_lbl_path in self.archive_root_folder_path.glob('*{}'.format( lbl_suffix )):
                    page_img_path = Path(re.sub(r'{}$'.format( lbl_suffix ), img_suffix, str(page_lbl_path)))
                    if page_img_path.exists():
                        shutil.copyfile( page_lbl_path, self.page_work_folder_path.joinpath( page_lbl_path.name ) )
                        logger.info("Copying {} to work folder {}...".format( page_lbl_path, self.page_work_folder_path))
                        shutil.copyfile( page_img_path, self.page_work_folder_path.joinpath( page_img_path.name) )
                        logger.info("Copying {} to work folder {}...".format( page_img_path, self.page_work_folder_path))

            page_paths = sorted( self.page_work_folder_path.glob('*{}'.format(lbl_suffix))) 

        if not page_paths:
            raise FileNotFoundError("Could not find a dataset source!")

        self._data = self.build_page_data( page_paths, lbl_suffix, img_suffix )

        self.config = {
                'count': count,
                'resume_task': resume_task,
                'lbl_suffix': lbl_suffix,
                'img_suffix': img_suffix,
        }


    def build_page_data( self, page_paths, lbl_suffix, img_suffix ):
        """
        Build raw samples (img, page), ensuring that every image has its annotation counterpart.

        Args:
            page_paths (list[Path]): a list of PageXML files.
            lbl_suffix (str): suffix of annotation file
            img_suffix (str): suffix of image file

        Return:
            list[tuple(Path,Path)]: a list of pairs (<img_file_path>, <annotation_file_path>)
        """
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        data = []
        for pp in page_paths:
            img_path = Path(re.sub(r'{}$'.format( lbl_suffix), img_suffix, str(pp) ))
            if not img_path.exists():
                continue

            img = None
            try:
                img = Image.open(img_path)#.convert('RGB')
            except Image.DecompressionBombWarning as dcb:
                logger.error( f'{dcb}: ignoring page {pp}' )
                continue
            data.append( (img_path, pp) )
        return data


    def download_and_extract( self, root_path: Path, fl_meta: dict, extract=False) -> None:
        """Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root_path (Path): where to save and extract the archive
            fl_meta (dict): a dictionary with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
            extract (bool): If False (default), skip archive extraction step.

        Raises:
            OSError: the base folder does not exist.
        """
        output_file_path = None
        # downloadable archive
        if 'url' in fl_meta:
            output_file_path = root_path.joinpath( fl_meta['tarball_filename'])

            if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
                logger.info("Downloading archive...")
                du.resumable_download(fl_meta['url'], root_path, fl_meta['tarball_filename'], google=(fl_meta['origin']=='google'))
            else:
                logger.info("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_resource['md5']))
        elif 'file' in fl_meta:
            output_file_path = Path(fl_meta['file'])

        archive_root_folder_path = root_path.joinpath( self.dataset_resource['tarball_root_name'] ) 
        # skip if archive already extracted (unless explicit override)
        if not extract: # and du.check_extracted( raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ) , fl_meta['full-md5'] ):
            logger.info("Skipping the extraction stage ('extract' option=F). You should check whether {} contains a valid archive tree.".format( archive_root_folder_path ))
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( root_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( root_path )


    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        """This method returns a page sample.

        Args:
            index (int): item index.

        Returns:
            tuple[Tensor,dict]: A tuple containing the image (as a tensor) and its associated target (annotations).
        """
        img_path, label_path = self._data[index]
        img_whc, target = self._load_image_and_target(img_path, label_path)

        if self._transforms:
            img_chw, target = self._transforms( img_whc, target )
        return (img_chw, target)

    def _load_image_and_target(self, img_path, annotation_path):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
            img_path (Path): image path
            annotation_path (Path): annotation path

        Returns:
            tuple[Image,dict]: A tuple containing the image and a dictionary with 'mask', 'boxes' and 'labels' keys.
        """
        img_whc = Image.open(img_path, 'r')
        
        start_time = time.time()
        page_dict = seglib.segmentation_dict_from_xml( annotation_path, get_text=True ) if (annotation_path.name)[-4:]=='.xml' else json.load( annotation_path )
        # one mask per page is enough (!= Mask-RCNN)
        start_time = time.time()
        bboxes_n4, masks_nhw = seglib.line_masks_from_img_segmentation_dict( img_whc, page_dict, polygon_key=self.polygon_key) 
        bboxes_n4, masks_nhw = torch.tensor(bboxes_n4), torch.tensor(masks_nhw)
        texts = [ l['text'] for l in page_dict['lines'] ]
        return img_whc, {'mask': Mask(torch.sum(masks_nhw, axis=0)), 'boxes': bboxes_n4, 'path': img_path, 'orig_size': img_whc.size, 'texts': texts}


    @staticmethod
    def augment_with_bboxes( sample, aug, device ):
        """  Augment a sample (img + masks), and add bounding boxes to the target.
        (For Tormentor only.)

        Args:
            sample (tuple[Tensor,dict]): tuple with image (as tensor) and label dictionary.
        """
        img, target = sample
        img = img.to(device)
        img = aug(img)
        masks, labels, texts = target['masks'].to(device), target['labels'].to(device), target['texts']
        masks = torch.stack( [ aug(m, is_mask=True) for m in target['masks'] ], axis=0).to(device)

        # first, filter empty masks
        keep = torch.sum( masks, dim=(1,2)) > 10
        masks, labels = masks[keep], labels[keep]
        # construct boxes, filter out invalid ones
        boxes=BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.shape)
        keep=(boxes[:,0]-boxes[:,2])*(boxes[:,1]-boxes[:,3]) != 0

        target['boxes'], target['masks'], target['texts'] = boxes[keep], masks[keep], texts[keep]
        return (img, target)


    def dump_lines( self, line_as_tensor=False, resume=False ):
        """
        Save line samples in the line work folder; each line yields:
        - line crop (as PNG, by default)
        - polygon mask (as tensor)
        - transcription text (text file)
        
        Args:
            line_as_tensor (bool): save line crops as tensors (compressed).
            resume (bool): resume a dump task---work folder is checked for existing, completed pages.

        """
        self.line_work_folder_path.mkdir( parents=True, exist_ok=True)

        if not resume:
            self._purge( self.line_work_folder_path )
        start = 0
        sentinel_path = self.line_work_folder_path.joinpath('.sentinel')
        if sentinel_path.exists():
            with open(sentinel_path, 'r') as sf:
                start = int(sf.read())
        for idx in tqdm( range(len(self))):
            if idx < start:
                continue
            img_chw, annotation = self[idx]
            img_prefix = re.sub(r'{}'.format( self.config['img_suffix']), '', annotation['path'].name)
            sentinel_path = self.line_work_folder_path.joinpath(f'{img_prefix}.sentinel')
            for idx, box in enumerate( annotation['boxes'] ):
                l,t,r,b = [ int(elt.item()) for elt in box ]
                line_tensor = img_chw[:,t:b+1, l:r+1]
                mask_tensor = annotation['mask'][t:b+1, l:r+1]
                line_text = annotation['texts'][idx]
                outfile_prefix = self.line_work_folder_path.joinpath( f"{img_prefix}-{idx}")
                if not line_as_tensor:
                    ski.io.imsave( f'{outfile_prefix}.png', line_tensor.permute(1,2,0)) 
                else:
                    with gzip.GzipFile( f'{outfile_prefix}.pt.gz', 'w') as zf:
                        torch.save( line_tensor, zf )
                with gzip.GzipFile( f'{outfile_prefix}.bool.pt.gz', 'w') as zf:
                    torch.save( mask_tensor, zf )
                with open( f'{outfile_prefix}.gt.txt', 'w') as of:
                    of.write( line_text )
            # indicates that this page dump is complete (for resuming task)
            with open( sentinel_path, 'w') as sf:
                sf.write(f'{idx}')
        
        if sentinel_path.exists():
            sentinel_path.unlink()


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of pages
        """
        return len(self._data)

    def dataset_stats(self) -> str:
        """Compute basic stats about sample sets.

        + avg, median, min, max on image heights and widths


        Returns:
            str: a string.
        """
        heights, widths, gt_lengths = [], [], []
        for img, lbl in tqdm( self ):
            widths.append( lbl['orig_size'][0] )
            heights.append( lbl['orig_size'][1] )
            gt_lengths.append( np.sum( [ len(txt) for txt in lbl['texts'] ]))

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
        filepath = Path(self.page_work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n\n\t+ ' + 
                  '\n+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)
            print( repr(self), file=of)



    def _purge(self, folder: str) -> int:
        """Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the TSV file.

        Args:
            folder (str): Name of the subfolder to _purge (relative the caller's pwd

        Returns:
            int: number of deleted files.
        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
            item.unlink()
            cnt += 1
        return cnt

    def __repr__(self) -> str:

        summary = '\n'.join([
                    f"Root folder:\t{self.root}",
                    f"Files extracted in:\t{self.archive_root_folder_path}",
                    f"Page_Work folder:\t{self.page_work_folder_path}",
                    f"Data points:\t{len(self.data)}",
                    "Stats:",
                    f"{self.dataset_stats(self.data)}" if self.data else 'No data',])
        
        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")



class MonasteriumDataset(PageDataset):
    """A subset of Monasterium charter images and their meta-data (PageXML).

        + its core is a set of charters segmented and transcribed by various contributors, mostly by correcting Transkribus-generated data.
        + it has vocation to grow through in-house, DiDip-produced transcriptions.
    """

    dataset_resource = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'tarball_filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            #'md5': '337929f65c52526b61d6c4073d08ab79',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
            'tarball_root_name': 'MonasteriumTekliaGTDataset',
            'comment': 'A clean, terse dataset, with no use of Unicode abbreviation marks.',
    }

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)
#

class KoenigsfeldenDataset(PageDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662',
            'comment': 'Transcriptions have been cleaned up (removal of obvious junk or non-printable characters, as well a redundant punctuation marks---star-shaped unicode symbols); unicode-abbreviation marks have been expanded.',
    }

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription




class KoenigsfeldenDatasetAbbrev(PageDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662_abbrev.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662_abbrev',
            'comment': 'Similar to the KoenigsfeldenDataset, with a notable difference: Unicode abbreviations have been kept.',
    }

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription


class NurembergLetterbooks(PageDataset):
    """
    Nuremberg letterbooks (15th century).
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/nuremberg_letterbooks/nuremberg_letterbooks.tar.gz",
            'tarball_filename': 'nuremberg_letterbooks.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Nuremberg letterbooks ground truth data',
            'origin': 'local',
            'tarball_root_name': 'nuremberg_letterbooks',
            'comment': 'Numerous struck-through lines (masked)'
    }

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)




def dummy():
    """"""
    return True
