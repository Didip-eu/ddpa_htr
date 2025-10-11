
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
import inspect
import functools

# 3rd-party
from tqdm import tqdm
import defusedxml.ElementTree as ET
#import xml.etree.ElementTree as ET
from PIL import Image, ImagePath
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
            count (int): Stops after extracting {count} image items (for testing 
                purpose only).
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
            self.page_paths = sorted( self.page_work_folder_path.glob('*{}'.format(lbl_suffix)))
        else:
            self.download_and_extract( self.root_path, self.dataset_resource, extract=extract_pages )
            # copy files from archive folder to page_work_folder
            print('{}, {}'.format(self.archive_root_folder_path, self.page_work_folder_path))
            if self.archive_root_folder_path != self.page_work_folder_path:
                logger.debug("Copying files...")
                self._purge( self.page_work_folder_path )
                for page_lbl_path in self.archive_root_folder_path.glob('*{}'.format( lbl_suffix )):
                    page_img_path = Path(re.sub(r'{}$'.format( lbl_suffix ), img_suffix, str(page_lbl_path)))
                    print(page_img_path)
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
        }

        return



    def build_page_data( self, page_paths, lbl_suffix, img_suffix ):
        """
        Build raw samples (img, page).

        Args:
            page_paths (list[Path]): a list of PageXML files.
            lbl_suffix (str): suffix of annotation file
            img_suffix (str): suffix of image file

        Return:
            list[tuple(Path,Path)]: a list of pairs (<img_file_path>, <annotation_file_path>)
        """
        data = []
        for pp in page_paths:
            img_path = Path(re.sub(r'{}$'.format( lbl_suffix), img_suffix, str(pp) ))
            if img_path.exists():
                data.append( (img_path, pp) )
        return data


    def download_and_extract(
            self,
            root_path: Path,
            fl_meta: dict,
            extract=False) -> None:
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

            print(fl_meta['md5'])

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


    def _build_task( self, 
                   build_items: bool=True, 
                   work_folder: str='', 
                   )->list[dict]:
        """Build the image/GT samples required for an HTR task, either from the raw files (extracted from archive)
        or a work folder that already contains compiled files.

        Args:
            build_items (bool): if True (default), go through the compilation step; otherwise, work from the existing work folder's content.
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                are to be created; default: './MonasteriumHandwritingDatasetHTR'.

        Returns:
            list[dict]: a list of dictionaries.

        Raises:
            FileNotFoundError: the TSV file passed to the `from_line_tsv_file` option does not exist.
        """
        # + image to GT mapping (TSV)
        if work_folder=='':
            self.work_folder_path = Path('data', self.work_folder_name+'HTR') 
            logger.debug("Setting default location for work folder: {}".format( self.work_folder_path ))
        else:
            self.work_folder_path = Path(work_folder)
            logger.debug("Work folder: {}".format( self.work_folder_path ))

        if not self.work_folder_path.is_dir():
            self.work_folder_path.mkdir(parents=True)
            logger.debug("Creating work folder = {}".format( self.work_folder_path ))

        # samples: all of them! 
        if build_items:
            print("Building samples")
            self._extract_lines( self.raw_data_folder_path, self.work_folder_path, )

        logger.info(f"Subset '{subset}' contains {len(data)} samples.")



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
        """This method returns a page sample.

        Args:
            index (int): item index.

        Returns:
            tuple[Tensor,dict]: A tuple containing the image (as a tensor) and its associated target (annotations).
        """
        img_path, label_path = self._data[index]
        image, target = self._load_image_and_target(img_path, label_path)

        if self._transforms:
            print(self._transforms)
            image, target = self._transforms( image, target )
        return (image, target)

    def _load_image_and_target(self, img_path, annotation_path):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
            img_path (Path): image path
            annotation_path (Path): annotation path

        Returns:
            tuple[Image,dict]: A tuple containing the image and a dictionary with 'masks', 'boxes' and 'labels' keys.
        """
        # Open the image file and convert it to RGB
        img = Image.open(img_path)#.convert('RGB')

        page_dict = seglib.segmentation_dict_from_xml( annotation_path, get_text=True ) if (annotation_path.name)[-4:]=='.xml' else json.load( annotation_path )
        masks = Mask( seglib.line_binary_mask_stack_from_segmentation_dict(page_dict, polygon_key=self.polygon_key))
        bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.size[::-1])
        text = '\n'.join( l['text'] for l in page_dict['lines'] )
        return img, {'masks': masks, 'boxes': bboxes, 'path': img_path, 'orig_size': img.size, 'text': text}




    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of pages
        """
        return len(self._data)

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



    def _extract_lines(self, raw_data_folder_path: Path, work_folder_path: Path,) -> list[Dict[str, Union[Tensor,str,int]]]:
        """Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder

        Returns:
            list[Dict[str,Union[Tensor,str,int]]]: An array of dictionaries of the form:: 

                {'img': <absolute img_file_path>,
                 'transcription': <transcription text>,
                 'height': <original height>,
                 'width': <original width>}
        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) 

        if not self.config['resume_task']:
            self._purge( work_folder_path ) 

        cnt = 0 # for testing purpose
        samples = [] 

        for page in tqdm(self.pages):
            line_samples = self.extract_lines_from_page( page, work_folder_path, config=self.config)
            if line_samples:
                samples.extend( line_samples )
                cnt += 1
                if self.config['count'] and cnt == self.config['count']:
                    break
        return samples

    @classmethod
    def extract_lines_from_page(cls, page: Union[str,Path], work_folder_path: Union[Path,str], config:dict={}):

        assert all([ k in config for k in ( 'channel_func', 'resume_task')])

        samples = []
        line_tuples = []

        # replace extra dots in some names (everything that is not in the suffix)
        page_id = re.match(r'(.+).{}'.format(config['lbl_suffix']), page.name).group(1)
        page_id = page_id.replace('.', '_')

        ###################### Case #1: PageXML ###################
        with open(page, 'r') as page_file:

            page_dict = seglib.segmentation_dict_from_xml( page_file, get_text=True ) if config['lbl_suffix']=='.xml' else json.load( page_file )
            
            img_path = Path(page).parent.joinpath( page_dict['image_filename'] )

            page_image = None

            try:
                page_image = Image.open( img_path, 'r')
            except Image.DecompressionBombWarning as dcb:
                logger.debug( f'{dcb}: ignoring page' )
                return None


            for tl in page_dict['lines']:
                sample = dict()
                textline_id, transcription=tl['id'], tl['text']
                polygon_coordinates = [ tuple(pair) for pair in tl['coords'] ]
                line_tuples.append( (sample, textline_id, polygon_coordinates, transcription) )


            for (sample, textline_id, polygon_coordinates, transcription) in line_tuples:
                
                transcription = transcription.replace("\t",' ')
                if len(transcription) == 0:
                    continue
                sample['transcription'] = transcription
                textline_bbox = ImagePath.Path( polygon_coordinates ).getbbox()
                
                x_left, y_up, x_right, y_low = textline_bbox
                sample['width'], sample['height'] = x_right-x_left, y_low-y_up
                
                img_path_prefix = work_folder_path.joinpath( f"{page_id}-{textline_id}" ) 
                sample['img'] = Path(img_path_prefix).with_suffix('.png').absolute()

                if not (config['resume_task'] and sample['img'].exists()):
                    bbox_img = page_image.crop( textline_bbox)
                    img_hwc = np.array( bbox_img )
                    leftx, topy = textline_bbox[:2]
                    transposed_coordinates = np.array([ (x-leftx, y-topy) for x,y in polygon_coordinates ], dtype='int')[:,::-1]

                    boolean_mask = ski.draw.polygon2mask( img_hwc.shape[:2], transposed_coordinates )
                    sample['binary_mask']=img_path_prefix.with_suffix('.bool.npy.gz')
                    with gzip.GzipFile( sample['binary_mask'], 'w') as zf:
                        np.save( zf, boolean_mask )
                    bbox_img.save( sample['img'] )

                    # construct an additional, flat channel
                    if config['channel_func'] is not None:
                        img_channel_hw = config['channel_func']( img_hwc, boolean_mask)
                        sample['img_channel']=img_path_prefix.with_suffix( '.channel.npy.gz' )
                        with gzip.GzipFile(sample['img_channel'], 'w') as zf:
                            np.save( zf, img_channel_hw ) 

                with open( img_path_prefix.with_suffix('.gt.txt'), 'w') as gt_file:
                    gt_file.write( sample['transcription'])

                samples.append( sample )
        return samples

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
