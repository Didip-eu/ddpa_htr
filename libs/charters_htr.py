
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
import torchvision.transforms as transforms

from . import download_utils as du

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
    """A generic dataset class for charters, equipped with a rich set of methods for HTR tasks:

        * region and line/transcription extraction methods (from original page images and XML metadata)
        * commonly-used transforms, for use in getitem()

        Attributes:
            dataset_resource (dict): meta-data (URL, archive name, type of repository).

            root_folder_basename (str): A basename for the root folder, that contains
                * the archive, if the dataset is to be downloaded
                * the subfolder that is created from it (with page data)

            work_folder_name (str): The work folder where line samples (images and transcriptions) are
                to be extracted. If it is not passed to the constructor, a default path 
                `dataset/<work_folder_name>` is created in the current directory.
    """

    dataset_resource = None

    def __init__( self,
                root: str='./data',
                work_folder: str = './dataset/htr_line_dataset', 
                from_page_dir: str = '',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = lambda x: x,
                extract_pages: bool = False,
                build_items: bool = True,
                channel_func: Callable[[np.ndarray, np.ndarray],np.ndarray]= None,
                channel_suffix: str='',
                count: int = 0,
                line_padding_style: str = None,
                resume_task: bool = False,
                gt_suffix: str = 'xml'
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data' in this project's directory.
            work_folder (str): Where line images and ground truth transcriptions 
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
            build_items (bool): if True (default), extract and store images for the task
                from the pages; otherwise, just extract the original data from the archive.
            count (int): Stops after extracting {count} image items (for testing 
                purpose only).
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)
            gt_suffix (str): 'xml' for PageXML (default) or valid, unique suffix of JSON file.
                Ex. 'htr.gt.json'

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_dir or from_line_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_dir, -from_work_folder, -from_line_tsv_file")

        trf = v2.Compose( [ v2.ToDtype(torch.float32, scale=True) ])  
        if transform is not None:
            trf = v2.Compose( [ trf, transform ] ) 
        super().__init__(root, transform=trf, target_transform=target_transform ) # if target_transform else self.filter_transcription)

        self.root = Path(root) if root else Path(__file__).parents[1].joinpath('data', self.root_folder_basename)

        logger.debug("Root folder: {}".format( self.root ))
        if not self.root.exists():
            self.root.mkdir( parents=True )
            logger.debug("Create root path: {}".format(self.root))

        self.raw_data_folder_path = None
        self.work_folder_path = None # task-dependent

        # Local file system with data samples, no archive
        if from_work_folder != '':
            work_folder = from_work_folder
            logger.debug("work_folder="+ work_folder)
            if not Path(work_folder).exists():
                raise FileNotFoundError(f"Work folder {self.work_folder_path} does not exist. Abort.")
            
        # Local file system with raw page data, no archive 
        elif from_page_dir != '':
            self.raw_data_folder_path = Path( from_page_dir )
            if not self.raw_data_folder_path.exists():
                raise FileNotFoundError(f"Directory {self.raw_data_folder_path} does not exist. Abort.")
            self.pages = sorted( self.raw_data_folder_path.glob('*.{}'.format(gt_suffix)))

        # Online archive
        elif self.dataset_resource is not None:
            # tarball creates its own base folder
            self.raw_data_folder_path = self.root.joinpath( self.dataset_resource['tarball_root_name'] )
            self.download_and_extract( self.root, self.root, self.dataset_resource, extract_pages )
            # input PageXML files are at the root of the resulting tree
            #        (sorting is necessary for deterministic output)
            self.pages = sorted( self.raw_data_folder_path.glob('*.{}'.format(gt_suffix))) 
        else:
            raise FileNotFoundError("Could not find a dataset source!")

        # bbox or polygons and/or masks
        self.config = {
                'channel_func': channel_func,
                'channel_suffix': channel_suffix,
                'count': count,
                'resume_task': resume_task,
                'gt_suffix': gt_suffix,
        }


        self.data = []

        # when loading a set from compiled samples, just build all 3 CSV files
        if from_work_folder and not from_line_tsv_file:
            for ss in ('train', 'validate', 'test'):
                if ss==subset:
                    continue
                data = self._build_task(build_items=False, work_folder=work_folder, subset=ss )
                self.dump_data_to_tsv(data, Path(self.work_folder_path.joinpath(f"charters_ds_{ss}.tsv")) )

        self.data = self._build_task(build_items=build_items, work_folder=work_folder, subset=subset )
        if self.data and not from_line_tsv_file:
            # Generate a TSV file with one entry per img/transcription pair
            self.dump_data_to_tsv(self.data, Path(self.work_folder_path.joinpath(f"charters_ds_{subset}.tsv")) )

            channel_function_def = ''
            if channel_func is not None:
                if type(channel_func) is functools.partial:
                    channel_function_def = inspect.getsource(channel_func.func) + str( channel_func.keywords)
                else:
                    channel_function_def = inspect.getsource( channel_func )

            self._generate_readme("README.md", 
                    { 'subset': subset,
                      'subset_ratios': subset_ratios, 
                      'build_items': build_items, 
                      'count': count, 
                      'from_line_tsv_file': from_line_tsv_file,
                      'from_page_dir': from_page_dir,
                      'from_work_folder': from_work_folder,
                      'work_folder': work_folder, 
                      'line_padding_style': line_padding_style,
                      'channel_func': channel_function_def,
                      'channel_suffix': channel_suffix,
                     } )


    def download_and_extract(
            self,
            root: Path,
            raw_data_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root (Path): where to save the archive raw_data_folder_path (Path): where to extract the archive.
            fl_meta (dict): a dictionary with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
            extract (bool): If False (default), skip archive extraction step.

        Returns:
            None

        Raises:
            OSError: the base folder does not exist.
        """
        output_file_path = None
        # downloadable archive
        if 'url' in fl_meta:
            output_file_path = root.joinpath( fl_meta['tarball_filename'])

            if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
                logger.info("Downloading archive...")
                du.resumable_download(fl_meta['url'], root, fl_meta['tarball_filename'], google=(fl_meta['origin']=='google'))
            else:
                logger.info("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_resource['md5']))
        elif 'file' in fl_meta:
            output_file_path = Path(fl_meta['file'])

        if not raw_data_folder_path.exists() or not raw_data_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract: # and du.check_extracted( raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ) , fl_meta['full-md5'] ):
            logger.info('Found valid file tree in {}: skipping the extraction stage.'.format(str(raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ))))
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )


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
        page_id = re.match(r'(.+).{}'.format(config['gt_suffix']), page.name).group(1)
        page_id = page_id.replace('.', '_')

        ###################### Case #1: PageXML ###################
        with open(page, 'r') as page_file:

            page_dict = seglib.segmentation_dict_from_xml( page_file, get_text=True ) if ['gt_suffix']=='xml' else json.load( page_file )
            
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



    @abstractmethod
    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        """This method generate a line sample.

        Args:
            index (int): item index.

        Returns:
            dict[str,Union[Tensor,int,str]]: a sample dictionary
        """
        pass

    @abstractmethod
    def __getitems__(self, indexes: list ) -> list[dict]:
        """To help with batching lines.

        Args:
            indexes (list): a list of indexes.

        Returns:
            list[dict]: a list of samples.
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of data points.
        """
        pass


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
                    f"Files extracted in:\t{self.raw_data_folder_path}",
                    f"Work folder:\t{self.work_folder_path}",
                    f"Data points:\t{len(self.data)}",
                    "Stats:",
                    f"{self.dataset_stats(self.data)}" if self.data else 'No data',])
        
        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")


class MonasteriumDataset(ChartersDataset):
    """A subset of Monasterium charter images and their meta-data (PageXML).

        + its core is a set of charters segmented and transcribed by various contributors, mostly by correcting Transkribus-generated data.
        + it has vocation to grow through in-house, DiDip-produced transcriptions.
    """

    dataset_resource = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'tarball_filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
            'tarball_root_name': 'MonasteriumTekliaGTDataset',
            'comment': 'A clean, terse dataset, with no use of Unicode abbreviation marks.',
    }

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)


class KoenigsfeldenDataset(ChartersDataset):
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




class KoenigsfeldenDatasetAbbrev(ChartersDataset):
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


class NurembergLetterbooks(ChartersDataset):
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
