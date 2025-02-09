
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
Utility classes to manage charter data.

"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)



class ChartersDataset(VisionDataset):
    """A generic dataset class for charters, equipped with a rich set of methods for HTR tasks:

        * region and line/transcription extraction methods (from original page images and XML metadata)
        * commonly-used transforms, for use in getitem()

        Attributes:
            dataset_resource (dict): meta-data (URL, archive name, type of repository).

            work_folder_name (str): The work folder is where a task-specific instance of the data is created; if it not
                passed to the constructor, a default path is constructed, using this default name.

            root_folder_basename (str): A basename for the root folder, that contains
                * the archive
                * the subfolder that is created from it.
                * work folders for specific tasks
                By default, a folder named ``data/<root_folder_basename>`` is created in the *project directory*,
                if no other path is passed to the constructor.

    """

    dataset_resource = None

    work_folder_name = "ChartersHandwritingDataset"

    root_folder_basename="Charters"

    def __init__( self,
                root: str='',
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = lambda x: x,
                extract_pages: bool = False,
                from_line_tsv_file: str = '',
                from_page_xml_dir: str = '',
                from_work_folder: str = '',
                build_items: bool = True,
                expansion_masks = False,
                channel_func: Callable[[np.ndarray, np.ndarray],np.ndarray]= None,
                count: int = 0,
                line_padding_style: str = None,
                resume_task: bool = False
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data/Charters' in this project's directory.
            work_folder (str): Where line images and ground truth transcriptions fitting a
                particular task are to be created; default: '<root>/ChartersHandwritingDatasetHTR';
                if parameter is a relative path, the work folder is created under
                <root>; an absolute path overrides this.
            subset (str): 'train' (default), 'validate' or 'test'.
            subset_ratios (Tuple[float, float, float]): ratios for respective ('train', 
                'validate', ...) subsets
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground
                truth at loading time.
            extract_pages (bool): if True, extract the archive's content into the base
                folder no matter what; otherwise (default), check first for a file tree 
                with matching name and checksum.
            expansion_masks (bool): if True (default), add transcription expansion offsets
                to the sample if it is present in the XML source line annotations.
            channel_func (Callable): function that takes image and binary polygon mask as inputs,
                and generates an additional channel in the sample. Default: None.
            build_items (bool): if True (default), extract and store images for the task
                from the pages; otherwise, just extract the original data from the archive.
            from_line_tsv_file (str): if set, the data are to be loaded from the given file
                (containing folder is assumed to be the work folder, superceding the
                work_folder option).
            from_page_xml_dir (str): if set, the samples have to be extracted from the 
                raw page data contained in the given directory.
            from_work_folder (str): if set, the samples are to be loaded from the 
                given directory, without prior processing.
            count (int): Stops after extracting {count} image items (for testing 
                purpose only).
            line_padding_style (str): When extracting line bounding boxes, padding to be 
                used around the polygon: 'median' pads with the median value of
                the polygon; 'noise' pads with random noise; with either choice, the 
                polygon boolean mask is automatically saved on/retrieved from the disk;
                None is the default.
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_xml_dir or from_line_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_xml_dir, -from_work_folder, -from_line_tsv_file")

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

        self.from_line_tsv_file = ''
        if from_line_tsv_file == '':
            # Local file system with data samples, no archive
            if from_work_folder != '':
                work_folder = from_work_folder
                logger.debug("work_folder="+ work_folder)
                if not Path(work_folder).exists():
                    raise FileNotFoundError(f"Work folder {self.work_folder_path} does not exist. Abort.")
                
            # Local file system with raw page data, no archive 
            elif from_page_xml_dir != '':
                self.raw_data_folder_path = Path( from_page_xml_dir )
                if not self.raw_data_folder_path.exists():
                    raise FileNotFoundError(f"Directory {self.raw_data_folder_path} does not exist. Abort.")
                self.pagexmls = sorted( self.raw_data_folder_path.glob('*.xml'))

            # Online archive
            elif self.dataset_resource is not None:
                # tarball creates its own base folder
                self.raw_data_folder_path = self.root.joinpath( self.dataset_resource['tarball_root_name'] )
                self.download_and_extract( self.root, self.root, self.dataset_resource, extract_pages )
                # input PageXML files are at the root of the resulting tree
                #        (sorting is necessary for deterministic output)
                self.pagexmls = sorted( self.raw_data_folder_path.glob('*.xml'))
            else:
                raise FileNotFoundError("Could not find a dataset source!")
        else:
            # used only by __str__ method
            self.from_line_tsv_file = from_line_tsv_file

        # bbox or polygons and/or masks
        self.config = {
                'channel_func': channel_func,
                'line_padding_style': line_padding_style,
                'count': count,
                'resume_task': resume_task,
                'from_line_tsv_file': from_line_tsv_file,
                'subset': subset,
                'subset_ratios': subset_ratios,
                'expansion_masks': expansion_masks,
        }


        self.data = []

        build_ok = False if (from_line_tsv_file!='' or from_work_folder!='' ) else build_items
        
        self._build_task(build_items=build_ok, work_folder=work_folder )

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
                      'from_page_xml_dir': from_page_xml_dir,
                      'from_work_folder': from_work_folder,
                      'work_folder': work_folder, 
                      'line_padding_style': line_padding_style,
                      'channel_func': channel_function_def,
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
                   )->None:
        """From the read-only, uncompressed archive files, build the image/GT files required for the task at hand:

        + only creates the files needed for a particular task (train, validate, or test): if more than one subset
          is needed, just initialize a new dataset with desired parameters (work directory, subset)
        + by default, 'train' subset contains 70% of the samples, 'validate', 10%, and 'test', 20%.
        + set samples are randomly picked, but two subsets are guaranteed to be complementary.

        Args:
            build_items (bool): if True (default), extract and store
                images for the task from the pages;
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                are to be created; default: './MonasteriumHandwritingDatasetHTR'.

        Returns:
            None

        Raises:
            FileNotFoundError: the TSV file passed to the `from_line_tsv_file` option does not exist.
        """
        from_line_tsv_file, line_padding_style, expansion_masks = [ self.config[k] for k in ('from_line_tsv_file', 'line_padding_style', 'expansion_masks')]
        # create from existing TSV files - passed directory that contains:
        # + image to GT mapping (TSV)
        if from_line_tsv_file != '':
            tsv_path = Path( from_line_tsv_file )
            if tsv_path.exists():
                self.work_folder_path = tsv_path.parent
                # paths are assumed to be absolute
                self.data = self.load_from_tsv( tsv_path, expansion_masks )
                logger.debug("data={}".format( self.data[:6]))
                #logger.debug("height: {} type={}".format( self.data[0]['height'], type(self.data[0]['height'])))
            else:
                raise FileNotFoundError(f'File {tsv_path} does not exist!')

        else:
            if work_folder=='':
                self.work_folder_path = Path(self.root, self.work_folder_name+'HTR') 
                logger.debug("Setting default location for work folder: {}".format( self.work_folder_path ))
            else:
                # if work folder is an absolute path, it overrides the root
                self.work_folder_path = self.root.joinpath( work_folder )
                logger.debug("Work folder: {}".format( self.work_folder_path ))

            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir(parents=True)
                logger.debug("Creating work folder = {}".format( self.work_folder_path ))

            # samples: all of them! (Splitting into subsets happens in an ulterior step.)
            if build_items:
                samples = self._extract_lines( self.raw_data_folder_path, self.work_folder_path, on_disk=build_items, )
                                                
                                                
            else:
                logger.info("Building samples from existing images and transcription files in {}".format(self.work_folder_path))
                samples = self.load_line_items_from_dir( self.work_folder_path )

            self.data = self._split_set( samples, ratios=self.config['subset_ratios'], subset=self.config['subset'])
            logger.info(f"Subset contains {len(self.data)} samples.")



    @staticmethod
    def load_line_items_from_dir( work_folder_path: Union[Path,str] ) -> List[dict]:
        """Construct a list of samples from a directory that has been populated with
        line images and line transcriptions

        Args:
            work_folder_path (Union[Path,str]): a folder containing images (`*.png`),
            transcription files (`*.gt.txt`) and optional extra channel ('*.channel.npy.gz')

        Returns:
            List[dict]: a list of samples.
        """
        logger.debug('In function')
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
            channel_file_path = img_file_path.with_suffix('.channel.npy.gz')
            if channel_file_path.exists():
                sample['img_channel']=channel_file_path

            samples.append( sample )

        logger.debug(f"Loaded {len(samples)} samples from {work_folder_path}")
        return samples
                

    @staticmethod
    def load_from_tsv(file_path: Path, expansion_masks=False) -> List[dict]:
        """Load samples (as dictionaries) from an existing TSV file. Each input line is a tuple::

            <img file path> <transcription text> <height> <width> [<polygon points>]

        Args:
            file_path (Path): A file path (relative to the caller's pwd).
            expansion_masks (bool): Load expansion mask field.

        Returns:
            List[dict]: A list of dictionaries of the form::

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

    def _extract_lines(self, raw_data_folder_path: Path,
                        work_folder_path: Path, 
                        on_disk=False,) -> List[Dict[str, Union[Tensor,str,int]]]:
        """Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the
                caller's pwd).
            on_disk (bool): If False (default), samples are only built in memory; otherwise line images 
                and transcriptions are written into the work folder.
                around the polygon.

        Returns:
            List[Dict[str,Union[Tensor,str,int]]]: An array of dictionaries of the form:: 

                {'img': <absolute img_file_path>,
                 'transcription': <transcription text>,
                 'height': <original height>,
                 'width': <original width>}
        """
        logger.debug("_extract_lines()")

        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) 

        if not self.config['resume_task']:
            self._purge( work_folder_path ) 

        cnt = 0 # for testing purpose
        samples = [] 

        for page in tqdm(self.pagexmls):

            #img_path = Path( raw_data_folder_path, f'{xml_id}.jpg')
            #logger.debug( img_path )

            line_samples = self.extract_lines_from_pagexml(page, 
                        on_disk_folder=work_folder_path if on_disk else None, config=self.config) 
                        
            if line_samples:
                samples.extend( line_samples )
                cnt += 1
                if self.config['count'] and cnt == self.config['count']:
                    break
        return samples

    @classmethod
    def extract_lines_from_pagexml(cls, page: Union[str,Path], on_disk_folder: Union[Path,str]=None, config:dict={}):

        assert all([ k in config for k in ('line_padding_style', 'channel_func', 'resume_task', 'expansion_masks')])

        samples = []
        xml_id = Path( page ).stem

        with open(page, 'r') as page_file:

            page_tree = ET.parse( page_file )
            ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
            page_root = page_tree.getroot()
            page_elt = page_root.find('.//pc:Page', ns)
            imageFilename = page_elt.get('imageFilename')

            img_path = Path(page).parent.joinpath( imageFilename )

            page_image = None

            if on_disk_folder:
                try:
                    page_image = Image.open( img_path, 'r')
                except Image.DecompressionBombWarning as dcb:
                    logger.debug( f'{dcb}: ignoring page' )
                    return None
            
            for textline_elt in page_root.findall( './/pc:TextLine', ns ):

                sample = dict()
                textline_id=textline_elt.get("id")
                transcription_element = textline_elt.find('./pc:TextEquiv', ns)
                if transcription_element is None:
                    continue
                transcription_text_element = transcription_element.find('./pc:Unicode', ns)
                if transcription_text_element is None:
                    continue
                transcription = transcription_text_element.text
                if not transcription or re.match(r'\s+$', transcription):
                    continue
                transcription = transcription.replace("\t",' ')
                sample['transcription'] = transcription

                if config['expansion_masks'] and 'custom' in textline_elt.keys():
                    sample['expansion_masks'] = [ (int(o), int(l)) for (o,l) in re.findall(r'expansion *{ *offset:(\d+); *length:(\d+);', textline_elt.get('custom')) ]

                polygon_string=textline_elt.find('./pc:Coords', ns).get('points')
                coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                textline_bbox = ImagePath.Path( coordinates ).getbbox()
            
                x_left, y_up, x_right, y_low = textline_bbox
                sample['width'], sample['height'] = x_right-x_left, y_low-y_up
            
                img_path_prefix = on_disk_folder.joinpath( f"{xml_id}-{textline_id}" ) if on_disk_folder is not None else f"{xml_id}-{textline_id}"
                sample['img'] = Path(img_path_prefix).with_suffix('.png')

                if on_disk_folder is not None:
                    if not Path(on_disk_folder).is_dir():
                        raise FileNotFoundError("Abort. Check that directory {} exists.")

                    bbox_img = page_image.crop( textline_bbox )

                    if not (config['resume_task'] and sample['img'].exists()):
                        # Image -> (H,W,C)-uint8 array
                        img_hwc = np.array( bbox_img )
                        # 2D Boolean polygon mask, from points
                        leftx, topy = textline_bbox[:2]
                        transposed_coordinates = np.array([ (x-leftx, y-topy) for x,y in coordinates ], dtype='int')[:,::-1]
                        boolean_mask = ski.draw.polygon2mask( img_hwc.shape[:2], transposed_coordinates )
                        
                        if config['line_padding_style'] is not None:
                            sample['binary_mask']=img_path_prefix.with_suffix('.bool.npy.gz')
                            with gzip.GzipFile( sample['binary_mask'], 'w') as zf:
                                np.save( zf, boolean_mask ) 
                    
                        bbox_img.save( sample['img'] )
                         
                        # construct an additional, flat channel
                        if config['channel_func'] is not None:
                            img_channel_hw = config['channel_func']( img_hwc, boolean_mask)
                            sample['img_channel']=img_path_prefix.with_suffix('.channel.npy.gz')
                            #sample['img_channel']=img_path_prefix.with_suffix('.channel.npy')
                            with gzip.GzipFile(sample['img_channel'], 'w') as zf:
                                np.save( zf, img_channel_hw ) 
                            #np.save( sample['img_channel'], img_channel_hw)

                    with open( img_path_prefix.with_suffix('.gt.txt'), 'w') as gt_file:
                        gt_file.write( sample['transcription'])
                        if 'expansion_masks' in sample:
                            gt_file.write( '<{}>'.format( sample['expansion_masks']))

                samples.append( sample )

        return samples
    

    @staticmethod
    def dump_data_to_tsv(samples: List[dict], file_path: str='', all_path_style=False) -> None:
        """Create a CSV file with all tuples (`<line image absolute path>`, `<transcription>`, `<height>`, `<width>` `[<polygon points]`).
        Height and widths are the original heights and widths.

        Args:
            samples (List[dict]): dataset samples.
            file_path (str): A TSV (absolute) file path (Default value = '')
            all_path_style (bool): list GT file name instead of GT content. (Default value = False)

        Returns:
            None
        """
        if file_path == '':
            for sample in samples:
                # note: TSV only contains the image file name (load_from_tsv() takes care of applying the correct path prefix)
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                logger.debug("{}\t{}\t{}\t{}".format( img_path, 
                      gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'), int(height), int(width)))
            return
        with open( file_path, 'w' ) as of:
            for sample in samples:
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                #logger.debug('{}\t{}'.format( img_path, gt, height, width ))
                if 'expansion_masks' in sample and sample['expansion_masks'] is not None:
                    gt = gt + '<{}>'.format( sample['expansion_masks'] )
                of.write( '{}\t{}\t{}\t{}'.format( img_path,
                                             gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'),
                                             int(height), int(width) ))
                if 'img_channel' in sample and sample['img_channel'] is not None:
                    of.write('\t{}'.format( sample['img_channel'].name ))
                of.write('\n')
                                            

    @staticmethod
    def dataset_stats( samples: List[dict] ) -> str:
        """Compute basic stats about sample sets.

        + avg, median, min, max on image heights and widths
        + avg, median, min, max on transcriptions

        Args:
            samples (List[dict]): a list of samples.

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



    @staticmethod
    def _split_set(samples: object, ratios: Tuple[float, float, float], subset: str) -> List[object]:
        """Split a dataset into 3 sets: train, validation, test.

        Args:
            samples (object): any dataset sample.
            ratios (Tuple[float, float, float]): respective proportions for possible subsets
            subset (str): subset to be build  ('train', 'validate', or 'test')

        Returns:
            List[object]: a list of samples.

        Raises:
            ValueError: The subset type does not exist.
        """

        random.seed(10)
        logger.debug("Splitting set of {} samples with ratios {}".format( len(samples), ratios))

        if 1.0 in ratios:
            return list( samples )
        if subset not in ('train', 'validate', 'test'):
            raise ValueError("Incorrect subset type: choose among 'train', 'validate', and 'test'.")

        subset_2_count = int( len(samples)* ratios[1])
        subset_3_count = int( len(samples)* ratios[2] )

        subset_1_indices = set( range(len(samples)))
        
        if ratios[1] != 0:
            subset_2_indices = set( random.sample( subset_1_indices, subset_2_count))
            subset_1_indices -= subset_2_indices

        if ratios[2] != 0:
            subset_3_indices = set( random.sample( subset_1_indices, subset_3_count))
            subset_1_indices -= subset_3_indices

        if subset == 'train':
            return [ samples[i] for i in subset_1_indices ]
        if subset == 'validate':
            return [ samples[i] for i in subset_2_indices ]
        if subset == 'test':
            return [ samples[i] for i in subset_3_indices ]


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

        # In the sample, image filename replaced with 
        # - file id ('id')
        # - tensor ('img')
        # - dimensions of transformed image ('height' and 'width')
        # 
        sample = self.data[index].copy()
        sample['transcription']=self.target_transform( sample['transcription'] )
        img_array_hwc = ski.io.imread( img_path )

        if self.config['line_padding_style'] is not None and 'binary_mask' in sample and sample['binary_mask'].exists():
            with gzip.GzipFile(sample['binary_mask'], 'r') as mask_in:
                binary_mask_hw = np.load( mask_in )
                padding_func = self.bbox_noise_pad if self.config['line_padding_style']=='noise' else self.bbox_median_pad
                img_array_hwc = padding_func( img_array_hwc, binary_mask_hw, channel_dim=2 )
                print(img_array_hwc.shape)

        # np.ndarrays are not picked up by v2.transforms when in dict: hence the conversion
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

    def __getitems__(self, indexes: list ) -> List[dict]:
        """To help with batching.

        Args:
            indexes (list): a list of indexes.

        Returns:
            List[dict]: a list of samples.
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of data points.
        """
        return len( self.data )


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
        if self.from_line_tsv_file:
             summary += "\nBuilt from TSV input:\t{}".format( self.from_line_tsv_file )
        
#        prototype_alphabet = alphabet.Alphabet.prototype_from_data_samples( 
#                list(itertools.chain.from_iterable( character_classes.charsets )),
#                self.data ) if data else None
#
#        if prototype_alphabet is not None:
#            summary += f"\n + A prototype alphabet generated from this subset would have {len(prototype_alphabet)} codes." 
#        
#            symbols_shared = self.alphabet.symbol_intersection( prototype_alphabet )
#            symbols_only_here, symbols_only_prototype = self.alphabet.symbol_differences( prototype_alphabet )
#
#
#            summary += f"\n + Dataset alphabet shares {len(symbols_shared)} symbols with a data-generated charset."
#            summary += f"\n + Dataset alphabet and a data-generated charset are identical: {self.alphabet == prototype_alphabet}"
#            if symbols_only_here:
#                summary += f"\n + Dataset alphabet's symbols that are not in a data-generated charset: {symbols_only_here}"
#            if symbols_only_prototype:
#                summary += f"\n + Data-generated charset's symbols that are not in the dataset alphabet: {symbols_only_prototype}"

        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")


    def count_line_items(self, folder) -> Tuple[int, int]:
        """Count dataset items in the given folder.

        Args:
            folder (str): a directory path.

        Returns:
            Tuple[int,int]: a pair `(<number of GT files>, <number of image files>)`
        """
        return (
                len( [ i for i in Path(folder).glob('*.gt.txt') ] ),
                len( [ i for i in Path(folder).glob('*.png') ] )
                )


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
        img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
        padding_bg = np.zeros( img.shape, dtype=img.dtype)

        for ch in range( img.shape[0] ):
            med = np.median( img[ch][mask_hw] ).astype( img.dtype )
            padding_bg[ch] += np.logical_not(mask_hw) * med
            padding_bg[ch] += img[ch] * mask_hw
        return padding_bg.transpose(1,2,0) if channel_dim==2 else padding_bg

    @staticmethod
    def bbox_noise_pad(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
        """Pad a polygon BBox with noise. Used by the line extraction method.

        Args:
            img_chw (np.ndarray): an array (C,H,W). Optionally: (H,W,C)
            mask_hw (np.ndarray): a 2D Boolean mask (H,W).
            channel_dim (int): the channel dimension: 2 for (H,W,C) images. Default is 0.

        Returns:
            np.ndarray: the padded image, with same shape as input.
        """
        img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
        padding_bg = np.random.randint(0, 255, img.shape, dtype=img_chw.dtype)
        
        padding_bg *= np.logical_not(mask_hw) 
        mask_chw = np.stack( [ mask_hw, mask_hw, mask_hw ] )
        padding_bg += img * mask_chw
        return padding_bg.transpose(1,2,0) if channel_dim==2 else padding_bg

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
        img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
        mask_chw = np.stack( [ mask_hw, mask_hw, mask_hw ] )
        img_out = img * mask_chw
        return img_out.transpose(1,2,0) if channel_dim == 2 else img_out


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

        # add a field
        mask = torch.zeros( new_t_chw.shape, dtype=torch.bool)
        mask[:,:,:w] = 1

        transformed_sample = sample.copy()
        transformed_sample.update( {'img': new_t_chw, 'mask': mask } )
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

    work_folder_name="MonasteriumHandwritingDataset"

    root_folder_basename="Monasterium"

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

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="Koenigsfelden"
    "This is the root of the archive tree."

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

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="KoenigsfeldenAbbrev"
    "This is the root of the archive tree."

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

    work_folder_name="NurembergLetterbooksDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="NurembergLetterbooks"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)




def dummy():
    """"""
    return True
