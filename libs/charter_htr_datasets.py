"""
htr_datasets.py
nprenet@gmail.com, 11/2025

Utility classes to support the HTR workflow, from charter data (images + metadata) to line samples.
"""

# stdlib
import sys
import warnings
import tarfile
import json
import shutil
import re
import os
from pathlib import Path
from typing import Callable, Union, Optional

# 3rd-party
from tqdm import tqdm
from PIL import Image
import skimage as ski
import gzip
import pandas

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms as transforms

# local
from . import download_utils as du
from . import seglib
from . import transforms as tsf
#from . import alphabet, character_classes.py

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2



import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


class DataException( Exception ):
    """"""
    pass

class PageDataset(VisionDataset):
    """A generic dataset class for charters, equipped with page-wide functionalities:

    * region and line/transcription extraction methods (from original page images and XML metadata)
    * page/region augmentation functionalities

    File-management logic relies on several locations:

    - a **root** folder where archives are saved and decompressed (set at initialization time, with a 
      reasonable default in the current location); past this step, all write operations happen in 
      the work folders described below.
    - a **page work folder**
      + defaults to the root directory of the tarball, in last resort
      + OR: implied from <from_page_files> or <from_region_files> option, when loading data from existing
        pages or regions
      + OR: implied from <from_page_folder> option, when loading data from existing page folder
      + OR: specified/created through the <page_work_folder> parameter - archive is checked;
        after extraction, page files are copied in the work folder
    - [optional] a cache for augmented pages? 
    - a **line work folder** where line samples extracted from the page are to be saved: it is a parameter
      to the line serialization routine.

    Remarks:

    - Depending on the options passed at initialization time, the page work folder may be under <root> or not.
    - So far, we assume that there is a one-to-one relationship between PageXML files and an image. In practice,
      this is not always true for all external datasets, but it is a reasonable assumption for a previous curation
      step takes care of this.

    TODO:
    - ability to extract line polygons with scaled height (JSON-only)? The segmentation app 'ddpa_lines_ng' already
      provides this, either as a library function or a stand-alone script; adding this feature would introduce 
      dependencies that better to the segmentation tools.
    - ability to cache augmented pages (Tormentor augmentations are _very_ costly) as tensors? Since the pages/regions
      themselves are not used for training, this is not very useful; the augmentation is used only once, in order
      to generate the lines.


    Attributes:
        dataset_resource (dict): meta-data (URL, archive name, type of repository).
    """

    dataset_resource = {
            'file': '',
            'tarball_filename': '-',
            'md5': '-',
            'desc': 'Constructed from files.',
            'origin': 'local',
            'tarball_root_name': '-',
            'comment': '',
    }

    def __init__( self,
                root: str='./data',
                page_work_folder: str = '',
                line_work_folder: str = './dataset/htr_line_dataset', 
                from_page_folder: str = '',
                from_page_files: list[Path]=[],
                from_region_files: list[Path]=[],
                transform: Optional[Callable] = None,
                augmentation_class: Callable= None,
                extract_pages: bool = False,
                dry_run: bool = False,
                limit: int = 0,
                resume_task: bool = False,
                lbl_suffix: str = '.xml',
                img_suffix: str = '.jpg',
                device: str = 'cpu',
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data' in this project's directory.
            page_work_folder (str): Where page images and XML annotations are to be extracted.
            from_page_folder (str): if set, the samples have to be extracted from the
                raw page data contained in the given directory. GT metadata are either
                JSON files or PageXML.
            from page_files (list[Path]): if set, supersedes previous options; all files must
                be contained in the same folder.
            from region_files (list[Path]): if set, supersedes previous options; all files must
                be contained in the same folder.
            transform (Callable): A v2.transform to apply to the PIL image at loading time.
            augmentation_class (Callable[tuple[Tensor,dict]]): any transform that takes a tensor as an input
                and cannot be safely composed as a v2.transform. Eg. Tormentor augmentation
            extract_pages (bool): if True, extract the archive's content into the base
                folder no matter what; otherwise (default), check first for a file tree 
                with matching name and checksum.
            dry_run (bool): if True (default), compute all paths (root, page_work_folder, line_work_folder)
                but does not write anything.
            limit (int): stop the compilation after <limit> charters, if <limit> > 0. Default: 0.
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)
            lbl_suffix (str): '.xml' for PageXML (default) or valid, unique suffix of JSON file.
                Ex. '.htr.gt.json'
            img_suffix (str): image suffix. Default: '.jpg'

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_folder or from_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_folder, -from_work_folder, -from_tsv_file")

        self._transforms = v2.Compose([
                            v2.ToImage(),
                            v2.ToDtype( torch.float32, scale=True),])
        if transform is not None:
            self._transforms = v2.Compose([ self._transforms, tranform] )
        self.augmentation_class = augmentation_class
        self.device = device

        self._data = []

        self.root_path = Path(root) 
        self.archive_root_folder_path = self.root_path.joinpath( self.dataset_resource['tarball_root_name'] )
        self.page_work_folder_path = self.archive_root_folder_path

        from_page_files = list(from_page_files)
        from_region_files = list(from_region_files)

        if from_region_files or from_page_files:
            dataset_dirs = set( [ img.parent for img in (from_region_files if from_region_files else from_page_files) ] )
            if len(dataset_dirs) > 1:
                raise Exception(f'Source files should belong to the same directory (found {len(dataset_dirs)} parent folders: {[str(f) for f in dataset_dirs]}.')
            self.page_work_folder_path = list(dataset_dirs)[0]
        elif from_page_folder:
            self.page_work_folder_path = Path(from_page_folder) 
        elif page_work_folder:
            self.page_work_folder_path = Path(page_work_folder)

        logger.debug(self)
        if dry_run:
            logger.info('Dry run: exiting.')
            return

        # Folder creation, when needed
        if from_page_folder != '' and not self.page_work_folder_path.exists():
           raise FileNotFoundError(f"Work folder {self.page_work_folder_path} does not exist. Abort.")
        elif not (from_page_files or from_region_files):
            self.root_path.mkdir( parents=True, exist_ok=True )
            self.page_work_folder_path.mkdir( parents=True, exist_ok=True)

        image_paths = []
        # Assume pages are already there = ignore archive
        if from_region_files:
            image_paths = sorted( from_region_files )
        elif from_page_files:
            image_paths = sorted( from_page_files )
        elif from_page_folder:
            image_paths = sorted( self.page_work_folder_path.glob('*{}'.format(img_suffix)))
        else:
            self.download_and_extract( self.root_path, self.dataset_resource, extract=extract_pages )
            # copy files from archive folder to page_work_folder
            if self.archive_root_folder_path != self.page_work_folder_path:
                logger.debug("Copying files...")
                self._purge( self.page_work_folder_path )
                for page_img_path in self.archive_root_folder_path.glob('*{}'.format( img_suffix )):
                    lbl_img_path = Path(re.sub(r'{}$'.format( img_suffix ), lbl_suffix, str(page_img_path)))
                    if page_lbl_path.exists():
                        shutil.copyfile( page_img_path, self.page_work_folder_path.joinpath( page_img_path.name) )
                        logger.debug("Copying {} to work folder {}...".format( page_img_path, self.page_work_folder_path))
                        shutil.copyfile( page_lbl_path, self.page_work_folder_path.joinpath( page_lbl_path.name ) )
                        logger.debug("Copying {} to work folder {}...".format( page_lbl_path, self.page_work_folder_path))

            image_paths = sorted( self.page_work_folder_path.glob('*{}'.format(img_suffix))) 

        if not image_paths:
            raise FileNotFoundError("Could not find a dataset source!")

        if from_region_files:
            self._data = [ (ip, Path( re.sub(r'\.png$', '.json', str(ip)))) for ip in image_paths ]
        else:
            self._data = self.build_page_region_data( image_paths, img_suffix, lbl_suffix, limit=limit )

        self.config = {
                'resume_task': resume_task,
                'lbl_suffix': lbl_suffix,
                'img_suffix': img_suffix,
                'reg_img_suffix': '.png',
        }

        logger.info(self)


    def build_page_region_data( self, image_paths, img_suffix, lbl_suffix, limit=0):
        """
        Build and save data samples (img, label), with a 1-to-many relationship between original image 
        and samples:
        + ensure that every image has its annotation counterpart.
        + a sample is a page region and a corresponding line dictionary (stored as json)

        Args:
            image_paths (list[Path]): a list of page image files.
            img_suffix (str): suffix of page image file
            lbl_suffix (str): suffix of page annotation file
            limit (int): stop the compilation after <limit> charters

        Return:
            list[tuple(Path,Path)]: a list of pairs (<region_img_file_path.png>, <annotation_file_path.json>)
        """
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        data = []
        logger.info("Building region data items (this might take a while).")

        for idx, ip in enumerate( tqdm( image_paths )):

            if limit and idx >= limit:
                break
            img = None
            try:
                img = Image.open(ip)
            except Image.DecompressionBombWarning as dcb:
                logger.error( f'{dcb}: ignoring page {ip}' )
                continue

            lbl_path = Path(re.sub(r'{}$'.format( img_suffix ), lbl_suffix, str(ip) ))
            if not lbl_path.exists():
                continue

            page_dict = {}
            if (lbl_path.name)[-4:]=='.xml':
                page_dict = seglib.segmentation_dict_from_xml( lbl_path, get_text=True )
            elif (lbl_path.name)[-5:]=='json':
                with open( lbl_path ) as jsonf:
                    page_dict = json.load( jsonf )

            regions = {}
            for reg in page_dict['regions']:
                # - crop and name image
                reg_dict = {}
                img_prefix = ip.with_suffix('')
                reg_dict['image_filename'] = re.sub(r'{}'.format(img_suffix), f"-{reg['id']}.png", str(ip))
                reg_dict['bbox_ltrb'] = [ *reg['coords'][0], *reg['coords'][2] ]
                reg_dict['image_width'] = reg_dict['bbox_ltrb'][2]-reg_dict['bbox_ltrb'][0]
                reg_dict['image_height'] = reg_dict['bbox_ltrb'][3]-reg_dict['bbox_ltrb'][1] 
                reg_dict['lines']=[]
                regions[ reg['id'] ]=reg_dict
            for line in page_dict['lines']:
                outer_reg = regions[ line['regions'][0] ]
                polyg_array, baseline_array = np.array( line['coords'] ), np.array( line['baseline'] )
                # shifting crop coordinates
                line['coords'] = (polyg_array - outer_reg['bbox_ltrb'][:2] ).tolist()
                line['baseline'] = (baseline_array - outer_reg['bbox_ltrb'][:2] ).tolist()
                del line['regions']
                outer_reg['lines'].append( line )
            for r in regions.values():
                if not r['lines']:
                    continue
                new_img = img.crop( r['bbox_ltrb'] )
                new_img.save( r['image_filename'])
                new_lbl_path = Path(r['image_filename']).with_suffix('.json')
                with open( new_lbl_path, 'w') as jsonf:
                    del r['bbox_ltrb']
                    jsonf.write( json.dumps(r, indent=4) )

                data.append( (Path(r['image_filename']), new_lbl_path) )
        return data

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        """This method returns a page sample.

        Args:
            index (int): item index.

        Returns:
            tuple[Tensor,dict]: A tuple containing the image (as a tensor) and its associated target (annotations).
        """
        img_path, label_path = self._data[index]
        img_whc, label = self._load_image_and_label(img_path, label_path)

        if self._transforms:
            img_chw, target = self._transforms( img_whc, label )
        if self.augmentation_class:
            img_chw, label = self.augment_with_bboxes( (img_chw, label), self.augmentation_class(), device=self.device)
        return (img_chw, label)

    def _load_image_and_label(self, img_path:Path, annotation_path:Path, augmentation:Callable=None):
        """
        Load an image and its target (bounding boxes and masks). Note that at this point, the
        data samples are made of regions crops and corresponding JSON labels. The XML switch
        has been kept anyway, for easy reversal to page-wide processing.

        Parameters:
            img_path (Path): image path
            annotation_path (Path): annotation path
            augmentation (Callable): suitable augmentation function

        Returns:
            tuple[Image,dict]: A tuple containing the image and a dictionary with 'masks', 'boxes' and 'labels' keys.
        """
        img_whc = Image.open(img_path, 'r')
        
        page_dict = {}
        if (annotation_path.name)[-4:]=='.xml':
            page_dict = seglib.segmentation_dict_from_xml( annotation_path, get_text=True ) 
        elif (annotation_path.name)[-5:]=='.json':
            with open( annotation_path ) as jsonf:
                page_dict = json.load( jsonf )
        masks_nhw = Mask( seglib.line_binary_mask_stack_from_segmentation_dict( page_dict )) 
        bboxes_n4 = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks_nhw), format='xyxy', canvas_size=img_whc.size[::-1])
        texts = [ l['text'] for l in page_dict['lines'] ]

        return img_whc, {'masks': masks_nhw, 'boxes': bboxes_n4, 'path': img_path, 'orig_size': img_whc.size, 'texts': texts}


    @staticmethod
    def augment_with_bboxes( sample, augmentation: Callable, device='cpu' ):
        """  Augment a sample (img + masks), and add bounding boxes to the target.
        (For Tormentor only.)

        Args:
            sample (tuple[Tensor,dict]): tuple with image (as tensor) and label dictionary.
            augmentation (Callable): a Tormentor-style augmentation function.

        Returns:
            Tensor: the transformed image.
            dict: the labels, with the appropriate changes on the masks and bounding boxes.
        """
        img, label = sample
        img_chw = sample[0].to(device)
        img_chw = augmentation(img_chw)
        masks, texts = label['masks'].to(device), label['texts']
        masks = torch.stack( [ augmentation(m, is_mask=True) for m in label['masks'] ], axis=0).to(device) 

        # construct boxes, filter out invalid ones
        boxes=BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img_chw.shape)
        keep=(boxes[:,0]-boxes[:,2])*(boxes[:,1]-boxes[:,3]) != 0
        
        
        label['boxes'], label['masks'], label['texts'] = boxes[keep], masks[keep], [ t for (t,k) in zip(label['texts'], keep) if k ]
        return (img_chw.cpu(), label)


    def dump_lines( self, line_work_folder: Union[str,Path], overwrite_existing=False, line_as_tensor=False, resume=False, iteration=-1 ):
        """
        Save line samples in the line work folder; each line yields:
        - line crop (as PNG, by default)
        - polygon mask (as tensor)
        - transcription text (text file)
        
        Args:
            line_work_folder (Union[str,Path]): where to serialize the lines.
            overwrite_existing (bool): write over existing line files.
            line_as_tensor (bool): save line crops as tensors (compressed).
            resume (bool): resume a dump task---work folder is checked for existing, completed pages.
            iteration (int): integer suffix to be added to sample filename (when generating randomly augmented samples).
        """
        line_work_folder_path = Path( line_work_folder )
        line_work_folder_path.mkdir( parents=True, exist_ok=True)

        # when this routine is called only once per region
        if iteration==-1:
            if not overwrite_existing and len(list(line_work_folder_path.glob('*'))):
                logger.info("Line work folder {} is not empty: check or set 'overwrite_existing=True")
                return
            if not resume:
                self._purge( line_work_folder_path )
        start = 0
        sentinel_path = line_work_folder_path.joinpath('.sentinel')
        if sentinel_path.exists():
            with open(sentinel_path, 'r') as sf:
                start = int(sf.read())
        # each page is loaded (and augmented) here
        for page_idx in tqdm( range(len(self))):
            if page_idx < start:
                continue
            img_chw, annotation = self[page_idx]
            img_prefix = re.sub(r'{}'.format( self.config['reg_img_suffix']), '', Path(annotation['path']).name)
            for line_idx, box in enumerate( annotation['boxes'] ):
                l,t,r,b = [ int(elt.item()) for elt in box ]
                line_tensor = (img_chw.cpu().numpy())[:,t:b+1, l:r+1]
                # compressed arrays << compressed tensors
                mask_array = (annotation['masks'][line_idx,t:b+1, l:r+1]).cpu().numpy()
                line_text = annotation['texts'][line_idx]
                outfile_prefix = line_work_folder_path.joinpath( f"{img_prefix}-l{line_idx}")
                if iteration >= 0:
                    outfile_prefix = f"{outfile_prefix}-i{iteration}"
                    print("outfile_prefix={}".format( outfile_prefix ))
                if not line_as_tensor:
                    line_array = (line_tensor.transpose(1,2,0)*255).astype('uint8')
                    ski.io.imsave( f'{outfile_prefix}.png', line_array )
                else:
                    with gzip.GzipFile( f'{outfile_prefix}.pt.gz', 'w') as zf:
                        torch.save( line_tensor, zf )
                with gzip.GzipFile( f'{outfile_prefix}.bool.npy.gz', 'w') as zf:
                    np.save( zf, mask_array )
                with open( f'{outfile_prefix}.gt.txt', 'w') as of:
                    of.write( line_text )
            # indicates that this page dump is complete (for resuming task)
            with open( sentinel_path, 'w') as sf:
                sf.write(f'{page_idx}')
        
        if sentinel_path.exists():
            sentinel_path.unlink()

        logger.info("Compiled {} lines".format( len(list(line_work_folder_path.glob('*.png')))))


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of pages
        """
        return len(self._data)

    def statistics(self) -> str:
        """Compute basic stats about sample sets.

        + avg, median, min, max on image heights and widths


        Returns:
            str: a string.
        """
        heights, widths, page_gt_lengths, line_gt_lengths = [], [], [], []
        for img, lbl in tqdm( self._data ):
            page_dict = {}
            if (lbl.name)[-4:]=='.xml':
                page_dict = seglib.segmentation_dict_from_xml( lbl, get_text=True ) 
            elif (lbl.name)[-5:]=='.json':
                with open( lbl ) as jsonf:
                    page_dict = json.load( jsonf )
            widths.append( int(page_dict['image_width'] ))
            heights.append( int(page_dict['image_height'] ))
            page_gt_lengths.append( np.sum( [ len(line['text']) for line in page_dict['lines'] ]))
            line_gt_lengths.extend( [ len(line['text']) for line in page_dict['lines'] ])

        height_stats = [ int(s) for s in(np.mean( heights ), np.median(heights), np.min(heights), np.max(heights))]
        width_stats = [int(s) for s in (np.mean( widths ), np.median(widths), np.min(widths), np.max(widths))]
        page_gt_length_stats = [int(s) for s in (np.mean( page_gt_lengths ), np.median(page_gt_lengths), np.min(page_gt_lengths), np.max(page_gt_lengths))]
        line_gt_length_stats = [int(s) for s in (np.mean( line_gt_lengths ), np.median(line_gt_lengths), np.min(line_gt_lengths), np.max(line_gt_lengths))]

        stat_list = ('Mean', 'Median', 'Min', 'Max')
        row_format = "{:>20}"+ ("{:>10}" * len(stat_list))
        return '\n'.join([
            row_format.format("", *stat_list),
            row_format.format("Img height", *height_stats),
            row_format.format("Img width", *width_stats),
            row_format.format("GT length (page)", *page_gt_length_stats),
            row_format.format("GT length (line)", *line_gt_length_stats),
        ])


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
        return f"""
                Root path:\t{self.root_path}
                Archive path:\t{self.root_path.joinpath( self.dataset_resource['tarball_filename'])}
                Archive root folder:\t{self.archive_root_folder_path}
                Page_work folder:\t{self.page_work_folder_path}
                Data points:\t{len(self._data)}
                """


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


## Specific page-wide datasets
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




class HTRLineDataset(VisionDataset):
    """A generic dataset class for HTR training tasks, with minimal functionalities for accessing ready-made 
    line samples (eg. as generated from PageDataset class) in a location that is either:

    + a folder path: all files in it are then included in the dataset.
    + a list of files in a common location → for building any dataset on-the-fly.
    + a TSV file: its parent folder is assumed to contain the sample files; the TSV lists those entries that
      are to be included in the dataset.

    The first two options allow serializing the dataset into a TSV, for later reuse.

    The class loads and manipulates line samples, where each sample is a dictionary with
    { line_img, polygon mask, target }

    What this class does _not_ do:

    + compile regions or lines out of PageXML and images
    + apply geometrical transformations on polygons (polygons are handled by the PageDataset class;
      at this stage, we know only about pixel masks)
    + augment data at page or region level
    + construct train, validation and test subsets (this the the training script's responsibility)

    What it could do:

    - take a PageDataset as a source
    """

    def __init__(self,
                from_tsv_file: str='',
                from_work_folder: str='dataset',
                from_line_files: list[Path]=[],
                img_suffix: str='.png',
                gt_suffix: str='.gt.txt',
                to_tsv_file: str='',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = lambda x: x,
                expansion_masks = False,
                channel_func: Callable[[np.ndarray, np.ndarray],np.ndarray]= None,
                channel_suffix: str='',
                padding_style: str = 'median',
                ) -> None:
        """Initialize a dataset instance.

        Args:
            from_tsv_file (str): if set, the data are to be loaded from the given file
                (containing folder is assumed to be the work folder, superceding the
                from_work_folder option).
            from_work_folder (str): if set, the samples are to be loaded from the
                given directory.
            from_line_files (list[Path]): if set, supersedes previous options; all files must be contained
                in the same folder.
            img_suffix (str): suffix of line image (default: '.png').
            gt_suffix (str): suffix of gt transcription file (default: '.gt.txt').
            to_tsv_file (str): serialize the dataset list as a TSV in the work folder.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground
                truth at loading time.
            expansion_masks (bool): if True (default), add transcription expansion offsets
                to the sample if it is present in the XML source line annotations.
            channel_func (Callable): function that takes image and binary polygon mask as inputs,
                and generates an additional channel in the sample. Default: None.
            channel_suffix (str): when loading items from a work folder, which suffix
                to read for the channel file. Default: '' (=ignore channel file).
            padding_style (str): When extracting line bounding boxes, padding to be 
                used around the polygon: 'median'=median value of the polygon; 'noise'=random;
                'zero'=0s. The polygon boolean mask is automatically saved on/retrieved from the disk;
                Default is None.
        """

        data = []
        from_line_files = [ Path(f) for f in from_line_files ] 

        self.img_suffix = img_suffix
        self.gt_suffix = gt_suffix
        self.channel_suffix = channel_suffix

        if from_tsv_file:
            tsv_path = Path( from_tsv_file )
            if tsv_path.exists():
                self.work_folder_path = tsv_path.parent
                # paths are assumed to be absolute
                self._data = self.load_from_tsv( tsv_path, expansion_masks )
                logger.debug("data={}".format( data[:6]))
                #logger.debug("height: {} type={}".format( data[0]['height'], type(data[0]['height'])))
            else:
                raise FileNotFoundError(f'File {tsv_path} does not exist!')
        else:
            if from_line_files:
                dataset_dirs = set( [ img.parent for img in from_line_files ] )
                if len(dataset_dirs) > 1:
                    raise Exception(f'Source files should belong to the same directory (found {len(dataset_dirs)} parent folders: {[str(f) for f in dataset_dirs]}.')
                self.work_folder_path = list(dataset_dirs)[0]
                self._data = self.load_line_items_from_files( from_line_files )
            elif from_work_folder:
                self.work_folder_path = Path(from_work_folder)
                logger.info("Building samples from existing images and transcription files in {}".format(self.work_folder_path))
                self._data = self.load_line_items_from_dir( self.work_folder_path )
            if to_tsv_file:
                self.dump_data_to_tsv(self._data, Path(self.work_folder_path.joinpath(to_tsv_file)) )

        if not self._data:
            raise DataException("No data found. from_tsv_file={}, from_work_folder={}, from_line_files={}".format(from_tsv_file, from_work_folder, from_line_files))

        trf = v2.Compose( [ v2.ToImage(), v2.ToDtype(torch.float32, scale=True) ])  
        if transform is not None:
            trf = v2.Compose( [ trf, transform ] ) 
        super().__init__(root=self.work_folder_path, transform=trf, target_transform=target_transform ) # if target_transform else self.filter_transcription)

        if padding_style and padding_style not in ['noise', 'zero', 'median', 'none']:
            raise ValueError(f"Incorrect padding style: '{line_padding_style}'. Valid styles: 'noise', 'zero', or 'median'.")

        # bbox or polygons and/or masks
        self.config = {
                'from_tsv_file': from_tsv_file,
                'from_work_folder': from_work_folder,
                'channel_func': channel_func,
                'channel_suffix': channel_suffix,
                'padding_style': padding_style,
                'expansion_masks': expansion_masks,
        }


    def load_line_items_from_dir(self, work_folder_path: Union[Path,str] ) -> list[dict]:
        """Construct a list of samples from a directory that has been populated with
        line images and line transcriptions

        Args:
            work_folder_path (Union[Path,str]): a folder containing images (`*.png`), transcription 
                files (`*.gt.txt`) and optional extra channel.

        Returns:
            list[dict]: a list of samples.
        """
        if type(work_folder_path) is str:
            work_folder_path = Path( work_folder_path )
        file_paths = list( work_folder_path.glob('*{}'.format( self.img_suffix )))
        return self.load_line_items_from_files( file_paths )

                
    def load_line_items_from_files(self, file_paths:list[Path] ) -> list[dict]:
        """Construct a list of samples from a list of line images.

        Args:
            file_paths (list[Path]): images paths (typically `*.png`)

        Returns:
            list[dict]: a list of samples.
        """
        samples = []
        for img_file_path in file_paths:
            sample=dict()
            logger.debug(img_file_path)            
            gt_file_path = Path( re.sub(r'{}$'.format( self.img_suffix ), self.gt_suffix, str(img_file_path)))
            sample['img']=img_file_path
            with Image.open( img_file_path, 'r') as img:
                sample['width'], sample['height'] = img.size
            
            with open(gt_file_path, 'r') as gt_if:
                transcription=gt_if.read().rstrip()
                expansion_masks_match = re.search(r'^(.+)<([^>]+)>$', transcription)
                if expansion_masks_match is not None:
                    sample['transcription']=expansion_masks_match.group(1)
                    sample['expansion_masks']=eval(expansion_masks_match.group(2))
                else:
                    sample['transcription']=transcription
            # binary mask
            binary_mask_path = Path(  re.sub(r'{}$'.format( self.img_suffix ), '.bool.npy.gz', str(img_file_path)))
            assert binary_mask_path.exists()
            sample['binary_mask']=binary_mask_path

            # optional mask
            channel_file_path = Path( re.sub(r'{}$'.format( self.img_suffix ), self.channel_suffix, str(img_file_path)))
            if channel_file_path.exists():
                sample['img_channel']=channel_file_path

            samples.append( sample )

        logger.debug("Loaded {} samples from {} image files.".format( len(samples), len(file_paths)))
        return samples
                

    @staticmethod
    def load_from_tsv(file_path: Path, expansion_masks=False) -> list[dict]:
        """Load samples (as dictionaries) from an existing TSV file. Each input line is a tuple::

           <img file path> <transcription text> <height> <width> [<polygon points>]

        Each line image is assumed to have a binary mask counterpart `*.bool.npy.gz` (computing
        from an optional field <polygon points> is not implemented).

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
        sample_df = pandas.read_csv( file_path, sep='\t', header=0)
        samples = []
        for row in range( sample_df.shape[0] ):
            img_file, gt_field, height, width = sample_df.loc[ row ][:4]
            channel_file = sample_df.loc[ row ][4] if len(sample_df.columns)>4 else None
            binary_mask_file = work_folder_path.joinpath( img_file ).with_suffix('.bool.npy.gz')

            expansion_masks_match = re.search(r'^(.+)<([^>]+)>$', gt_field)
            if expansion_masks_match is not None:
                gt_field = expansion_masks_match.group(1)

            spl = { 'img': work_folder_path.joinpath( img_file ), 'transcription': gt_field,
                    'height': int(height), 'width': int(width) }
            if channel_file is not None:
                spl['img_channel']=work_folder_path.joinpath( channel_file )
            if expansion_masks and expansion_masks_match is not None:
                spl['expansion_masks']=eval( expansion_masks_match.group(2))
            if binary_mask_file.exists():
                spl['binary_mask']=binary_mask_file

            samples.append( spl )
                               
        return samples


    @staticmethod
    def dump_data_to_tsv(samples: list[dict], file_path: str='', all_path_style=False) -> None:
        """Create a TSV file with all tuples (`<line image absolute path>`, `<transcription>`, `<height>`, `<width>` `[<polygon points]`).
        Height and widths are the original heights and widths.

        Args:
            samples (list[dict]): dataset samples.
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
            # header: ImgPath  GT  Height  Width [Channel]
            of.write('ImgPath\tGT\tHeight\tWidth{}\n'.format( '\tChannel' if 'img_channel' in samples[0] else ''))
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
                                            


    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        """Callback function for the iterator. Assumption: the raw sample always contains
        the bounding box image + binary polygon mask. Any combined image (ex. noise-background)
        is constructed from those, _before_ any transform that is passed to the DS constructor.

        Args:
            index (int): item index.

        Returns:
            dict[str,Union[Tensor,int,str]]: a sample dictionary
        """
        img_path = self._data[index]['img']
        
        assert isinstance(img_path, Path) or isinstance(img_path, str)

        sample = self._data[index].copy()
        sample['transcription']=self.target_transform( sample['transcription'] )

        with Image.open( img_path ) as img:
            
            img_array_hwc = np.array( img ) # img path --> img ndarray

            if self.config['padding_style'] is not None:
                assert 'binary_mask' in sample and sample['binary_mask'].exists()
                with gzip.GzipFile(sample['binary_mask'], 'r') as mask_in:
                    binary_mask_hw = np.load( mask_in )
                    padding_func = lambda x, m, channel_dim=2: x
                    if self.config['padding_style']=='noise':
                        padding_func = tsf.bbox_noise_pad
                    elif self.config['padding_style']=='zero':
                        padding_func = tsf.bbox_zero_pad
                    elif self.config['padding_style']=='median':
                        padding_func = tsf.bbox_median_pad
                    img_array_hwc = padding_func( img_array_hwc, binary_mask_hw, channel_dim=2 )
                    if len(img_array_hwc.shape) == 2: # for ToImage() transf. to work in older torchvision
                        img_array_hwc=img_array_hwc[:,:,None]
            del sample['binary_mask']

            # img ndarray --> tensor
            sample['img']=img_array_hwc 
            logger.debug("Before transform: sample['img'].dtype={}".format( sample['img'].dtype))
            sample = self.transform( sample )

            if 'img_channel' in self._data[index]:
                channel_t = None
                if self._data[index]['img_channel'].suffix == '.gz':
                    with gzip.GzipFile(self._data[index]['img_channel'], 'r') as channel_in:
                        channel_t = torch.from_numpy( np.load( channel_in ) )/255
                else:
                    channel_t = np.load(self._data[index]['img_channel'])/255
                sample['img']=torch.cat( [sample['img'], channel_t[None,:,:]] )

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
        return len( self._data )


    def __repr__(self) -> str:

        summary = '\n'.join([
                    f"Work folder:\t{self.work_folder_path}",
                    f"Data points:\t{len(self._data)}",
                    "Stats:",
                    f"{self.dataset_stats(self._data)}" if self._data else 'No data',])
        if self.config['from_tsv_file']:
             summary += "\nBuilt from TSV input:\t{}".format( self.config['from_tsv_file'] )
        
        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")



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



def dummy():
    """"""
    return True

