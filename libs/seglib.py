
#stdlib
from pathlib import Path
import json
from typing import Callable, Optional, Union, Mapping, Any
import itertools
import re
import copy
import sys
import math
from datetime import datetime

# 3rd-party
from PIL import Image, ImageDraw
import skimage as ski
import xml.etree.ElementTree as ET
import torch
from torch import Tensor
from torchvision.tv_tensors import Mask
import numpy as np
import numpy.ma as ma


__LABEL_SIZE__=8

"""Functions for segmentation output management: a subset for HTR purpose.
"""


def line_binary_mask_from_json_file( segmentation_json: str, polygon_key='coords' ) -> Tensor:
    """From a JSON segmentation file,  return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_json (str): a JSON file describing the lines.
        polygon_key (str): polygon dictionary entry.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    with open( segmentation_json, 'r' ) as json_file:
        return line_binary_mask_from_segmentation_dict( json.load( json_file ), polygon_key=polygon_key)

def line_binary_mask_from_xml_file( page_xml: str ) -> Tensor:
    """From a PageXML file describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        page_xml (str): a Page XML file describing the lines.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    segmentation_dict = segmentation_dict_from_xml( page_xml )
    return line_binary_mask_from_segmentation_dict( segmentation_dict )


def line_binary_mask_from_segmentation_dict( segmentation_dict: dict, polygon_key='coords' ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.
        polygon_key (str): polygon dictionary entry.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict, polygon_key=polygon_key)
    # create 2D boolean matrix
    mask_size = (segmentation_dict['image_width'], segmentation_dict['image_height'])
    return torch.tensor( np.sum( [ ski.draw.polygon2mask( mask_size, polyg ).transpose(1,0) for polyg in polygon_boundaries ], axis=0))

def line_binary_mask_stack_from_json_file( segmentation_json: str, polygon_key='coords' ) -> Tensor:
    """From a JSON file describing polygons, return a stack of boolean masks where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_json (str): a JSON file describing the lines.
        polygon_key (str): polygon dictionary entry.

    Returns:
        Tensor: a boolean tensor with size (N,H,W)
    """
    with open( segmentation_json, 'r' ) as json_file:
        return line_binary_mask_stack_from_segmentation_dict( json.load( json_file ), polygon_key=polygon_key)

def line_binary_mask_stack_from_segmentation_dict( segmentation_dict: dict, polygon_key='coords' ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a stack of boolean masks where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.
        polygon_key (str): polygon dictionary entry.

    Returns:
        Tensor: a boolean tensor with size (N,H,W)
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict, polygon_key=polygon_key)
    # create 2D boolean matrix
    mask_size = (segmentation_dict['image_width'], segmentation_dict['image_height'])
    return torch.tensor( np.stack( [ ski.draw.polygon2mask( mask_size, polyg ).transpose(1,0) for polyg in polygon_boundaries ]))

def line_polygons_from_segmentation_dict( segmentation_dict: dict, polygon_key='coords' ) -> list[list[int]]:
    """From a segmentation dictionary describing polygons, return a list of polygon boundaries, i.e. lists of points.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file. The 'lines' entry is either
        top-level key, or nested as in 'regions > region > lists'.
    Returns:
        list[list[int]]: a list of lists of coordinates.
    """
    if 'lines' in segmentation_dict:
        return [ line[polygon_key] for line in segmentation_dict['lines'] ]
    elif 'regions' in segmentation_dict:
        return [ line[polygon_key] for reg in segmentation_dict['regions'] for line in reg['lines']] 
    return []

def line_dicts_from_segmentation_dict( segmentation_dict: dict) -> list[dict]:
    """From a segmentation dictionary, return a list of all line dictionaries.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file. The 'lines' entry is either
        top-level key, or nested as in 'regions > region > lists'.
    Returns:
        list[dict]: a list of dictionaries.
    """
    if 'lines' in segmentation_dict:
        return segmentation_dict['lines']
    elif 'regions' in segmentation_dict:
        return [ line for reg in segmentation_dict['regions'] for line in reg['lines']]
    return []


def line_images_from_img_xml_files(img: str, page_xml: str, as_dictionary=False ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img (str): the input image's file path
        page_xml: :type page_xml: str a Page XML file describing the
            lines.
        as_dictionary (bool): return segmentation dict where each line is a tuple (<img>,<msk>,<line_dict>); useful
            for keeping track of line ids when running inference.

    Returns:
        list: a list of pairs (<line image BB>: np.ndarray (HWC), mask:
        np.ndarray (HW))
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = segmentation_dict_from_xml( page_xml )
        line_pairs = line_images_from_img_segmentation_dict( img_wh, segmentation_dict )
        line_triplets = [ (*line_pair, line_dict) for line_pair, line_dict in zip( line_pairs, line_dicts_from_segmentation_dict(segmentation_dict)) ]
        if as_dictionary:
            segmentation_dict['lines'] = line_triplets
            return segmentation_dict
        return line_pairs


def line_images_from_img_json_files( img: str, segmentation_json: str, as_dictionary=False ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file
        as_dictionary (bool): return segmentation dict where each line is a tuple (<img>,<msk>,<line_dict>); useful
            for keeping track of line ids when running inference.

    Returns:
        Union[list,dict]: a segmentation dictionary or a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW))
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        segmentation_dict = json.load( json_file )
        line_pairs = line_images_from_img_segmentation_dict( img_wh, segmentation_dict )
        line_triplets = [ (*line_pair, line_dict) for line_pair, line_dict in zip( line_pairs, line_dicts_from_segmentation_dict(segmentation_dict)) ]
        if as_dictionary:
            segmentation_dict['lines'] = line_triplets
            return segmentation_dict
        return line_pairs

def line_images_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict, polygon_key='coords' ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict (dict) a dictionary, typically constructed from a JSON file.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: a list of pairs (<line
        image BB>: np.ndarray (HWC), mask: np.ndarray (HWC))
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict, polygon_key=polygon_key)
    img_hwc = np.asarray( img_whc )

    pairs_line_bb_and_mask = []# [None] * len(polygon_boundaries)

    for lbl, polyg in enumerate( polygon_boundaries ):

        points = np.array( polyg )[:,::-1] # polygon's points ( x <-> y )
        page_polyg_mask = ski.draw.polygon2mask( img_hwc.shape, points ) # np.ndarray (H,W,C)
        y_min, x_min, y_max, x_max = np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )
        line_bbox = img_hwc[y_min:y_max+1, x_min:x_max+1] # crop both img and mask
        # note: mask has as many channels as the original image
        bb_label_mask_hwc = page_polyg_mask[y_min:y_max+1, x_min:x_max+1]

        #pairs_line_bb_and_mask[lbl]=( line_bbox, bb_label_mask )
        pairs_line_bb_and_mask.append( (line_bbox, bb_label_mask_hwc) )

    return pairs_line_bb_and_mask


def line_masks_from_img_xml_files(img: str, page_xml: str ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        page_xml (page_xml): str a Page XML file describing the lines.

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = segmentation_dict_from_xml( page_xml )
        return line_masks_from_img_segmentation_dict( img_wh, segmentation_dict )


def line_masks_from_img_json_files( img: str, segmentation_json: str, key='coords' ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        return line_masks_from_img_segmentation_dict( img_wh, json.load( json_file ), key=key)

def line_masks_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict, polygon_key='coords' ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    the bounding box coordinates and the boolean masks.

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict: :type segmentation_dict: dict a dictionary, typically constructed from a JSON file.

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict, polygon_key=polygon_key)

    img_hwc = np.asarray( img_whc )

    bbs = []
    masks = []

    for polyg in polygon_boundaries:
        points = np.array( polyg )[:,::-1]  # polygon's points ( x <-> y )
        page_polyg_mask = ski.util.img_as_ubyte(ski.draw.polygon2mask( img_hwc.shape[:2], points )) # np.ndarray (H,W)
        y_min, x_min, y_max, x_max = [ float(p) for p in (np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )) ]
        bbs.append( (x_min,y_min,x_max,y_max) )
        masks.append( page_polyg_mask )

    return (np.stack( bbs ), np.stack( masks ))



def xml_from_segmentation_dict(seg_dict: str, pagexml_filename: str='', polygon_key='coords', with_text=False):
    """Serialize a JSON dictionary describing the lines into a PageXML file.
    Caution: this is a crude function, with no regard for validation.

    Args:
         seg_dict (dict[str,Union[str,list[Any]]]): segmentation dictionary of the form

            {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}
            or
            {"text_direction": ..., "type": "baselines", "regions": [ {"id": "r0", "lines": [{"tags": ..., "baseline": [ ... ]}]}, ... ]}
        pagexml_filename (str): if provided, output is saved in a PageXML file (standard output is the default).
        polygon_key (str): if the segmentation dictionary contain alternative polygons (f.i. 'extBoundary'),
            use them, instead of the usual line 'coords'.
        with_text (bool): encode line transcription, if it exists. Default is False.
    """
    def boundary_to_point_string( list_of_pts ):
        return ' '.join([ f"{pair[0]:.0f},{pair[1]:.0f}" for pair in list_of_pts ] )

    rootElt = ET.Element('PcGts', attrib={
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15", 
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", 
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"})
    metadataElt = ET.SubElement(rootElt, 'MetaData')
    creatorElt = ET.SubElement( metadataElt, 'Creator')
    creatorElt.text=seg_dict['metadata']['creator'] if ('metadata' in seg_dict and 'creator' in seg_dict['metadata']) else 'Universität Graz/DH/nprenet@uni-graz.at'
    createdElt = ET.SubElement( metadataElt, 'Created')
    createdElt.text=datetime.now().isoformat()
    lastChangeElt = ET.SubElement( metadataElt, 'LastChange')
    lastChangeElt.text=createdElt.text
    commentElt = ET.SubElement( metadataElt, 'Comments')
    if 'comments' in seg_dict['metadata']:
        commentElt.text = seg_dict['metadata']['comments']
    # for back-compatibility
    elif 'comment' in seg_dict:
        commentElt.text = seg_dict['comment']

    img_name = Path(seg_dict['image_filename']).name
    img_width, img_height = seg_dict['image_width'], seg_dict['image_height']    
    pageElt = ET.SubElement(rootElt, 'Page', attrib={'imageFilename': img_name, 'imageWidth': f"{img_width}", 'imageHeight': f"{img_height}"})
    # if no region in segmentation dict, create one (image-wide)
    if 'regions' not in seg_dict:
        seg_dict['regions']=[{'id': 'r0', 'coords': [[0,0],[img_width-1,0],[img_width-1,img_height-1],[0,img_height-1]]}, ]
    for reg in seg_dict['regions']:
        reg_xml_id = f"r{reg['id']}" if (type(reg['id']) is int or reg['id'][0]!='r') else reg['id']
        regElt = ET.SubElement( pageElt, 'TextRegion', attrib={'id': reg_xml_id})
        ET.SubElement(regElt, 'Coords', attrib={'points': boundary_to_point_string(reg['coords'])})
        # 3 cases: 
        # - top-level list of lines with region ref
        # - top-level list of lines with no regions
        # - top-level regions with a list of lines in each
        lines = [ l for l in seg_dict['lines'] if (('region' in l and l['region']==reg['id']) or 'region' not in l) ] if 'lines' in seg_dict else reg['lines']
        for line in lines:
            line_xml_id = f"l{line['id']}" if type(line['id']) is int else line['id']
            textLineElt = ET.SubElement( regElt, 'TextLine', attrib={'id': line_xml_id} )
            ET.SubElement( textLineElt, 'Coords', attrib={'points': boundary_to_point_string(line[polygon_key])} )
            if 'baseline' in line:
                ET.SubElement( textLineElt, 'Baseline', attrib={'points': boundary_to_point_string(line['baseline'])})
            if with_text and 'text' in line:
                ET.SubElement( ET.SubElement( textLineElt, 'TextEquiv'), 'Unicode').text = line['text']

    tree = ET.ElementTree( rootElt )
    ET.indent(tree, space='\t', level=0)
    if pagexml_filename:
        tree.write( pagexml_filename, encoding='utf-8' )
    else:
        tree.write( sys.stdout, encoding='unicode' )


def segmentation_dict_from_xml(page: str, get_text=False, regions_as_boxes=True, strict=False) -> dict[str,Union[str,list[Any]]]:
    """Given a pageXML file name, return a JSON dictionary describing the lines.

    Args:
        page (str): path of a PageXML file.
        get_text (bool): extract line text content, if present (default: False); this
            option causes line with no text to be yanked from the dictionary.
        regions_as_boxes (bool): when regions have more than 4 points or are not rectangular,
            store their bounding boxes instead; the boxe's boundary is determined
            by its pertaining lines, not by its nominal coordinates(default: True).
        strict (bool): if True, raise an exception if line coordinates are not comprised within
            their region's boundaries; otherwise (default), the region value is automatically
            extended to encompass the line coordinates.

    Returns:
        dict[str,Union[str,list[Any]]]: a dictionary of the form::

            {"metadata": { ... },
             "text_direction": ..., "type": "baselines", 
             "lines": [{"id": ..., "coords": [ ... ], "baseline": [ ... ]}, ... ],
             "regions": [{"id": ..., "coords": [ ... ]}, ... ] }

           Regions are stored as a top-element.

    """
    def parse_coordinates( pts ):
        return [ [ int(p) for p in pt.split(',') ] for pt in pts.split(' ') ]

    def construct_line_entry(line: ET.Element, region_ids: list = [] ) -> dict:
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                return None
            bl_points = baseline_elt.get('points')
            if bl_points is None:
                return None
            baseline_points = parse_coordinates( bl_points )
            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                return None
            c_points = coord_elt.get('points')
            if c_points is None:
                return None
            polygon_points = parse_coordinates( c_points )

            line_text, line_custom_attribute = '', ''
            if get_text:
                text_elt = line.find('./pc:TextEquiv', ns) 
                if text_elt is not None:
                    line_custom_attribute = text_elt.get('custom') if 'custom' in text_elt.keys() else ''
                unicode_elt = text_elt.find('./pc:Unicode', ns)
                if unicode_elt is not None:
                    line_text = unicode_elt.text 
            line_dict = {'id': line_id, 'baseline': baseline_points, 
                        'coords': polygon_points, 'regions': region_ids}
            if line_text and not re.match(r'\s*$', line_text):
                line_dict['text'] = line_text 
                if line_custom_attribute:
                    line_dict['custom']=line_custom_attribute
            elif get_text:
                return None
            return line_dict

    def check_line_entry(line_dict: dict, region_dict: dict):
        """ Check whether line coords are within region boundaries."""
        reg_l, reg_t, reg_r, reg_b = region_dict['coords']
        return all([ (pt[0] >= reg_l[0] and pt[0] <= reg_r[0] and pt[1] >= reg_t[1] and pt[1] <= reg_b[1]) for pt in line_dict['coords']])

    def extend_box( box_coords, inner_coords ):
        """Extend box coordinates to encompass inner boundaries """
        inner_xs, inner_ys = [ pt[0] for pt in inner_coords ], [ pt[1] for pt in inner_coords ]
        inner_left, inner_right, inner_top, inner_bottom = min(inner_xs), max(inner_xs), min(inner_ys), max(inner_ys)
        return [ [ inner_left if inner_left < box_coords[0][0] else box_coords[0][0],
                 inner_top if inner_top < box_coords[0][1] else box_coords[0][1]],
                [ inner_right if inner_right > box_coords[1][0] else box_coords[1][0],
                 inner_top if inner_top < box_coords[1][1] else box_coords[1][1]],
                [ inner_right if inner_right > box_coords[2][0] else box_coords[2][0],
                 inner_bottom if inner_bottom > box_coords[2][1] else box_coords[2][1]],
                [ inner_left if inner_left < box_coords[3][0] else box_coords[3][0],
                 inner_bottom if inner_bottom > box_coords[3][1] else box_coords[3][1]],]

    def process_region( region: ET.Element, region_accum: list, line_accum: list, region_ids:list ):
        # order of regions: outer -> inner
        region_ids = region_ids + [ region.get('id') ]
        
        region_coord_elt = region.find('./pc:Coords', ns)
        if region_coord_elt is not None:
            rg_points = region_coord_elt.get('points')
            if rg_points is None:
                raise ValueError("Region has no coordinates. Aborting.")
            rg_points = parse_coordinates( rg_points )
            if regions_as_boxes:
                xs, ys = [ pt[0] for pt in rg_points ], [ pt[1] for pt in rg_points ] 
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                rg_points = [[left,top], [right,top], [right,bottom], [left, bottom]]

        region_accum.append( {'id': region.get('id'), 'coords': rg_points } )

        for line_idx, elt in enumerate( list(region.iter())[1:] ):
            if elt.tag == "{{{}}}TextLine".format(ns['pc']):
                line_entry = construct_line_entry( elt, region_ids )
                if line_entry is None:
                    continue
                if not check_line_entry(line_entry, region_accum[-1] ):
                    if strict:
                        raise ValueError("Page {}, region {}, l. {}: boundaries are not contained within its region.".format(page, region_ids[-1], line_idx))
                    else: # extend region's bounding box boundary
                        region_accum[-1]['coords'] = extend_box( region_accum[-1]['coords'], line_entry['coords']+line_entry['baseline'] )
                line_accum.append( line_entry )
            elif elt.tag == "{{{}}}TextRegion".format(ns['pc']):
                process_region(elt, region_accum, line_accum, region_ids)

    with open( page, 'r' ) as page_file:

        # extract namespace
        ns = {}
        for line in page_file:
            m = re.match(r'\s*<([^:]+:)?PcGts\s+xmlns(:[^=]+)?=[\'"]([^"]+)["\']', line)
            if m:
                ns['pc'] = m.group(3)
                page_file.seek(0)
                break

        if 'pc' not in ns:
            raise ValueError(f"Could not find a name space in file {page}. Parsing aborted.")

        lines = []
        regions = []
        page_dict = {}

        page_tree = ET.parse( page_file )
        page_root = page_tree.getroot()

        metadata_elt = page_root.find('./pc:Metadata', ns)
        if metadata_elt is None:
            page_dict = { 'metadata': { 'created': str(datetime.now()), 'creator': __file__, } }
        else:
            created_elt = metadata_elt.find('./pc:Created', ns)
            creator_elt = metadata_elt.find('./pc:Creator', ns)
            comments_elt = metadata_elt.find('./pc:Comments', ns)
            page_dict: {
                    'metadata': {
                        'created': created_elt.text if created_elt else str(datetime.datetime.now()),
                        'creator': creator_elt.text if creator_elt else __filename__,
                        'comments': comments_elt.text if comments_elt else "",
                    }
            }

        page_dict['type']='baselines'
        page_dict['text_direction']='horizontal-lr'

        pageElement = page_root.find('./pc:Page', ns)
        
        page_dict['image_filename']=pageElement.get('imageFilename')
        page_dict['image_width'], page_dict['image_height']=[ int(pageElement.get('imageWidth')), int(pageElement.get('imageHeight'))]

        
        for textRegionElement in pageElement.findall('./pc:TextRegion', ns):
            process_region( textRegionElement, regions, lines, [] )

        page_dict['lines'] = lines
        page_dict['regions'] = regions

    return page_dict 


def segdict_sink_lines(segdict: dict):
    """Convert a segmentation dictionary with top-level line array ('lines') 
    to a nested dictionary where each region in the 'regions' array contains its 
    corresponding 'lines' array. No change applied if lines are already wrapped
    into the regions.

    Args:
        segdict (dict): segmentation dictionary of the form::

                {..., "lines": [ {"id":..., "regions": [...]}, ... ], "regions": [ ... ] }

            OR

                {..., "lines": [ {"id":..., "region": "r0"}, ... ], "regions": [ ... ] }

    Returns:
        dict: a modified copy of the original dictionary::

            {..., "regions": [ {"id":..., lines=[{"id": ... }, ... ]}, ... ] }
    """
    segdict = segdict.copy()
    if 'lines' not in segdict or not segdict['lines']:
        return segdict
    # if no 'regions' entry for lines, assign to each line its proper region
    if 'regions' not in segdict['lines'][0]:
        for line in segdict['lines']:
            if 'region' in line:
                line['regions']=[ line['region'] ]
                del line['region']
            else:
                for reg in segdict['regions']:
                    if (line['coords'] >= np.min( reg['coords'], axis=0 )).all() and (line['coords'] <= np.max( reg['coords'], axis=0 )).all():
                        print("Check coordinates")
                        if 'regions' not in line:
                            line['regions']=[]
                    line['regions'].append( reg['id'] )
        
    for line in segdict['lines']:
        this_reg=[ reg for reg in segdict['regions'] if reg['id']==line['regions'][0] ][0] if ('regions' in line and line['regions']) else line['region']
        if 'lines' not in this_reg:
            this_reg['lines']=[]
        this_reg['lines'].append(line)
        del line['regions']
    del segdict['lines']
    return segdict


def layout_regseg_to_crops( img: Image.Image, regseg: dict, region_labels: list[str], force_rgb=False ) -> tuple[list[Image.Image], list[str]]:
    """From a layout-app segmentation dictionary, return the regions with matching
    labels as a list of images.

    Args:
        img (Image.Image): Image to crop.
        regseg (dict): the regional segmentation json, as given by the 'layout' app
        region_labels (list[str]): Labels to be extracted.

    Returns:
        tuple[list[Image.Image], list[str]]: a tuple with 
            - a list of images (HWC)
            - a list of box coordinates (LTRB)
            - a list of class names
    """
    clsid_2_clsname = { i:n for (i,n) in enumerate( regseg['class_names'] )}
    to_keep = [ i for (i,v) in enumerate( regseg['rect_classes'] ) if clsid_2_clsname[v] in region_labels ]

    if force_rgb and img.mode != 'RGB':
        img = img.convert('RGB')

    return tuple( zip(*[ ( img.crop( regseg['rect_LTRB'][i] ),
                  regseg['rect_LTRB'][i],
                  clsid_2_clsname[ regseg['rect_classes'][i]]) for i in to_keep ]))


def layout_regseg_check_class(regseg: dict, region_labels: list[str] ) -> list[bool]:
    """From a layout-app segmentation dictionary, check if rectangle with given labels
    have been detected.

    Args:
        regseg (dict): the regional segmentation json, as given by the 'layout' app
        region_labels (list[str]): Labels to check.

    Returns:
        list[bool]: a list of boolean values.
    """

    clsname_2_clsid = { n:i for (i,n) in enumerate( regseg['class_names'] )}
    
    output = None
    try:
        output = [ clsname_2_clsid[l] in regseg['rect_classes'] for l in region_labels ]
    except KeyError as e:
        print(f"Class label {e} does not exist in the segmentation file.")
    return output




def dummy():
    """Just to check that the module is testable."""
    return True
