import skimage as ski
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gzip


# 2D-channel functions, to be used in combination with either the BB image or the polygon/background image,
# as a possible value for the 'channel_func' constructor's paramete.
#
# Channels are 
# + created and stored on-disk as numpy arrays with image types (int, 0-255)
# + loaded as numpy arrays (0-255)
# Conversion to tensor (0-1) is done by the transform for both.

def bbox_alpha_mask(img_hwc: np.ndarray, mask_hw: np.ndarray, transparency=.4) -> np.ndarray:
    """ Create a transparency mask that keeps polygon clear, but background faded.
    Meant to be applied to a bounding box image.

    Args:
        img_hwc (np.ndarray): Ignored here, but required by the interface.
        mask_hw (np.ndarray): boolean array.
    Output:
        np.ndarray: (H,W) channel, with int values in the range 0-255 out of the polygons and 255 in
            the polygon.
    """
    mask = (np.full( mask_hw.shape, transparency ) * np.logical_not( mask_hw) + mask_hw)
    return (mask*255).astype('uint8')

def bbox_blurry_channel(img_hwc: np.ndarray, ignored=None, kernel_size=10) -> np.ndarray:
    """ Create a mean-filtered version of the img.

    Args:
        img_hwc (np.ndarray): array (H,W,C)-uint8, as created from a PIL image.
        ignored (np.ndarray): ignored here, but required by the interface.
    Output:
        np.ndarray: (H,W) channel, with int values in the range 0-255.
    """
    # note: skimage functions typically return 0-1 floats, no matter the input
    return ski.filters.rank.mean( ski.util.img_as_ubyte(ski.color.rgb2gray(img_hwc)), np.full((kernel_size,kernel_size),1))

def bbox_blurry_channel_file_from_img(img_file_path:str, kernel_size=10, suffix=".mask.npy", compress=True) -> None:
    img_file_path = Path( img_file_path)
    img_hwc = ski.io.imread( img_file_path )
    channel_hw = bbox_blurry_channel( img_hwc, kernel_size=kernel_size)
    if compress:
        suffix += '.gz'
        with gzip.GzipFile( img_file_path.with_suffix('').with_suffix(suffix), 'w') as zf:
            np.save( zf, channel_hw)
    else:
        np.save( img_file_path.with_suffix('').with_suffix(suffix), channel_hw)


def bbox_gray_channel(img_hwc: np.ndarray, ignored=None) -> np.ndarray:
    """ Create a gray version of the img.
    
    Args:
        img_hwc (np.ndarray): array (H,W,C)-uint8, as created from a PIL image.
        ignored (np.ndarray): unused, but required by the interface.
    Output:
        np.ndarray: (H,W) channel, with int values in the range 0-255.
    """
    return ski.util.img_as_ubyte( ski.color.rgb2gray(img_hwc))


def bbox_gray_channel_file_from_img(img_file_path:str, kernel_size=10, suffix=".mask.npy", compress=True) -> None:
    img_file_path = Path( img_file_path)
    img_hwc = ski.io.imread( img_file_path )
    channel_hw = bbox_gray_channel( img_hwc )
    if compress:
        suffix += '.gz'
        with gzip.GzipFile( img_file_path.with_suffix('').with_suffix(suffix), 'w') as zf:
            np.save( zf, channel_hw)
    else:
        np.save( img_file_path.with_suffix('').with_suffix(suffix), channel_hw)



def show_img_three_plus_one(img_chw):
    """ Combine first 3 channels with the 4th one 
    
    Args:
        img_chw (np.ndarray): (C,H,W) array, typically obtained from a tensor. 
    """
    img_array = (img_chw[:3].numpy()).transpose(1,2,0)
    mask_array = img_chw[3].numpy()
    mask_array = np.stack( [ mask_array, mask_array, mask_array ]).transpose(1,2,0)
    plt.imshow( img_array * mask_array )
    plt.show()

