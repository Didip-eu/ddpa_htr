import numpy as np
import torch
from torchvision.transforms import v2


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

def bbox_median_pad_old(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
    """ Pad a polygon BBox with the median value of the polygon. Used by
    the line extraction method.

    :param img_chw: an array (C,H,W). Optionally: (H,W,C)
    :type img_chw: np.ndarray.

    :param mask_hw: a 2D Boolean mask (H,W).
    :type mask_hw: np.ndarray

    :param channel_dim: the channel dimension: 2 for (H,W,C) images. Default is 0.
    :type channel_dim: int

    :returns: the padded image, with same shape as input.
    :rtype: np.ndarray.
    """
    img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
    padding_bg = np.zeros( img.shape, dtype=img.dtype)

    for ch in range( img.shape[0] ):
        med = np.median( img[ch][mask_hw] ).astype( img.dtype )
        padding_bg[ch] += np.logical_not(mask_hw) * med
        padding_bg[ch] += img[ch] * mask_hw
    return padding_bg.transpose(1,2,0) if channel_dim==2 else padding_bg



def bbox_noise_pad_old(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
    """ Pad a polygon BBox with noise. Used by the line extraction method.

    :param img_chw: an array (C,H,W). Optionally: (H,W,C)
    :type img_chw: np.ndarray

    :param mask_hw: a 2D Boolean mask (H,W).
    :type mask_hw: np.ndarray

    :param channel_dim: the channel dimension: 2 for (H,W,C) images. Default is 0.
    :type channel_dim: int

    :returns: the padded image, with same shape as input.
    :rtype: np.ndarray.
    """
    img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
    padding_bg = np.random.randint(0, 255, img.shape, dtype=img_chw.dtype)
    
    padding_bg *= np.logical_not(mask_hw) 
    mask_chw = np.stack( [ mask_hw, mask_hw, mask_hw ] )
    padding_bg += img * mask_chw
    return padding_bg.transpose(1,2,0) if channel_dim==2 else padding_bg

def bbox_zero_pad_old(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0 ) -> np.ndarray:
    """ Pad a polygon BBox with zeros. Used by the line extraction method.

    :param img_chw: an array (C,H,W). Optionally: (H,W,C)
    :type img_chw: np.ndarray

    :param mask_hw: a 2D Boolean mask (H,W).
    :type mask_hw: np.ndarray

    :param channel_dim: the channel dimension: 2 for (H,W,C) images. Default is 0.
    :type channel_dim: int

    :returns: the padded image, with same shape as input.
    :rtype: np.ndarray.
    """
    img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
    mask_chw = np.stack( [ mask_hw, mask_hw, mask_hw ] )
    img_out = img * mask_chw
    return img_out.transpose(1,2,0) if channel_dim == 2 else img_out



class PadToWidth():
    """ Pad an image to desired length."""

    def __init__( self, max_w ):
        self.max_w = max_w

    def __call__(self, sample: dict) -> dict:
        """Transform a sample: only the image is modified, not the nominal height and width.
        A mask covers the unpadded part of the image.
        
            
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
        
