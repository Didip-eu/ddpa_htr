from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from kraken.vgsl import TorchVGSLModel
import numpy as np
import re
import Levenshtein
from didip_handwriting_datasets.alphabet import Alphabet


from typing import Union,Tuple,List
import warnings


import logging
logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)



class HTR_Model():
    """
    Note: by convention, VGSL specifies the dimensions in NHWC order, while Torch uses NCHW. The example
    below uses a NHWC = (1, 64, 2048, 3) image as an input.

    +-------------+------------------------------------------+---------------------------------------------+
    | VGSL        | DESCRIPTION                              | Output size (NHWC)     | Output size (NCHW) |
    +=============+==========================================+=============================================+
    | Cr3,13,32   | kernel filter 3x13, 32 activations relu  | 1, 64, 2048, 32        | 1, 32, 64, 2048    |
    +-------------+------------------------------------------+---------------------------------------------+
    | Do0.1,2     | dropout prob 0.1 dims 2                  | -                      | -                  |
    +-------------+------------------------------------------+---------------------------------------------+
    | Mp2,2       | Max Pool kernel 2x2 stride 2x2           | 1, 32, 1024, 32        | 1, 32, 32, 1024    | 
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         | (same)                                   | 1, 16,  512, 32        | 1, 32, 16, 512     |
    | Cr3,9,64    | kernel filter 3x9, 64 activations relu   | 1, 16,  512, 64        | 1, 64, 16, 512     |
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         |                                          |                        |                    |
    | Mp2,2       |                                          | 1, 8, 256, 64          | 1, 64, 8, 256      |
    +-------------+------------------------------------------+---------------------------------------------+
    | Cr3,9,64    |                                          | 1, 8, 256, 64          | 1, 64, 8, 256      |
    | Do0.1,2     |                                          |                        |                    |
    +-------------+------------------------------------------+---------------------------------------------+
    | S1(1x0)1,3  | reshape (N,H,W,C) into N, 1, W,C*H       | 1, 1, 256, 64x8=512    | 1, 1024, 1, 256    |
    +-------------+------------------------------------------+---------------------------------------------+
    | Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400         | 1, 400, 1, 256     |
    |             | with 200 output channels                 | (either forward (f) or |                    |
    |             |                                          |reverse (r) would yield |                    |
    |             |                                          | 200-sized output)      |                    |
    +-------------+------------------------------------------+---------------------------------------------+
    | ...         | (same)                                   |                        |                    |
    | Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400         | 1, 400, 1, 256     |
    +-------------+------------------------------------------+---------------------------------------------+

    """
    default_model_spec = '[4,64,0,3 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'

    def __init__(self, alphabet:'Alphabet'=Alphabet(['a','b','c']), height=64, model=None, model_spec=default_model_spec, decoder=None, add_output_layer=True, train=False):

        # encoder 
        # during save/resume cycles, alphabet may be serialized into a list
        self.alphabet = Alphabet( alphabet ) if type(alphabet) is list else alphabet
        

        # initialize self.nn = torch Module
        if not model:

            # In kraken, TorchVGSLModel is used both for constructing a NN 
            # (parse(), build*(), ... methods) and for hiding it: it is
            # a wrapper-class for a 'nn:Module' property, to which a number
            # of NN-specific method calls (to()...) are forwarded, including
            #  + train(), eval()
            # what is the added value of this complexity?
            # + model modifications (append)
            # + more abstraction when handling hyper-parameters
            # Otherwise, better to manipulate the NN directly.

            # insert output layer if not already defined
            if re.search(r'O\S+ ?\]$', model_spec) is None and add_output_layer:
                model_spec = '[{} O1c{}]'.format( model_spec[1:-1], self.alphabet.maxcode + 1)
            model_spec = re.sub(r'\[(\d+),\d+', '[\\1,{}'.format(height), model_spec )


            self.net = TorchVGSLModel( model_spec ).nn
        else:
            self.net = model
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / batch_size
        self.net.to( self.device )

        # decoder
        self.decoder = self.decode_greedy if decoder is None else decoder


        # a list of dictionaries = for each epoch: { CER, loss, duration }
        self.train_epochs = []
        self.validation_epochs = []

        self.net.train( mode=train )
        
        self.constructor_params = {
                # serialize the alphabet 
                'alphabet': self.alphabet.to_list(),
                'model': model,
                'model_spec': model_spec,
                'decoder': decoder,
                'add_output_layer': add_output_layer,
                'train': train
        }


    
    def forward(self, img_nchw: Tensor, widths_n: Tensor=None, split_output=False):
        """ The internal logics is entirely delegated to the layers wrapped 
        into the VGSL-defined module: by defaut, an instance of 
        `kraken.lib.layers.MultiParamSequential`.
        
        .. note:: 
            In spite of its name, this method is different from a torch.nn.Module.forward()
            function; it is just a convenience function, that is meant to be called explicitly,
            prior to the decoding stage (i.e. outside a training step), not as a callback.

        :param img_nchw: a batch of line images.
        :type img_nchw: Tensor
        :param widths: sequence of image lengths.
        :type widths: Tensor
        :param split_output: if True, only keep first half of the output channels (for pseudo-parallel nets).
        :type split_output: bool

        :returns: Tuple with (N,C,W) array and final output sequence lengths; C should match the number of character classes.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if self.device:
            img_nchw = img_nchw.to( self.device )

        with torch.no_grad():

            o_nchw, owidths = self.net(img_nchw, widths_n)
            if split_output:
                if o_nchw.shape[1]%2 != 0:
                    raise ValueError(f"Output tensor cannot be split: odd number of channels ({o_nchw.shape[1]})")
                o_nchw = o_nchw[:,:o_nchw.shape[1]//2]
            logger.debug("Network outputs have shape {} (lengths={}.".format(o_nchw.shape, widths_n ))
            outputs_ncw = o_nchw.detach().squeeze(2).float().cpu().numpy()
            logger.debug("-> distilled into Numpy array with shape {}.".format(outputs_ncw.shape))
            if owidths is not None:
                owidths = owidths.cpu().numpy()

            return (outputs_ncw, owidths)

        
    def decode_batch(self, outputs_ncw: np.ndarray, lengths: np.ndarray=None):
        """ Decode a batch of network logits into labels.

        :param outputs_ncw: a network output batch (N,C,W) of length W where C matches the number of character classes.
        :type outputs_ncw: np.ndarray

        :rtype: List[List[Tuple[int,float]]]
        :returns: a list of N lists of W tuples `(label, score)` where the score is the max. logit. Eg.::

                [[(30, 0.045990627), (8, 0.04730653), (8, 0.048647244), (8, 0.049242754), (8, 0.049613364), ...],
                 [(8, 0.04726322), (8, 0.047953878), (8, 0.047865044), (8, 0.04712664), (8, 0.046230078), ... ],
                 ...
                ]
        """
        if lengths is not None:
            return [ self.decoder(o_cw[:,:lgth]) for o_cw, lgth in zip( outputs_ncw, lengths) ]
        else:
            return [ self.decoder( o_cw ) for o_cw in output_ncw ]


    @staticmethod
    def decode_greedy(outputs_cw: np.ndarray):
        """ Decode a single output frame (C,W) by choosing the class C with max. logit; model-independent.

        :param outputs_cw: a single output sequence (C,W) of length W where C matches the number of character classes.
        :type outputs_cw: np.ndarray

        :returns: a list of tuples (label, score)  
        :rtype: List[Tuple[int,float]]
        """
        labels = np.argmax( outputs_cw, 0 )
        scores = np.max( outputs_cw, 0 )
        #symbols = self.alphabet 
        return list(zip(labels, scores))


    def inference_task( self, img_nchw: Tensor, widths_n: Tensor=None, masks: Tensor=None, split_output=False) -> List[str]:
        """ Make predictions on a batch of images.

        :param img_nchw: a batch of images.
        :type img_nchw: Tensor
        :param widths_n: a 1D tensor of lengths.
        :type widths_n: Tensor
        :param split_output: if True, only keep first half of the output channels (for pseudo-parallel nets).
        :type split_output: bool

        :returns: a list of human-readable strings.
        :rtype: List[str]
        """
       
        assert isinstance( img_nchw, Tensor ) and len(img_nchw.shape) == 4
        assert isinstance( widths_n, Tensor) and len(widths_n) == img_nchw.shape[0]

        # raw outputs
        outputs_ncw, output_widths = self.forward( img_nchw, widths_n, split_output=split_output ) 

        # decoding labels and scores
        # [[(l1,s1), ...],[(l1,s1), ... ], ...]
        decoded_msgs = self.decode_batch( outputs_ncw, output_widths )
        
        # decoding symbols from labels
        return [ self.alphabet.decode_ctc( np.array([ label for (label,score) in msg ])) for msg in decoded_msgs ]



    def save(self, file_name: str):
        state_dict = self.net.state_dict()
        state_dict['train_mode'] = self.net.training
        state_dict['constructor_params'] = self.constructor_params
        state_dict['train_epochs'] = self.train_epochs
        state_dict['validation_epochs'] = self.validation_epochs
        torch.save( state_dict, file_name ) 


    @staticmethod
    def resume( file_name: str, **kwargs):
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            constructor_params = state_dict['constructor_params']
            del state_dict['constructor_params']
            train_epochs = state_dict["train_epochs"]
            del state_dict["train_epochs"]
            validation_epochs = state_dict["validation_epochs"]
            del state_dict["validation_epochs"]
            train_mode = state_dict["train_mode"]
            del state_dict["train_mode"]
            
        
            model = HTR_Model( **constructor_params )
            model.net.load_state_dict( state_dict )
            model.train_epochs = train_epochs
            model.validation_epochs = validation_epochs

            # switch net to train/eval mode
            model.net.train( mode=train_mode )

            return model
        return HTR_Model( **kwargs )

            

    def __repr__( self ):
        return "HTR_Model()"



def dummy():
    return True
