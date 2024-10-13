from os import PathLike
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.ctc_decoder import greedy_decoder
import numpy as np
import re
from didip_handwriting_datasets.alphabet import Alphabet


from typing import Union,Tuple,List
import warnings



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

    def __init__(self, alphabet:'Alphabet'=Alphabet(['a','b','c']), model=None, model_spec=default_model_spec, decoder=None, add_output_layer=True):

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

            self.net = TorchVGSLModel( model_spec ).nn
        else:
            self.net = model

        self.net.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / batch_size
        self.net.to( self.device )

        # decoder
        self.decoder = self.decode_greedy if decoder is None else decoder

        self.validation_epochs = {}
        self.train_epochs = []

        self.constructor_params = {
                # serialize the alphabet 
                'alphabet': self.alphabet.to_list(),
                'model': model,
                'model_spec': model_spec,
                'decoder': decoder,
                'add_output_layer': add_output_layer,
        }



    def forward(self, img_nchw: Tensor, widths_n: Tensor=None):
        """
        The internal logics is entirely delegated to the layers wrapped 
        into the VGSL-defined module: by defaut, an instance of 
        `kraken.lib.layers.MultiParamSequential`.
        Note: in spite of its name, this method is different from a torch.nn.Module.forward()
        function; it is meant to be called explicitly, not as a callback.

        Args:
            img_nchw (Tensor): a batch of line images
            widths (Tensor): sequence of image lengths

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with (N,C,W) array and
            final output sequence lengths; C should match the number of 
            character classes.
        """
        if self.device:
            img_nchw = img_nchw.to( self.device )

        # note the dereferencing: the actual NN is a property of the TorchVGSL object
        o, owidths = self.net(img_nchw, widths_n)
        outputs_ncw = o.detach().squeeze(2).float().cpu().numpy()
        if owidths is not None:
            owidths = owidths.cpu().numpy()
        return (outputs_ncw, owidths)

        
    def decode_batch(self, outputs_ncw: np.ndarray, lengths: np.ndarray=None):
        """
        Decode a batch of network logits into labels.

        Args:
            outputs_ncw (np.ndarray): a output batch (N,C,W) of length W where C
                    matches the number of character classes.
        Returns:
            List[List[Tuple[int,float]]]: a list of N lists of W tuples `(label, score)` where
            the score is the max. logit. Eg.

            ```
            [[(30, 0.045990627), (8, 0.04730653), (8, 0.048647244), (8, 0.049242754), (8, 0.049613364), ...],
             [(8, 0.04726322), (8, 0.047953878), (8, 0.047865044), (8, 0.04712664), (8, 0.046230078), ... ],
             ...
            ]
            ```
        """
        decoded = []
        for o_cw in outputs_ncw:
                decoded.append( self.decoder( o_cw ))
        if lengths is not None:
            for o_cw, lgth in zip( outputs_ncw, lengths):
                decoded.append( self.decoder( o_cw[:,:lgth])) 
        # batch 1, 10-classes, full-length
        return decoded


    @staticmethod
    def decode_greedy(outputs_cw: np.ndarray):
        """
        Decode a single output frame (C,W); model-independent.

        Args:
            outputs_cw (np.ndarray): a single output sequence (C,W) of length W where C
                    matches the number of character classes.
        Returns:
            List[Tuple[int,float]]: a list of tuples (label, score)  
        """
        labels = np.argmax( outputs_cw, 0 )
        scores = np.max( outputs_cw, 0 )
        #symbols = self.alphabet 
        return list(zip(labels, scores))


    def inference_task( self, img_nchw: Tensor, widths_n: Tensor=None, masks: Tensor=None):
       
        assert isinstance( img_nchw, Tensor ) and len(img_nchw.shape) == 4
        assert isinstance( widths_n, Tensor) and len(widths_n) == img_nchw.shape[0]

        # raw outputs
        outputs_ncw, output_widths = self.forward( img_nchw, widths_n ) 

        # decoding labels and scores
        # [[(l1,s1), ...],[(l1,s1), ... ], ...]
        decoded_msgs = self.decode_batch( outputs_ncw, output_widths )
        
        # decoding symbols from labels
        msg_strings = [ self.alphabet.decode_ctc( np.array([ label for (label,score) in msg ])) for msg in decoded_msgs ]

        return msg_strings

    
    def save(self, file_name: str):
        state_dict = self.net.state_dict()
        state_dict['constructor_params'] = self.constructor_params
        state_dict['validation_epochs'] = self.validation_epochs
        state_dict['train_epochs'] = self.train_epochs
        torch.save( state_dict, file_name ) 


    @staticmethod
    def resume( file_name: str, **kwargs):
        try:
            state_dict = torch.load(file_name, map_location="cpu")
            constructor_params = state_dict['constructor_params']
            del state_dict['constructor_params']
            validation_epochs = state_dict["validation_epochs"]
            del state_dict["validation_epochs"]
            train_epochs = state_dict["train_epochs"]
            del state_dict["train_epochs"]
        
            model = HTR_Model( **constructor_params )
            model.net.load_state_dict( state_dict )
            model.validation_epochs = validation_epochs
            model.train_epochs = train_epochs
            return model
        except FileNotFoundError:
            return HTR_Model( **kwargs )

            

    def __repr__( self ):
        return "HTR_Model()"



def dummy():
    return True
