
# stdlib
from pathlib import Path
import logging
from typing import Union,Tuple,List
import re
import warnings
import itertools

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

# local
from kraken.vgsl import TorchVGSLModel
from libs.alphabet import Alphabet
import character_classes as cc


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


class HTR_Model():
    """
    Initializing and saving an HTR model, from VGSL specs.

    """

    default_model_spec = '[0,0,0,3 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'

    def __init__( self, 
                  alphabet:'Alphabet'=None,
                  net=None, 
                  model_spec=default_model_spec, 
                  decoder=None, 
                  add_output_layer=True,
                  train=False):
        """Initialize a new network wrapper.

        Args:
            alphabet (alphabet.Alphabet): the alphabet object, with encoding/decoding functionalities
            net (str): path of an existing, serialized network/Torch module
            model_spec (str): a VGSL specification for constructing a model.
            decoder (Callable[[np.ndarray], List[Tuple[int,float]]]: an alphabet-agnostic decoding function, 
                that decodes logits into labels.
            add_output_layer (bool): if True (default), add the output layer string to the VGSL spec.
            train (bool)): if True, set mode to train; default is False.
        """

        if alphabet is None:
            self.alphabet = Alphabet( cc.space_charset + cc.latin_charset + cc.punctuation_charset, case_insensitive=True )
        else:
            # during save/resume cycles, alphabet may be serialized into a list
            self.alphabet = Alphabet( alphabet ) if type(alphabet) is list else alphabet
        
        if net:
            self.net = self.load( net )
        
        else:
            # insert output layer if not already defined
            if re.search(r'O\S+ ?\]$', model_spec) is None and add_output_layer:
                model_spec = '[{} O1c{}]'.format( model_spec[1:-1], len(self.alphabet))
                print(model_spec)
            #model_spec = re.sub(r'\[(\d+),\d+', '[\\1,{}'.format(height), model_spec )

            self.model_spec = model_spec
            self.net = TorchVGSLModel( self.model_spec ).nn
        
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
                'net': net,
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

        Args:
            img_nchw (Tensor): a batch of line images.
            widths (Tensor): sequence of image lengths.
            split_output (bool): if True, only keep first half of the output channels (for pseudo-parallel nets).

        Returns:
            Tuple[np.ndarray, np.ndarray]: pair with (N,C,W) array and final output sequence lengths; C should match the number of character classes.
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

        Args:
            outputs_ncw (np.ndarray): a network output batch (N,C,W) of length W where C matches the number of character classes.

        Returns:
            List[List[Tuple[int,float]]] : a list of N lists of W tuples `(label, score)` where the score is the max. logit. Eg.::

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

        Args:
            outputs_cw (np.ndarray): a single output sequence (C,W) of length W where C matches the number of character classes.

        Returns:
            List[Tuple[int,float]]: a list of tuples (label, score)  
        """
        labels = np.argmax( outputs_cw, 0 )
        scores = np.max( outputs_cw, 0 )
        #symbols = self.alphabet 
        return list(zip(labels, scores))


    def inference_task( self, img_nchw: Tensor, widths_n: Tensor=None, masks: Tensor=None, split_output=False)->Tuple[List[str], np.ndarray]:
        """ Make predictions on a batch of images.

        Args:
            img_nchw (Tensor): a batch of images.

            widths_n (Tensor): a 1D tensor of lengths.

            split_output (bool): if True, only keep first half of the output channels (for pseudo-parallel nets).
        
        Returns:
             Tuple[List[str], np.ndarray]: A pair of lists: 
                + the human-readable predicted strings, post CTC-decoding
                + for diagnosis: a (N,W) array where each row is a sequence of logits; each logit is the max. score
                  for each, null-separated output subsequence.
        """
       
        assert isinstance( img_nchw, Tensor ) and len(img_nchw.shape) == 4
        assert isinstance( widths_n, Tensor) and len(widths_n) == img_nchw.shape[0]

        #self.net.to('cpu')
        # raw outputs
        outputs_ncw, output_widths = self.forward( img_nchw, widths_n, split_output=split_output ) 

        # decoding: lists of pairs (<integer label>, <score>): [[(l1,s1),(l2,s2), ...],[(l1,s1), ... ], ...]
        decoded_labels_and_scores = self.decode_batch( outputs_ncw, output_widths )

        # fast ctc-decoding
        mesgs = [ self.alphabet.decode_ctc( np.array([ label for (label,score) in msg ])) for msg in decoded_labels_and_scores ]
        # max score for each non-null char
        grouped_label_lists = [ itertools.groupby( lst, key=lambda x: x[0] ) for lst in decoded_labels_and_scores ]
        filtered_label_lists = [ itertools.filterfalse(lambda x: x[0]==self.alphabet.null_value, lst ) for lst in grouped_label_lists ]
        ctc_scores = [[ max(s)[1] for k,s in lst ] for lst in filtered_label_lists ]

        assert all( len(mesg)==len(ctc_score) for (mesg,ctc_score) in zip(mesgs, ctc_scores) )

        max_width = max( len(msg) for msg in mesgs) 
        ctc_scores_nw = np.stack([ np.pad( np.array( lst ), (0,max_width-len(lst))) for lst in ctc_scores ])

        return (mesgs, ctc_scores_nw)


    def save(self, file_name: str):
        state_dict = self.net.state_dict()
        state_dict['train_mode'] = self.net.training
        state_dict['constructor_params'] = self.constructor_params
        state_dict['train_epochs'] = self.train_epochs
        state_dict['validation_epochs'] = self.validation_epochs
        torch.save( state_dict, file_name ) 


    @staticmethod
    def resume( file_name: str, reset_epochs=False, **kwargs):
        """ Resume a training task

        Args:
            file_name (str): a serialized Torch module dictionary.
        """
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

            if not reset_epochs:
                model.train_epochs = train_epochs
                model.validation_epochs = validation_epochs 

            # switch net to train/eval mode
            model.net.train( mode=train_mode )

            return model
        return HTR_Model( **kwargs )

    @staticmethod
    def load( file_name: str, **kwargs):
        """ Load an existing model, for evaluation
        """
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            constructor_params = state_dict['constructor_params']
            for k in ('constructor_params', 'train_epochs', 'validation_epochs', 'train_mode' ):
                if k in state_dict:
                    del state_dict[ k ]

            model = HTR_Model( **constructor_params )
            model.net.load_state_dict( state_dict )
            # evaluation mode
            model.net.train( mode=False )
            return model
        else:
            raise FileNotFoundError(f"Serialized model {file_name} not to be found.")


    def __repr__( self ):
        return "HTR_Model()"



def dummy():
    return True
