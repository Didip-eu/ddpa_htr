#!/usr/bin/env python3

# stdlib

import sys
from pathlib import Path
import time
import logging
import random
from typing import NamedTuple
# 3rd-party
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm
# didip
import fargv
from didip_handwriting_datasets import monasterium
from didip_handwriting_datasets.alphabet import Alphabet

"""
Todo:

"""


root = Path(__file__).parents[1] 
sys.path.append( str(root) )

import logging_utils
from model_htr import HTR_Model
from kraken import vgsl
import metrics

# local logger
# root logger
logging.basicConfig(stream=sys.stdout, level='INFO', format="%(asctime)s - %(funcName)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


p = {
    "appname": "htr_train",
    "batch_size": 2,
    "img_height": 128,
    "img_width": 2048,
    "max_epoch": 200,
    "dataset_path_train": [str(root.joinpath('tests','data','polygons', 'monasterium_ds_train.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "dataset_path_validate": [str(root.joinpath('tests','data','polygons', 'monasterium_ds_validate.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "validation_freq": 100,
    "save_freq": 100,
    "resume_fname": 'model_save.ml',
    "mode": ('train', 'validate', 'test'),
    "auxhead": [False, 'Combine output with CTC shortcut'],
}




if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))


    #-------------- Dataset ---------------
    # Alphabet is to be found in the same directory as the TSV file:
    # the two datasets below share the same directory, and consequently, their alphabet
    ds_train = monasterium.MonasteriumDataset( task='htr', shape='polygons',
        from_tsv_file=args.dataset_path_train,
        transform=Compose([ monasterium.ResizeToHeight( args.img_height, args.img_width ), monasterium.PadToWidth( args.img_width ) ]))

    logger.debug( str(ds_train) )

    ds_val = monasterium.MonasteriumDataset( task='htr', shape='polygons',
        from_tsv_file=args.dataset_path_validate,
        transform=Compose([ monasterium.ResizeToHeight( args.img_height, args.img_width ), monasterium.PadToWidth( args.img_width ) ]))

    logger.debug( str(ds_val) )

    n_classes = len( ds_train.alphabet )

    #------------- Models ------------

    model_spec_rnn_top = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,3'),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do O1c{classes}' ) ],
          height = args.img_height, classes=n_classes)

    model_spec_rnn_and_shortcut = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,3'),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head and shortcut', '([Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do O1c{classes}] [Do Cr1,3,{classes}])' ) ],
          height = args.img_height, classes=n_classes)

    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already computes the softmax )

    criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y, t, ly, lt) / args.batch_size
    if args.auxhead: 
        def combined_ctc_loss(y, t, ly, lt):
            """ Assume that y the (W,N,2*<n classes>)-concatenation of the two tensors we want to combine"""
            y1, y2 = y[:,:,:n_classes], y[:,:,n_classes:]
            loss = torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y1, t, ly, lt)
            loss +=  0.1 * torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y2, t, ly, lt)
            return loss / args.batch_size
        criterion = combined_ctc_loss

   

    #optimizer = torch.optim.AdamW(list(self.net.parameters(), lr=1e-3, weight_decay=0.00005))

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, shuffle=True) 
    eval_loader = DataLoader( ds_val, batch_size=args.batch_size)

    model = HTR_Model.resume( args.resume_fname, 
                              alphabet=ds_train.alphabet, 
                             height=args.img_height, 
                             model_spec=model_spec_rnn_and_shortcut if args.auxhead else model_spec_rnn_top,
                             add_output_layer=False ) # to allow for parallel network


    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    # multi-step scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*args.max_epoch), int(.75*args.max_epoch)])

    
    # TensorBoard writer
    writer = SummaryWriter()
    
    class Metric(NamedTuple):
        value: float 
        epoch: int 
        
        def __lt__(self, other):
            return self.value <= other.value
        
    best_cer, best_wer = Metric(1.0, -1), Metric(1.0, -1)
    last_cer, last_wer = Metric(1.0, -1), Metric(1.0, -1)
    
    def sample_prediction_log( epoch:int, cut:int ):
        model.net.eval()

        b = next(iter(eval_loader))
   
        msg_strings = model.inference_task( b['img'][:cut], b['width'][:cut], split_output=args.auxhead )
        gt_strings = b['transcription'][:cut]
        logger.info('epoch {}'.format( epoch ))
        for (img_name, gt_string, decoded_string ) in zip(  b['id'][:cut], b['transcription'][:cut], msg_strings ):
                logger.info("{}:\n\t[{}]\n\t {}".format(img_name, decoded_string, gt_string ))

        model.net.train()

    def validate():
        """ Validation step: runs inference on the validation set.

        """

        model.net.eval()
        
        batches = iter( eval_loader )
        cer = 0.0
        wer = 0.0

        for batch_index in tqdm( range(len(batches))):
            batch = next(batches)
            img, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            predictions = model.inference_task( img, lengths, split_output=args.auxhead )

            # predictions on encoded strings, not on raw GT
            predictions, transcriptions = [ [ eval_loader.dataset.alphabet.encode(ss) for ss in s ] for s in (predictions, transcriptions) ]
            batch_cer, batch_wer, _ = metrics.cer_wer_ler( predictions, transcriptions, word_separator=eval_loader.dataset.alphabet.get_code(' ') )
            cer += batch_cer
            wer += batch_wer

        mean_cer, mean_wer = cer/len(batches), wer/len(batches)

        model.net.train()
        return (mean_cer, mean_wer)

    def train_epoch(epoch ):
        """ Training step.

        :param epoch: epoch index.
        :type epoch: int
        """
        t = time.time()

        epoch_losses = []
        batches = iter( train_loader )
        for batch_index in tqdm ( range(len( batches ))):
            batch = next( batches )
            img, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            labels, target_lengths = ds_train.alphabet.encode_batch( transcriptions, padded=False ) 
            img, labels = img.cuda(), labels.cuda()


            if args.dry_run:
                continue

            optimizer.zero_grad()

            outputs_nchw, output_lengths_n = model.net( img, lengths )
            logger.debug(f"Output_nchw shape: {outputs_nchw.shape} Lengths: {output_lengths_n}")
            outputs_ncw = outputs_nchw.squeeze(2)
            logger.debug(f"Output_ncw shape: {outputs_ncw.shape}")
            outputs_wnc = outputs_ncw.permute(2,0,1)
            logger.debug("Output_wnc shape:i {outputs_wnc.shape}")

            # Loss is a scalar
            # transposing inputs: (N, C, W) -> (W, N, C)
            loss = criterion( outputs_wnc.cpu(), labels, output_lengths_n, target_lengths )
            epoch_losses.append( loss.detach()) 

            loss.backward()

            optimizer.step()

            #if (iter_idx % args.validation_freq) == (args.validation_freq-1) or iter_idx 
            if batch_index == len(train_loader)-1: #epoch % args.validation_freq == 0:

                cut = 4 if args.batch_size >= 4 else args.batch_size
                sample_prediction_log( epoch, cut )


        mean_loss = torch.stack(epoch_losses).mean().item()       
        # visualization
        writer.add_scalar("Loss/train", mean_loss, epoch)
        model.train_epochs.append({ "loss": mean_loss, "duration": time.time()-t })
        

    if args.mode == 'train':
    
        model.net.train()

        for epoch in range(0, 0 if args.dry_run else args.max_epoch ):
            train_epoch( epoch )

            # save model every time epoch completes and best CER has improved
            if epoch % args.save_freq == 0 or epoch == args.max_epoch-1:

                cer, wer = validate()

                last_cer, last_wer = Metric( cer, epoch ), Metric( wer, epoch )
                if last_cer <= best_cer:
                    best_cer = last_cer
                    model.save( args.resume_fname )
                if last_wer <= best_wer:
                    best_wer = last_wer

            writer.add_scalar("CER/validate", last_cer.value, epoch)
            writer.add_scalar("WER/validate", last_wer.value, epoch)
                

            logger.info('Epoch {}, mean loss={:3.3f}; CER={:1.3f}, WER={:1.3f}. Estimated time until completion: {}'.format( 
                    epoch, 
                    model.train_epochs[-1]['loss'],
                    last_cer.value, last_wer.value, 
                    logging_utils.duration_estimate(epoch+1, args.max_epoch, model.train_epochs[-1]['duration']) ) )
            logger.info('Best epoch={} with CER={}.'.format( best_cer.epoch, best_cer.value))

            scheduler.step()

    elif args.mode == 'validate':
        cer, wer = validate()
        logger.info('CER={:1.3f}, WER={:1.3f}'.format( cer, wer ))

    elif args.mode == 'test':
        pass


    writer.flush()
    writer.close()
