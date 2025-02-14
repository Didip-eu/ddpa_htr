#!/usr/bin/env python3

# stdlib

import sys
from pathlib import Path
import time
import logging
import random
import itertools
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


"""
Todo:
"""


root = Path(__file__).parents[1] 
sys.path.append( str(root) )

from libs import metrics, transforms as tsf, list_utils as lu
from model_htr import HTR_Model
from kraken import vgsl
from libs.charters_htr import ChartersDataset
import character_classes as cc

# local logger
# root logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


p = {
    "appname": "htr_train",
    "batch_size": 2,
    "img_height": 128,
    "img_width": 2048,
    "max_epoch": 250,
    "dataset_path_train": [str(root.joinpath('data','current_working_set', 'charters_ds_train.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    "dataset_path_validate": [str(root.joinpath('data','current_working_set', 'charters_ds_validate.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    "dataset_path_test": [str(root.joinpath('data','current_working_set', 'charters_ds_test.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    #"ignored_chars": [ cc.superscript_charset + cc.diacritic_charset, "Lists of characters that should be ignored (i.e. filtered out) at encoding time." ], 
    "ignored_chars": [[], "Lists of characters that should be ignored (i.e. filtered out) at encoding time." ], 
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "validation_freq": 100,
    "save_freq": 1,
    "resume_fname": 'model_save.mlmodel',
    "mode": ('train', 'validate', 'test'),
    "auxhead": [False, 'Combine output with CTC shortcut'],
}


def duration_estimate( iterations_past, iterations_total, current_duration ):
    time_left = time.gmtime((iterations_total - iterations_past) * current_duration)
    return ''.join([
     '{} d '.format( time_left.tm_mday-1 ) if time_left.tm_mday > 1 else '',
     '{} h '.format( time_left.tm_hour ) if time_left.tm_hour > 0 else '',
     '{} mn'.format( time_left.tm_min ) ])

def collate_with_expansion_masks( data_list ):
    """
    Without a custom collate, the default collation function would try to stack the lists of masks.
    """
    batch_dict = {}
    for k in ('img','mask'):
        batch_dict[k] = torch.stack([ sample[k] for sample in data_list ], 0)
    for k in ('height', 'width'):
        batch_dict[k] = torch.tensor( [ sample[k] for sample in data_list])
    for k in ('transcription', 'expansion_masks','id'):
        batch_dict[k] = [ sample[k] for sample in data_list ]
    return batch_dict

if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))

    #------------- Model ------------
    model_spec_rnn_top = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,3'),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do')],
          height = args.img_height)

    model = HTR_Model.resume( args.resume_fname, 
                             #height=args.img_height, 
                             model_spec=model_spec_rnn_and_shortcut if args.auxhead else model_spec_rnn_top,
                             add_output_layer=True ) 

    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already computes the softmax )

    criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y, t, ly, lt) / args.batch_size
   
    if args.dataset_path_train == '' or args.dataset_path_validate == '':
        sys.exit()

    #-------------- Dataset ---------------

    filter_transcription = lambda s: ''.join( itertools.filterfalse( lambda c: c in lu.flatten( args.ignored_chars ), s))

    ds_train = ChartersDataset( task='htr', shape='polygons',
        from_line_tsv_file=args.dataset_path_train, expansion_masks=True,
        transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
        target_transform=filter_transcription,)

    logger.debug( str(ds_train) )

    ds_val = ChartersDataset( task='htr', shape='polygons',
        from_line_tsv_file=args.dataset_path_validate, expansion_masks=True,
        transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
        target_transform=filter_transcription,)

    logger.debug( str(ds_val) )

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, collate_fn=collate_with_expansion_masks, shuffle=True) 
    val_loader = DataLoader( ds_val, batch_size=args.batch_size, collate_fn=collate_with_expansion_masks)

    # ------------ Training features ----------

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*args.max_epoch), int(.75*args.max_epoch)])
    
    writer = SummaryWriter()
    
    
    best_cer, best_wer, best_mer, best_mec, best_cer_epoch = 1.0, 1.0, 1.0, 0.0, -1
    if model.validation_epochs:
        best_cer, best_wer, best_mer, best_mec, best_cer_epoch = [ model.validation_epochs[-1][k] for k in ('best_cer', 'best_wer', 'best_mer', 'best_mec' 'best_cer_epoch') ]
    
    def sample_prediction_log( epoch:int, cut:int ):
        model.net.eval()

        b = next(iter(val_loader))
   
        msg_strings = model.inference_task( b['img'][:cut], b['width'][:cut], split_output=args.auxhead )
        gt_strings = b['transcription'][:cut]
        logger.info('epoch {}'.format( epoch ))
        for (img_name, gt_string, decoded_string ) in zip(  b['id'][:cut], b['transcription'][:cut], msg_strings ):
                logger.info("{}:\n\t[{}]\n\t {}".format(img_name, decoded_string, gt_string ))

        model.net.train()

    def validate( data_loader):
        """ Validation step: runs inference on the validation set.

        """

        model.net.eval()
        
        batches = iter( data_loader )
        cer = 0.0
        wer = 0.0
        mer = 0.0 # cer that does not consider masked parts.
        mec = 0.0 # contribution of the masked part to the character errors.

        for batch_index in tqdm( range(len(batches))):
            batch = next(batches)
            img, lengths, transcriptions, expansion_masks = ( batch[k] for k in ('img', 'width', 'transcription', 'expansion_masks') )
            
            # reduce charset
            transcriptions = [ model.alphabet.reduce(t) for t in transcriptions ]
            predictions = model.inference_task( img, lengths, split_output=args.auxhead )

            batch_cer, batch_wer, _, batch_mer, batch_mec = metrics.cer_wer_ler_with_masks( predictions, transcriptions, expansion_masks )
            cer += batch_cer
            wer += batch_wer
            mer += batch_mer
            mec += batch_mec

        mean_cer, mean_wer, mean_mer, mean_mec = cer/len(batches), wer/len(batches), mer/len(batches), mec/len(batches)

        model.net.train()
        return (mean_cer, mean_wer, mean_mer, mean_mec)


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
            labels, target_lengths = model.alphabet.encode_batch( transcriptions, padded=False ) 
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
        

    ################# TRAIN ####################
    if args.mode == 'train':
    
        model.net.train()

        epoch_start = len( model.train_epochs )
        if epoch_start > 0: 
            logger.info(f"Resuming training for epoch {epoch_start}")

        for epoch in range(epoch_start, 0 if args.dry_run else args.max_epoch ):
            train_epoch( epoch )

            if epoch % args.save_freq == 0 or epoch == args.max_epoch-1:
                model.save( args.resume_fname )

            cer, wer, mer, mec = validate()

            if cer <= best_cer:
                best_cer, best_cer_epoch, best_mer, best_mec = cer, epoch, mer, mec

                best_mer = mer # only make sense wr/ CER under consideration
                model.save( args.resume_fname + '.best' )
                # for easy check on the last best epoch
                #symlink_path = Path( args.resume_fname + f'.best.epoch-{epoch}' )
                #symlink_path.unlink( missing_ok=True )
                #symlink_path.symlink_to( args.resume_fname + '.best' )
            if wer <= best_wer:
                best_wer = wer

            model.validation_epochs.append({'cer': cer, 'best_cer': best_cer, 'best_cer_epoch': best_cer_epoch,
                                            'wer': wer, 'best_wer': best_wer, 
                                            'mer': mer, 'best_mer': best_mer, 'mec': mec, 'best_mec': best_mec})

            writer.add_scalar("CER/validate", cer, epoch)
            writer.add_scalar("WER/validate", wer, epoch)
            writer.add_scalar("MER/validate", mer, epoch)
                
            logger.info('Epoch {}, mean loss={:3.3f}; CER={:1.3f}, WER={:1.3f}, MER={:1.3f} (contribution: {:1.3f}). Estimated time until completion: {}'.format( 
                    epoch, 
                    model.train_epochs[-1]['loss'],
                    cer, wer, mer, mec,
                    duration_estimate(epoch+1, args.max_epoch, model.train_epochs[-1]['duration']) ) )
            logger.info('Best epoch={} with CER={} (with MER={} (contribution: {:1.3f})'.format( best_cer_epoch, best_cer, best_mer, best_mec ))

            scheduler.step()

    ################### TEST/VALIDATE ################
    elif args.mode == 'validate':

        cer, wer, mer = validate(val_loader)
        logger.info('CER={:1.3f}, WER={:1.3f}, MER={:1.3f} (contribution: {:1.3f})'.format( cer, wer, mer, mec ))

    elif args.mode == 'test':

        ds_test = ChartersDataset(shape='polygons',
            from_line_tsv_file=args.dataset_path_test,
            transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
            target_transform=filter_transcription,)
        test_loader = DataLoader( ds_test, batch_size=args.batch_size)

        cer, wer, mer = validate(test_loader)
        logger.info('CER={:1.3f}, WER={:1.3f}, MER={:1.3f} (contribution: {:1.3f})'.format( cer, wer, mer, mec ))


    writer.flush()
    writer.close()
