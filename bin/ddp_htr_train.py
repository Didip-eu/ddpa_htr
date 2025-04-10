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
import numpy as np
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

from libs import metrics, transforms as tsf, list_utils as lu, visuals
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
    "batch_size": 8,
    "input_channels": 3,
    "img_height": 128,
    "img_width": 2048,
    "max_epoch": 200,
    "dataset_path_train": [str(root.joinpath('data','current_working_set', 'charters_ds_train.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    "dataset_path_validate": [str(root.joinpath('data','current_working_set', 'charters_ds_validate.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    "dataset_path_test": [str(root.joinpath('data','current_working_set', 'charters_ds_test.tsv')), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions."],
    "ignored_chars": [ cc.superscript_charset + cc.diacritic_charset, "Lists of characters that should be ignored (i.e. filtered out) at encoding time." ], 
    "decoder": [('greedy','beam-search'), "Decoding layer: greedy or beam-search."],
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "validation_freq": 1,
    "save_freq": 1,
    "resume_fname": ['model_save.mlmodel', "Model *.mlmodel to load. By default, the epoch count will start from the epoch that has been last stored in this file's meta-data. To ignore this and reset the epoch count, set the -reset_epoch option."],
    "reset_epochs": [ False, "Ignore the the epoch data stored in the model file - use for fine-tuning an existing model on a different dataset."],
    "mode": ('train', 'test'),
    "confusion_matrix": 0,
    "auxhead": [False, '([BROKEN]Combine output with CTC shortcut'],
}


def duration_estimate( iterations_past, iterations_total, current_duration ):
    time_left = time.gmtime((iterations_total - iterations_past) * current_duration)
    return ''.join([
     '{} d '.format( time_left.tm_mday-1 ) if time_left.tm_mday > 0 else '',
     '{} h '.format( time_left.tm_hour ) if time_left.tm_hour > 0 else '',
     '{} mn'.format( time_left.tm_min ) ])


if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))

    #------------- Model ------------
    model_spec_rnn_top = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,{}'.format(args.input_channels)),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do')],
          height = args.img_height)

    model = HTR_Model.resume( args.resume_fname, 
                             #height=args.img_height, 
                             model_spec=model_spec_rnn_and_shortcut if args.auxhead else model_spec_rnn_top,
                             reset_epochs=args.reset_epochs,
                             add_output_layer=True,)
   
    if args.decoder=='beam-search': # this overrides whatever decoding function has been used during training
        model.decoder = HTR_Model.decode_beam_search

    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already computes the softmax )

    criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y, t, ly, lt) / args.batch_size
   
    if args.dataset_path_train == '' or args.dataset_path_validate == '':
        sys.exit()

    #-------------- Dataset ---------------

    filter_transcription = lambda s: ''.join( itertools.filterfalse( lambda c: c in lu.flatten( args.ignored_chars ), s))

    ds_train = ChartersDataset(
        from_line_tsv_file=args.dataset_path_train,
        line_padding_style='median',
        transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
        target_transform=filter_transcription,)

    logger.debug( str(ds_train) )

    ds_val = ChartersDataset(
        from_line_tsv_file=args.dataset_path_validate,
        line_padding_style='median',
        transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
        target_transform=filter_transcription,)

    logger.debug( str(ds_val) )

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader( ds_val, batch_size=args.batch_size)

    # ------------ Training features ----------

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*args.max_epoch), int(.75*args.max_epoch)])
    
    writer = SummaryWriter()
    
    best_cer, best_wer, best_cer_epoch = 1.0, 1.0, -1
    if model.validation_epochs:
        best_cer, best_wer, best_cer_epoch = [ model.validation_epochs[-1][k] for k in ('best_cer', 'best_wer', 'best_cer_epoch') ]
    
    def sample_prediction_log( epoch:int, cut:int ):
        model.net.eval()

        b = next(iter(val_loader))
   
        msg_strings, _ = model.inference_task( b['img'][:cut], b['width'][:cut], split_output=args.auxhead )
        gt_strings_redux = [ model.alphabet.reduce(s) for s in b['transcription'][:cut] ]
        logger.info('epoch {}'.format( epoch ))
        for (img, img_name, gt_str_raw, gt_str_redux, decoded_str ) in zip(  b['img'][:cut], b['id'][:cut], b['transcription'][:cut], gt_strings_redux, msg_strings ):
            logger.info("{}:\n\tPred: [{}]\n\tRedux: {}\n\t  Raw: {}".format(img_name, decoded_str, gt_str_redux, gt_str_raw ))
            writer.add_image(img_name, img )

        model.net.train()

    def validate( data_loader, confusion_matrix=False ):
        """ Test/validation step
        """

        model.net.eval()
        
        batches = iter( data_loader )
        cer = 0.0
        wer = 0.0

        conf_matrix, matrix_alph = None, None

        for batch_index in tqdm( range(len(batches))):
            batch = next(batches)
            img, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            # reduce charset
            transcriptions = [ model.alphabet.reduce(t) for t in transcriptions ]
            predictions, _ = model.inference_task( img, lengths, split_output=args.auxhead )

            batch_cer, batch_wer, _ = metrics.cer_wer_ler( predictions, transcriptions )
            cer += batch_cer
            wer += batch_wer
            
            if confusion_matrix:
                if conf_matrix is None:
                    matrix_alph = model.alphabet._utf_2_code_reduced()
                    print(matrix_alph)
                    conf_matrix = metrics.batch_char_confusion_matrix( transcriptions, predictions, matrix_alph )[0]
                else:
                    conf_matrix += metrics.batch_char_confusion_matrix( transcriptions, predictions, matrix_alph )[0]

        if confusion_matrix:
            import pickle
            conf_matrix /= len(batches)
            with open('confusion_matrix.pkl', 'wb') as pkl_file:
                pickle.dump({'cm': conf_matrix, 'alph': matrix_alph}, pkl_file )

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
        
    
    ########### TRAIN ################
    if args.mode == 'train':
    
        model.net.train()

        epoch_start = len( model.train_epochs )
        if epoch_start > 0: 
            logger.info(f"Resuming training for epoch {epoch_start}")

        for epoch in range(epoch_start, 0 if args.dry_run else args.max_epoch ):
            train_epoch( epoch )

            if epoch % args.save_freq == 0 or epoch == args.max_epoch-1:
                model.save( args.resume_fname )

            cer, wer = validate(val_loader)

            if cer <= best_cer:
                best_cer, best_cer_epoch = cer, epoch
                model.save( args.resume_fname + '.best' )
            if wer <= best_wer:
                best_wer = wer

            model.validation_epochs.append({'cer': cer, 'best_cer': best_cer, 'best_cer_epoch': best_cer_epoch,
                                            'wer': wer, 'best_wer': best_wer,})
            writer.add_scalar("CER/validate", cer, epoch)
            writer.add_scalar("WER/validate", wer, epoch)
                
            logger.info('Epoch {}, mean loss={:3.3f}; CER={:1.4f}, WER={:1.3f}. Estimated time until completion: {}'.format( 
                    epoch, 
                    model.train_epochs[-1]['loss'],
                    cer, wer, 
                    duration_estimate(epoch+1, args.max_epoch, model.train_epochs[-1]['duration']) ) )
            logger.info('Best epoch={} with CER={}.'.format( best_cer_epoch, best_cer))

            scheduler.step()


    ############# VALIDATE / TEST ############
    elif args.mode == 'test':
        
        ds_test = ChartersDataset(
            from_line_tsv_file=args.dataset_path_test,
            line_padding_style='median',
            transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
            target_transform=filter_transcription,)
        test_loader = DataLoader( ds_test, batch_size=args.batch_size)
        cer, wer = validate(test_loader, args.confusion_matrix)
        logger.info('CER={:1.4f}, WER={:1.3f}'.format( cer, wer ))


    writer.flush()
    writer.close()
