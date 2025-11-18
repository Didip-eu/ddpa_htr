#!/usr/bin/env python3

# stdlib

import sys
from pathlib import Path
import time
import logging
import random
import itertools
import re
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
+ clearer training/validation loop 
+ integrate PyLemmatizer to feeding logic
"""


root = Path(__file__).parents[1] 
sys.path.append( str(root) )

from libs import metrics, transforms as tsf, list_utils as lu, visuals
from libs.train_utils import split_set, duration_estimate
from model_htr import HTR_Model
from kraken import vgsl
from libs.charter_htr_datasets import HTRLineDataset
import character_classes as cc

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


p = {
    "appname": "htr_train",
    "batch_size": 8,
    "input_channels": 3,
    "img_height": 128,
    "img_width": 2048,
    "max_epoch": 200,
    "patience": 50,
    "img_paths": [set([]), "Line image samples (and implicit metadata files) from which to build the training, validation and testing sets."],
    "dataset_path": ['', "Directory with line image samples (and implicit metadata files) from which to build the training, validation and testing sets."],
    "img_file_suffix": '.png',
    "gt_file_suffix": '.gt.txt',
    "from_tsv": ['', "To build the train and validation subsets, look for TSV files (train.tsv and val.tsv) in the image folder."],
    "to_tsv": [False, "Store the training and validation sample data as TSV files (respectively as 'train.tsv' and 'val.tsv' in the same folder as the training files)."],
    "padding_style": [('median', 'noise', 'zero'), "Line padding style."],
    "ignored_chars": [ cc.superscript_charset + cc.diacritic_charset, "Lists of characters that should be ignored (i.e. filtered out) at encoding time." ], 
    "decoder": [('greedy','beam-search'), "Decoding layer: greedy or beam-search."],
    "lr": 1e-3,
    "dry_run": [0, "1: Load dataset and model but do not actually train, 2: same, but also display the validation samples."],
    "scheduler": 1,
    "scheduler_patience": 10,
    "scheduler_cooldown": 5,
    "scheduler_factor": 0.8,
    "device": [("cpu","cuda"), "Computing device"],
    "reset_epochs": [ False, "Ignore the epoch data stored in the model file - use for fine-tuning an existing model on a different dataset."],
    "resume_file": 'last.mlmodel',
    "mode": ('train', 'test'),
    "confusion_matrix": 0,
    "sample_log_window": [4, "How many samples should be decoded for end-of-epoch logging;"],
    "auxhead": [False, '([BROKEN]Combine output with CTC shortcut'],
}


if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))
    if args.dry_run:
        import matplotlib.pyplot as plt


    hyper_params = { varname:v for varname,v in vars(args).items() if varname in (
        'batch_size',
        'lr','scheduler', 'scheduler patience','scheduler_cooldown','schedular_factor',
        'max_epoch','patience'
    )}

    #------------- Model ------------
    model_spec_rnn_top = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,{}'.format(args.input_channels)),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do')],
          height = args.img_height)

    
    model = HTR_Model.resume( args.resume_file, 
                             #height=args.img_height, 
                             model_spec=model_spec_rnn_and_shortcut if args.auxhead else model_spec_rnn_top,
                             reset_epochs=args.reset_epochs,
                             add_output_layer=True,
                             device=args.device)
   
    if args.decoder=='beam-search': # this overrides whatever decoding function has been used during training
        model.decoder = HTR_Model.decode_beam_search

    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already computes the softmax )

    criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True, reduction='sum')(y, t, ly, lt) / hyper_params['batch_size']
   
    #-------------- Dataset ---------------

    # to be deprecated
    filter_transcription = lambda s: ''.join( itertools.filterfalse( lambda c: c in lu.flatten( args.ignored_chars ), s))

    resize_func = Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ])

    imgs_train, lbls_train, imgs_val, lbl_val = [], [], [], []
    
    # Option 1: a directory of images files
    if args.dataset_path:
        args.dataset_path = Path( args.dataset_path)
        # Compose each subset from the existing TSV
        if args.from_tsv:
            train_tsv_path, val_tsv_path = [ Path( args.dataset_path ).joinpath( tsv_filename ) for tsv_filename in ('train.tsv', 'val.tsv') ] 
            assert train_tsv_path.exists() and val_tsv_path.exists()
            ds_train, ds_val = [ HTRLineDataset( from_tsv_file=tsv_path, padding_style=args.padding_style, transform=resize_func, target_transform=filter_transcription,) for tsv_path in ( train_tsv_path, val_tsv_path ) ]
            logger.debug("Constructed subsets in {} from TSV files {} and {}".format( args.dataset_path, train_tsv_path, val_tsv_path))
        # ... or include all images
        else:
            imgs = list( Path(args.dataset_path).glob('*{}'.format(args.img_file_suffix)))
    # Option 2: a list of images
    elif args.img_paths:
        imgs = list( args.img_paths )

    # Given all images, split
    if not args.from_tsv:
        if not len(list( imgs )):
            logger.warning("Could not find valid training samples.")
            sys.exit()
        lbls =  [ re.sub(r'{}$'.format(args.img_file_suffix), args.gt_file_suffix, str(img_path)) for img_path in imgs ]
        # split sets
        imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
        imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

        ds_train = HTRLineDataset( 
                from_line_files=imgs_train, 
                padding_style=args.padding_style,
                transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
                target_transform=filter_transcription,
                to_tsv_file='train.tsv' if args.to_tsv else '',)

        ds_val = HTRLineDataset( 
                from_line_files=imgs_val,
                padding_style=args.padding_style,
                transform=Compose([ tsf.ResizeToHeight( args.img_height, args.img_width ), tsf.PadToWidth( args.img_width ) ]),
                target_transform=filter_transcription,
                to_tsv_file='val.tsv' if args.to_tsv else '',)
    
    logger.debug( str(ds_val) )
    logger.debug( str(ds_train) )

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader( ds_val, batch_size=args.batch_size)


    # ------------ Training features ----------

    optimizer = torch.optim.AdamW(list(model.net.parameters()), hyper_params['lr'], weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*hyper_params['max_epoch']), int(.75*hyper_params['max_epoch'])])
    
    writer = SummaryWriter()
    
    best_cer, best_wer, best_epoch = 1.0, 1.0, -1
    if model.epochs:
        best_cer, best_wer, best_epoch = [ model.epochs[-1][k] for k in ('best_cer', 'best_wer', 'best_epoch') ]
    
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

    def validate( confusion_matrix=False ):
        """ Test/validation step
        """
        model.net.eval()
        
        batches = iter( val_loader )
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

    def train_epoch(epoch, dry_run=False ):
        """ Training step.

        :param epoch: epoch index.
        :type epoch: int
        """
        t = time.time()

        epoch_losses = []
        batches = iter( train_loader )
        for batch_index in tqdm ( range(len( batches ))):
            batch = next( batches )
            img_nwhc, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            labels, target_lengths = model.alphabet.encode_batch( transcriptions, padded=False ) 
            img_nwhc, labels = img_nwhc.to( args.device ), labels.to( args.device )


            if args.dry_run > 0 and args.device=='cpu':
                plt.close()
                fig, ax = plt.subplots(len(batch), 1)
                for i, label in zip(range(len(batch)), labels):
                    logger.debug("{},{}".format( type(img[i]), transcriptions[i]))
                    ax[i].imshow( img[i].permute(1,2,0))
                plt.show()
                continue

            optimizer.zero_grad()

            outputs_nchw, output_lengths_n = model.net( img_nwhc, lengths )
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
            sample_prediction_log( epoch, min(args.sample_log_window, hyper_params['batch_size']))


        return None if dry_run else torch.stack(epoch_losses).mean().item()       
        # visualization
        #writer.add_scalar("Loss/train", mean_loss, epoch)
        #model.epochs.append({ "loss": mean_loss, "duration": time.time()-t })
        
    
    ########### TRAIN ################
    if args.mode == 'train':
    
        model.net.train()

        epoch_start = len( model.epochs )
        if epoch_start > 0: 
            logger.info(f"Resuming training for epoch {epoch_start}")

        for epoch in range(epoch_start, hyper_params['max_epoch'] ):

            epoch_start_time = time.time()
            mean_training_loss = train_epoch( epoch, args.dry_run )
            cer, wer = validate()

            if args.dry_run:
                continue

            model.save( args.resume_file )

            if cer <= best_cer:
                logger.info("Validation CER ({}) < best CER ({}): updating best model.".format( cer, best_cer ))
                best_cer, best_epoch = cer, epoch
                model.save( 'best.model' )
            if wer <= best_wer:
                best_wer = wer

            model.epochs.append({'loss': mean_training_loss, 'cer': cer, 'best_cer': best_cer, 'best_epoch': best_epoch,
                                 'wer': wer, 'best_wer': best_wer,
                                 'lr': scheduler.get_last_lr()[0], 'duration': time.time()-epoch_start_time,
                                 })
            writer.add_scalar("CER/validate", cer, epoch)
            writer.add_scalar("WER/validate", wer, epoch)
                
            logger.info('Epoch {}, mean loss={:3.3f}; CER={:1.4f}, WER={:1.3f}. Best epoch: {} (cer={}, wer={}) - Time left: {}'.format( 
                    epoch, 
                    model.epochs[-1]['loss'],
                    cer, wer, 
                    best_epoch, best_cer, best_wer,
                    duration_estimate(epoch+1, hyper_params['max_epoch'], model.epochs[-1]['duration']) ) )
            if epoch-best_epoch > hyper_params['patience']:
                logger.info("No improvement since epoch {}: early exit.".format(best_epoch))
                break
            if hyper_params['scheduler']:
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
