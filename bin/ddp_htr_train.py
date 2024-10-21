#!/usr/bin/env python3
import torch
import fargv
import sys
from didip_handwriting_datasets import monasterium
from didip_handwriting_datasets.alphabet import Alphabet
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import time
from tqdm import tqdm



root = Path(__file__).parents[1] 
sys.path.append( str(root) )


import logging_utils
from model_htr import HTR_Model

# local logger
import logging
# root logger
logging.basicConfig(stream=sys.stdout, level='INFO', format="%(asctime)s - %(funcName)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter

p = {
    "appname": "htr_train",
    "batch_size": 2,
    "img_height": 128,
    "img_width": 2048,
    "max_epoch": 200,
    "dataset_path_train": [root.joinpath('tests','data','polygons', 'monasterium_ds_train.tsv'), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "dataset_path_validate": [root.joinpath('tests','data','polygons', 'monasterium_ds_validate.tsv'), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "validation_freq": 100,
    "save_freq": 100,
    "resume_fname": 'model_save.ml',
    "mode": 'train',
    "auxhead": [True, 'Combine output with CTC shortcut'],
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
    from kraken import vgsl

    model_spec_rnn_top = vgsl.build_spec_from_chunks(
            [ ('Input','0,0,0,3'),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do O1c{classes}' ) ],
          height = args.img_height, classes=n_classes)

    model_spec_rnn_and_shortcut = vgsl.build_spec_from_chunks(
            [ ('Input','[0,0,0,3'),
              ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head and shortcut', '([Lbx256 Do0.2,2 Lbx256 Do0.2,2 Lbx256 Do O1c{classes}] [Do Cr1,3,{classes}])]' ) ],
          height = args.img_height, classes=n_classes)

    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already computes the softmax )

    if args.auxhead: 
        def criterion(y, t, ly, lt):
            """ Assume that y the (N,2*<n classes>,1,W)-concatenation of the two tensors we want to combine"""
            y1, y2 = y[:,:n_classes], y[:,nclasses]
            return (torch.nn.CTCLoss(zero_infinity=True)(y1, t, ly, lt) + 0.1 * zero_infinity=True)(y2, t, ly, lt))/batch_size
    else:
        criterion = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(y, t, ly, lt) / args.batch_size

   

    #optimizer = torch.optim.AdamW(list(self.net.parameters(), lr=1e-3, weight_decay=0.00005))

    #self.net.train()

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, shuffle=True) 
    eval_loader = DataLoader( ds_val, batch_size=args.batch_size)

    model = HTR_Model.resume( args.resume_fname, 
                              alphabet=ds_train.alphabet, 
                             height=args.img_height, 
                             model_spec=model_spec_rnn_top, 
                             add_output_layer=False )

    model.net.train()

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)

    t = time.time()
    
    # TensorBoard writer
    writer = SummaryWriter()
    
    
    def train_epoch(epoch ):

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

            # visualization
            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()

            optimizer.step()

            #if (iter_idx % args.validation_freq) == (args.validation_freq-1) or iter_idx 
            if batch_index == len(train_loader)-1: #epoch % args.validation_freq == 0:

                model.net.eval()

                b = next(iter(eval_loader))
                msg_strings = model.inference_task( b['img'], b['width'] )
                gt_strings = b['transcription']
                logger.info('epoch {}, iteration {}'.format( epoch, batch_index+1 ))
                for (img_name, gt_string, decoded_string ) in zip(  b['id'], b['transcription'], msg_strings ):
                        logger.info("{}: [{}] {}".format(img_name, decoded_string, gt_string ))

                model.net.train()


        mean_loss = torch.stack(epoch_losses).mean().item()       
        model.train_epochs.append({ "loss": mean_loss, "duration": time.time()-t })
        
        # save model every time epoch completes
        if epoch % args.save_freq == 0 or epoch == args.max_epoch-1:
            model.save( args.resume_fname )

        logger.info('Epoch {}, iteration {}, mean loss={}. Estimated time until completion: {}'.format( 
                epoch, 
                batch_index+1, 
                mean_loss,
                logging_utils.duration_estimate(epoch+1, args.max_epoch, model.train_epochs[-1]['duration']) ) )

    if args.mode == 'train':
    
        for epoch in range(1, 0 if args.dry_run else args.max_epoch ):
            train_epoch( epoch )


    elif args.mode == 'test':
        pass


    writer.flush()
    writer.close()
