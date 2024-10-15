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


root = Path(__file__).parents[1] 
sys.path.append( str(root) )


from model_htr import HTR_Model

# local logger
import logging
# root logger
logging.basicConfig(stream=sys.stdout, level='INFO', format="%(asctime)s - %(funcName)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


p = {
    "appname": "htr_train",
    "batch_size": 2,
    "max_epoch": 200,
    "dataset_path_train": [root.joinpath('tests','data','bbox', 'monasterium_ds_train.tsv'), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "dataset_path_validate": [root.joinpath('tests','data','bbox', 'monasterium_ds_validate.tsv'), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv)."],
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "eval_period": 20,
    "mode": 'train',
}

        

if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))



    #ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    # (our model already compute the softmax )
    ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(y, t, ly, lt) / args.batch_size

    #optimizer = torch.optim.AdamW(list(self.net.parameters(), lr=1e-3, weight_decay=0.00005))

    #self.net.train()

    # Alphabet is to be found in the same directory as the TSV file:
    # the two datasets below share the same directory, and consequently, their alphabet
    ds_train = monasterium.MonasteriumDataset( task='htr', shape='bbox',
        from_tsv_file=args.dataset_path_train,
        transform=Compose([ monasterium.ResizeToHeight(64, 2048), monasterium.PadToWidth(2048) ]))

    logger.debug( str(ds_train) )

    ds_val = monasterium.MonasteriumDataset( task='htr', shape='bbox',
        from_tsv_file=args.dataset_path_validate,
        transform=Compose([ monasterium.ResizeToHeight(64, 2048), monasterium.PadToWidth(2048) ]))

    logger.debug( str(ds_val) )

    train_loader = DataLoader( ds_train, batch_size=args.batch_size, shuffle=True) 
    eval_loader = DataLoader( ds_val, batch_size=args.batch_size)

    model = HTR_Model( ds_train.alphabet )

    model.net.train()

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    
    def train_epoch(epoch ):

        epoch_losses = []
        t = time.time()

        for iter_idx, batch in enumerate( train_loader ):
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

            loss = ctc_loss( outputs_wnc.cpu(), labels, output_lengths_n, target_lengths )
            epoch_losses.append( loss.detach())

            loss.backward()

            optimizer.step()

            if iter_idx % args.eval_period == args.eval_period-1:

                model.net.eval()

                b = next(iter(eval_loader))
                msg_strings = model.inference_task( b['img'], b['width'] )
                gt_strings = b['transcription']
                print( gt_strings )
                print( msg_strings )

                model.net.train()

        mean_loss = torch.stack(epoch_losses).mean().item()       
        model.train_epochs.append({ "loss": mean_loss, "duration": time.time()-t })
        
        if len(epoch_losses) > 0:
            logger.info('Epoch {}, iteration {}, mean loss={}'.format( epoch, iter_idx+1, mean_loss ) )

    if args.mode == 'train':
    
        for epoch in range( 0 if args.dry_run else args.max_epoch ):
            train_epoch( epoch )


    elif args.mode == 'test':
        pass


