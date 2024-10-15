import torch
import fargv
import sys
from didip_handwriting_datasets import monasterium
from didip_handwriting_datasets.alphabet import Alphabet
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path


root = Path(__file__).parents[1] 
sys.path.append( str(root) )


from model_htr import HTR_Model

# local logger
import logging
logger = logging.getLogger(__name__)
# root logger
logging.basicConfig(stream=sys.stdout, level='DEBUG', format="%(asctime)s - %(funcName)s: %(message)s", force=True)


p = {
    "appname": "htr_train",
    "batch_size": 2,
    "max_epoch": 200,
    "dataset_path": [root.joinpath('tests','data','bbox', 'monasterium_ds_train.tsv'), "TSV file containing the image paths and transcriptions. The parent folder is assumed to contain the alphabet (alphabet.tsv)."],
    "learning_rate": 1e-3,
    "dry_run": [False, "Iterate over the batches once, but do not run the network."],
    "eval_period": 20,
}

        

if __name__ == "__main__":

    logging.basicConfig(filename='ddp_htr_train;py', level=logging.DEBUG)

    args, _ = fargv.fargv( p )
    logger.debug("CLI arguments: {}".format( args ))


    ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size

    #optimizer = torch.optim.AdamW(list(self.net.parameters(), lr=1e-3, weight_decay=0.00005))

    #self.net.train()

    # Reminder: alphabet is to be found in the same directory as the TSV file
    ds = monasterium.MonasteriumDataset( task='htr', shape='bbox',
        from_tsv_file=args.dataset_path,
        transform=Compose([ monasterium.ResizeToHeight(64, 2048), monasterium.PadToWidth(2048) ]))
    logger.debug( str(ds) )

    train_loader = DataLoader( ds, batch_size=args.batch_size, shuffle=True) 

    model = HTR_Model( ds.alphabet )

    model.train = True

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    
    def train_epoch(epoch ):

        closs = []

        for iter_idx, batch in enumerate( train_loader ):
            img, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            labels, target_lengths = ds.alphabet.encode_batch( transcriptions, padded=False ) 
            img, labels = img.cuda(), labels.cuda()


            if args.dry_run:
                continue

            optimizer.zero_grad()

            outputs_nchw, output_lengths_n = model.net( img, lengths )
            print("Output_nchw shape =", outputs_nchw.shape, "Lengths shape =", output_lengths_n)
            outputs_ncw = outputs_nchw.squeeze(2)
            print("Output_ncw shape =", outputs_ncw.shape)
            outputs_wnc = outputs_ncw.permute(2,0,1)
            print("Output_wnc shape =", outputs_wnc.shape)

            # (N, C, W) -> (W, N, C)
            loss_val = ctc_loss( outputs_wnc.cpu(), labels, output_lengths_n, target_lengths )
            closs += [ loss_val.item() ]


            loss_val.backward()

            optimizer.step()

            if iter_idx % arg.eval_period == arg.eval_period-1:

                model.net.eval()


        if len(closs) > 0:
            logger.info('Epoch {}, iteration {}: {}'.format( epoch, iter_idx+1, sum(closs)/len(closs) ))


    
    for epoch in range( 0 if args.dry_run else args.max_epoch ):
        train_epoch( epoch )


