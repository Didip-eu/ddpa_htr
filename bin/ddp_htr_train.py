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

p = {
    "appname": "htr_train",
    "batch_size": 2,
    "max_epoch": 200,
    "dataset_path": root.joinpath('tests','data','bbox'),
    "learning_rate": 1e-3,
}

        

if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    print(args)


    ctc_loss = lambda y, t, ly, lt: torch.nn.CTCLoss(zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / args.batch_size
    #model = 

    #optimizer = torch.optim.AdamW(list(self.net.parameters(), lr=1e-3, weight_decay=0.00005))

    #self.net.train()

    # Reminder: alphabet is to be found in the same directory as the TSV file
    ds = monasterium.MonasteriumDataset( task='htr', shape='bbox',
        from_tsv_file=args.dataset_path.joinpath('monasterium_ds_train.tsv'),
        transform=Compose([ monasterium.ResizeToHeight(64, 2048), monasterium.PadToWidth(2048) ]))

    train_loader = DataLoader( ds, batch_size=args.batch_size, shuffle=True) 

    model = HTR_Model( ds.alphabet )

    optimizer = torch.optim.AdamW(list(model.net.parameters()), args.learning_rate, weight_decay=0.00005)
    
    def train_epoch(epoch ):

        closs = []

        for iter_idx, batch in enumerate( train_loader ):
            img, lengths, transcriptions = ( batch[k] for k in ('img', 'width', 'transcription') )
            labels, target_lengths = ds.alphabet.encode_batch( transcriptions, padded=False ) 

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

        if len(closs) > 0:
            logger.info('Epoch {}, iteration {}: {}'.format( epoch, iter_idx+1, sum(closs)/len(closs) ))


    
    for epoch in range( args.max_epoch ):
        train_epoch( epoch )


