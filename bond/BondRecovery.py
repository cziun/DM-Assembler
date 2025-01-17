import time
import torch
import pytorch_lightning as pl

from bond.generator import Generator


class BondRecovery(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(BondRecovery, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.atom_dim = config.generate.atom_embedding_dim + \
                        config.generate.piece_embedding_dim + \
                        config.generate.pos_embedding_dim
        self.decoder = Generator(tokenizer, config.generate.atom_embedding_dim, config.generate.piece_embedding_dim,
                                 config.generate.max_pos, config.generate.pos_embedding_dim,
                                 config.generate.num_edge_type, config.generate.node_hidden_dim)
        self.total_time = 0

    def forward(self, batch, return_accu=False):
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr'] 
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos'] 
        x = self.decoder.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape
        in_piece_edge_idx = batch['in_piece_edge_idx']
        res = self.decoder(x=x, edge_index=edge_index[:, in_piece_edge_idx],  # do not include the edges to be predicted
                              edge_attr=edge_attr[in_piece_edge_idx],  # do not include the edges to be predicted
                              pieces=batch['pieces'], 
                              edge_select=batch['edge_select'],
                              golden_edge=batch['golden_edge'],  # only include edges to be predicted
                              return_accu=return_accu)
        return res

    def training_step(self, batch, batch_idx):
        st = time.time()
        loss = self.forward(batch)
        self.log('train_loss', loss)
        self.total_time += time.time() - st
        self.log('total time', self.total_time)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accu = self.forward(batch, return_accu=True)
        self.log('val_loss', loss)
        self.log('val_accu', accu)

    def test_step(self, batch, batch_idx):
        loss, accu = self.forward(batch, return_accu=True)
        self.log('test_loss', loss)
        self.log('test_accu', accu)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.generate.lr)
        return optimizer

    # interface
    def inference_single_z(self, motif_ids, adj_inter_motif, add_edge_th, device):
        return self.decoder.inference(motif_ids, adj_inter_motif, add_edge_th, device)
