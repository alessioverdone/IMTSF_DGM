import torch
import torch.nn as nn

from src.config import Parameters
from src.layers.decoder import select_decoder
from src.layers.mds import DGMmodule
from src.layers.utils import build_graph_from_mask, select_gnn, build_graph_from_mask_v2


class QueryPooling(nn.Module):
    """
    QueryPooling risolve un problema specifico: dopo il GNN hai graph.x di shape (mask==1).sum(), F),
    cioè solo i nodi validi, numero variabile per ogni sample nel batch. Devi trasformarlo in qualcosa di shape fissa
    (B, N, F) per il decoder.

    Step by step:
    1. Padding: Raggruppa i nodi per sample (tramite batch index) e li mette in un tensore (B, N_max, F) con
        zero-padding. Crea una key_padding_mask booleana: True = posizione paddata (da ignorare nell'attention).
    2. Cross-Attention:
        Query: N_2 vettori learnable (B, N_2, F) — uno per ogni variabile della serie
        Key/Value: i nodi del grafo (B, N_max, F) con la mask
        Ogni query "interroga" l'insieme dei nodi osservati per quella serie e aggrega le loro feature. Le query sono
        learnable quindi il modello impara come aggregare.
    3. Output: out ha shape (B, N_2, F) — esattamente una rappresentazione per ogni variabile, indipendentemente da
        quanti nodi erano presenti. Viene reshapato a (B*N_2, F) e poi nel forward principale riportato a (B, N, D).

    Intuitivamente: è un'operazione di read-out del grafo che invece di fare mean/sum pooling usa attention con query
    fisse — una per variabile. Questo permette al modello di estrarre selettivamente informazioni rilevanti dai nodi
    osservati per ricostruire la rappresentazione di tutte le N variabili, anche quelle non osservate (che non avevano
    nodi nel grafo). È particolarmente adatto per multivariate irregolare proprio perché gestisce il numero variabile
    di osservazioni senza perdere informazione con un semplice mean pooling.

    """

    def __init__(self, N_2, F, num_heads=4):
        super().__init__()
        self.N_2 = N_2
        # N_2 query learnable
        self.queries = nn.Parameter(torch.randn(1, N_2, F))
        self.attn = nn.MultiheadAttention(F, num_heads, batch_first=True)

    def forward(self, x, batch):
        # x: (B*N_i, F) con N_i variabile
        B = batch.max().item() + 1

        # padding: porta ogni grafo a (B, N_max, F) con key_padding_mask
        counts = batch.bincount(minlength=B)          # (B,)
        N_max = counts.max().item()

        x_padded = torch.zeros(B, N_max, x.size(-1), device=x.device)
        mask = torch.ones(B, N_max, dtype=torch.bool, device=x.device)  # True = ignora

        idx = 0
        for b in range(B):
            n = counts[b].item()
            x_padded[b, :n] = x[idx:idx+n]
            mask[b, :n] = False  # questi sono validi
            idx += n

        # cross-attention: query learnable, key/value dai nodi
        q = self.queries.expand(B, -1, -1)            # (B, N_2, F)
        out, _ = self.attn(q, x_padded, x_padded, key_padding_mask=mask)
        # out: (B, N_2, F)
        return out.reshape(B * self.N_2, x.size(-1))  # (B*N_2, F)


# Graph-based Regression with Adaptive Pooling and Embeddings
class Grape(nn.Module):
    def __init__(self, args: Parameters):
        super(Grape, self).__init__()
        self.args = args
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None

        # Embedder
        self.relu = nn.ReLU()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, args.hid_dim)

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(args.hid_dim * 2, args.hid_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.hid_dim, args.hid_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.hid_dim, 1))

        # Decoder
        self.decoder = select_decoder(args)

        # Latent GNN
        self.gnn = select_gnn(args)

        # Pooling
        self.pool = QueryPooling(args.ndim,
                                 args.hid_dim,
                                 num_heads=args.pool_num_heads)

    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def preprocess_batch(self, X, truth_time_steps, mask):
        out_dict = dict()

        # Preprocess (patch input to normal input)
        if len(X.shape) == 4:
            X = X.squeeze()
            mask = mask.squeeze()
            truth_time_steps = truth_time_steps.squeeze()

        B, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 2, 1).unsqueeze(-1)  # permuto e aggiungo dimensione per espandere h_dim, [B, N, T_in, 1]
        # Data arranged in compact or non-compact way
        if not self.args.patch_ts:
            truth_time_steps = truth_time_steps.unsqueeze(1).repeat(1,X.shape[1],1).unsqueeze(-1)
        else:
            truth_time_steps = truth_time_steps.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]
        mask = mask.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]

        # Check zero-data values
        if (mask.sum(dim=(1, 2, 3)) == 0).any():
            print('Hey!')

        return out_dict, B, L_in, N, X, truth_time_steps, mask

    def forward(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        # Preprocess
        out_dict, B, L_in, N, X, truth_time_steps, mask = self.preprocess_batch(X, truth_time_steps, mask)

        # Encoder
        X = self.obs_enc(X)  # [B, N, T_in, 1] -> observation encoder -> [B, N, T_in, D]
        te_his = self.LearnableTE(truth_time_steps)  # [B, N, T_in, 1] -> time-step encoder -> [B, N, T_in, D]
        var_emb = self.nodevec.weight.view(1, N, 1, self.args.hid_dim).repeat(B, 1, L_in, 1)  # [B, N, T_in, 1] -> variable encoder -> [B, N, T_in, D]
        X = self.relu(X + var_emb + te_his)  # node-graph embedding  -> [B, N, T_in, D]

        # GNN
        graph = build_graph_from_mask_v2(X,
                                         mask.squeeze(),
                                         inter_mode=self.args.inner_mode)  # Data object: graph.x:[(mask == 1.).sum(), D], graph.edge_index:[2,E]
        graph.x = self.gnn(x=graph.x,
                           edge_index=graph.edge_index,
                           batch=graph.batch)

        h = self.pool(graph.x , graph.batch)  # [(mask == 1.).sum(), D] -> pool -> [B*N, D]
        h = torch.reshape(h, (B,N,-1))  # [B, N, D]

        # Decoder: adapt to time_steps_to_predict
        L_pred = time_steps_to_predict.shape[-1]  # 40
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # # [B, N, L_out, D]
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # [B, L_out] -> [B, N, L_out, 1]

        # Decoder: embeddings
        te_pred = self.LearnableTE(time_steps_to_predict)  # [B, N, L_out, 1] -> [B, N, L_out, D]

        # Decoder: forward
        outputs = self.decoder(h, te_pred).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)  # [B, N, L_out, 2D] -> decoder -> [1, B, L_out, N]


        out_dict['pred_y'] = outputs
        return out_dict


# Graph-based Regression with Adaptive Pooling and Embeddings
class GrapeDgm(nn.Module):
    def __init__(self, args: Parameters):
        super(GrapeDgm, self).__init__()
        self.args = args
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, args.hid_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(args.hid_dim * 2, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )
        self.dgm = DGMmodule(args)

        self.gcn = select_gnn(args)

        self.pool = QueryPooling(args.ndim,
                                 args.hid_dim,
                                 num_heads=args.pool_num_heads)

    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def preprocess_batch(self, X, truth_time_steps, mask):
        out_dict = dict()

        # Preprocess (patch input to normal input)
        if len(X.shape) == 4:
            X = X.squeeze()
            mask = mask.squeeze()
            truth_time_steps = truth_time_steps.squeeze()

        B, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 2, 1).unsqueeze(-1)  # permuto e aggiungo dimensione per espandere h_dim, [B, N, T_in, 1]
        # Data arranged in compact or non-compact way
        if not self.args.patch_ts:
            truth_time_steps = truth_time_steps.unsqueeze(1).repeat(1, X.shape[1], 1).unsqueeze(-1)
        else:
            truth_time_steps = truth_time_steps.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]
        mask = mask.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]

        # Check zero-data values
        if (mask.sum(dim=(1, 2, 3)) == 0).any():
            print('Hey!')

        return out_dict, B, L_in, N, X, truth_time_steps, mask

    def forward(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        # Preprocess
        out_dict, B, L_in, N, X, truth_time_steps, mask = self.preprocess_batch(X, truth_time_steps, mask)

        # Encoder
        X = self.obs_enc(X)  # [B, N, T_in, 1] -> observation encoder -> [B, N, T_in, D]
        te_his = self.LearnableTE(truth_time_steps)  # [B, N, T_in, 1] -> time-step encoder -> [B, N, T_in, D]
        var_emb = self.nodevec.weight.view(1, N, 1, self.args.hid_dim).repeat(B, 1, L_in,
                                                                              1)  # [B, N, T_in, 1] -> variable encoder -> [B, N, T_in, D]
        X = self.relu(X + var_emb + te_his)  # node-graph embedding  -> [B, N, T_in, D]

        # GNN
        graph = build_graph_from_mask(X,
                                      mask.squeeze())  # Data object: graph.x:[(mask == 1.).sum(), D], graph.edge_index:[2,E]
        graph.x = self.gcn(x=graph.x,
                           edge_index=graph.edge_index,
                           batch=graph.batch)

        h = self.pool(graph.x, graph.batch)  # [(mask == 1.).sum(), D] -> pool -> [B*N, D]
        h = torch.reshape(h, (B, N, -1))  # [B, N, D]

        # DGM
        h, l_probs = self.dgm(h)  #TODO: usare modo migliore per compressare T dim (B,N,T,F).mean(2) = 32, 41, 32
        out_dict['l_probs'] = l_probs

        # Decoder
        L_pred = time_steps_to_predict.shape[-1]  # 40
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # # [B, N, L_out, D]
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1,
                                                                                   1)  # [B, L_out] -> [B, N, L_out, 1]
        te_pred = self.LearnableTE(time_steps_to_predict)  # [B, N, L_out, 1] -> [B, N, L_out, D]
        h = torch.cat([h, te_pred], dim=-1)  # [B, N, L_out, D] -> [B, N, L_out, 2D]
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(
            dim=0)  # [B, N, L_out, 2D] -> decoder -> [1, B, L_out, N]
        out_dict['pred_y'] = outputs
        return out_dict
