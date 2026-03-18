import torch

# Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x, x) ** 2
    return dist, x


# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x ** 2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    x = x / (x_norm * (1 + 1e-2))
    x_norm = (x ** 2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
    return dist, x


def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


import torch
from torch_geometric.data import Data, Batch


def build_graph_from_mask(X: torch.Tensor, Mask: torch.Tensor):
    """
    X    : (B, F, T, H)
    Mask : (B, F, T)  — valori in {0,1}

    Returns:
        Data con:
          - x          : (N_valid, H)
          - edge_index : (2, E)
          - batch      : (N_valid,)  — indice del grafo di appartenenza
    """
    B, F, T, H = X.shape
    device = X.device

    # 1. Mappa da indice raw (b*F*T + f*T + t) → indice compatto          #
    # flat mask: (B*F*T,)
    flat_mask = Mask.reshape(-1).bool()  # (B*F*T,)

    # indice compatto per ogni posizione raw (-1 se non valida)
    compact_idx = torch.full((B * F * T,), -1, dtype=torch.long, device=device)
    valid_raw = flat_mask.nonzero(as_tuple=False).squeeze(1)  # posizioni valide
    compact_idx[valid_raw] = torch.arange(valid_raw.shape[0], device=device)

    # 2. Nodi: features e batch vector                                    #
    # reshape X in (B*F*T, H) e filtra
    X_flat = X.reshape(B * F * T, H)
    x_valid = X_flat[flat_mask]  # (N_valid, H)

    # batch vector: a quale b appartiene ogni nodo valido?
    b_idx = torch.arange(B, device=device).repeat_interleave(F * T)  # (B*F*T,)
    batch_vec = b_idx[flat_mask]  # (N_valid,)

    # 3. Edge temporali intra-canale: (b,f,t-1) → (b,f,t)               #
    # Per ogni (b,f,t) con t>0, controlla che entrambi siano validi
    # raw index di t-1: b*F*T + f*T + (t-1)

    # indici raw di tutti i possibili "t" (t >= 1)
    b_all = torch.arange(B, device=device).repeat_interleave(F * (T - 1))
    f_all = torch.arange(F, device=device).repeat(B).repeat_interleave(T - 1)
    t_all = torch.arange(1, T, device=device).repeat(B * F)

    raw_t = b_all * (F * T) + f_all * T + t_all  # idx di t
    raw_tm1 = b_all * (F * T) + f_all * T + (t_all - 1)  # idx di t-1

    valid_temporal = flat_mask[raw_t] & flat_mask[raw_tm1]  # entrambi devono essere nodi validi


    src_temp = compact_idx[raw_tm1[valid_temporal]]
    dst_temp = compact_idx[raw_t[valid_temporal]]

    # 4. Edge inter-canale: ultimo nodo di (b,f) → primo nodo di (b,f+1) #
    edges_cross_src = []
    edges_cross_dst = []

    for b in range(B):
        for f in range(F - 1):
            # ultimo t valido del canale f
            raw_f = b * F * T + f * T
            raw_f1 = b * F * T + (f + 1) * T

            mask_f = flat_mask[raw_f: raw_f + T]
            mask_f1 = flat_mask[raw_f1: raw_f1 + T]

            t_last = mask_f.nonzero(as_tuple=False)
            t_first = mask_f1.nonzero(as_tuple=False)

            if t_last.numel() == 0 or t_first.numel() == 0:
                continue  # uno dei due canali è vuoto

            t_last_idx = t_last[-1].item()
            t_first_idx = t_first[0].item()

            src_raw = raw_f + t_last_idx
            dst_raw = raw_f1 + t_first_idx

            edges_cross_src.append(compact_idx[src_raw].item())
            edges_cross_dst.append(compact_idx[dst_raw].item())

    # 5. Assembla edge_index                                              #
    if edges_cross_src:
        cross_src = torch.tensor(edges_cross_src, dtype=torch.long, device=device)
        cross_dst = torch.tensor(edges_cross_dst, dtype=torch.long, device=device)

        all_src = torch.cat([src_temp, cross_src])
        all_dst = torch.cat([dst_temp, cross_dst])
    else:
        all_src = src_temp
        all_dst = dst_temp

    edge_index = torch.stack([all_src, all_dst], dim=0)  # (2, E)

    return Data(x=x_valid, edge_index=edge_index, batch=batch_vec)