import torch
from torch_geometric.nn import GCN, GAT
from torch_geometric.data import Data

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



def build_graph_from_mask(X: torch.Tensor,
                          Mask: torch.Tensor):
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

    # 1. Mappa da indice raw (b*F*T + f*T + t) → indice compatto
    flat_mask = Mask.reshape(-1).bool()  # (B*F*T,)

    # indice compatto per ogni posizione raw (-1 se no valida)
    compact_idx = torch.full((B * F * T,), -1, dtype=torch.long, device=device)
    valid_raw = flat_mask.nonzero(as_tuple=False).squeeze(1)  # posizioni valide
    compact_idx[valid_raw] = torch.arange(valid_raw.shape[0], device=device)

    # 2. Nodi: features e batch vector; reshape X in (B*F*T, H) e filtra
    X_flat = X.reshape(B * F * T, H)
    x_valid = X_flat[flat_mask]  # (N_valid, H)

    # batch vector: a quale b appartiene ogni nodo valido?
    b_idx = torch.arange(B, device=device).repeat_interleave(F * T)  # (B*F*T,)
    batch_vec = b_idx[flat_mask]  # (N_valid,)

    # 3. Edge temporali intra-canale: (b,f,t-1) → (b,f,t)
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

    # 4. Edge inter-canale: ultimo nodo di (b,f) → primo nodo di (b,f+1)
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

    # 5. Assembla edge_index
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


def build_graph_from_mask_v2(
    X: torch.Tensor,
    Mask: torch.Tensor,
    inter_mode: int = 1,
    random_seed: int = None
):
    """
    X    : (B, F, T, H)
    Mask : (B, F, T)  — valori in {0,1}

    inter_mode:
        1 — inter-canale: ultimo(f) → primo(f+1)          [default]
        2 — inter-canale: primo(f)  → primo(f+1)
        3 — inter-canale: ultimo(f) → primo(f+1)
                        + primo(f)  → primo(f+1)
        4 — inter-canale: archi random tra nodi di canali diversi,
                          con garanzia di grafo connesso (no nodi isolati)

    random_seed: seed per riproducibilità (usato solo in inter_mode=4)

    Returns:
        Data con:
          - x          : (N_valid, H)
          - edge_index : (2, E)
          - batch      : (N_valid,)
    """
    assert inter_mode in (1, 2, 3, 4), "inter_mode deve essere 1, 2, 3 o 4"

    B, F, T, H = X.shape
    device = X.device

    # 1. Mappa raw → compatto
    flat_mask = Mask.reshape(-1).bool()
    compact_idx = torch.full((B * F * T,), -1, dtype=torch.long, device=device)
    valid_raw = flat_mask.nonzero(as_tuple=False).squeeze(1)
    compact_idx[valid_raw] = torch.arange(valid_raw.shape[0], device=device)

    # 2. Nodi
    X_flat = X.reshape(B * F * T, H)
    x_valid = X_flat[flat_mask]
    b_idx = torch.arange(B, device=device).repeat_interleave(F * T)
    batch_vec = b_idx[flat_mask]

    # 3. Edge intra-canale: (b,f,t-1) → (b,f,t)
    b_all  = torch.arange(B, device=device).repeat_interleave(F * (T - 1))
    f_all  = torch.arange(F, device=device).repeat(B).repeat_interleave(T - 1)
    t_all  = torch.arange(1, T, device=device).repeat(B * F)

    raw_t   = b_all * (F * T) + f_all * T + t_all
    raw_tm1 = b_all * (F * T) + f_all * T + (t_all - 1)

    valid_temporal = flat_mask[raw_t] & flat_mask[raw_tm1]
    src_temp = compact_idx[raw_tm1[valid_temporal]]
    dst_temp = compact_idx[raw_t[valid_temporal]]

    # 4. Edge inter-canale
    edges_cross_src = []
    edges_cross_dst = []

    if inter_mode == 4:
        rng = torch.Generator(device='cpu')
        if random_seed is not None:
            rng.manual_seed(random_seed)

        for b in range(B):
            # Raccogli i nodi validi per ogni canale nel campione b
            # channel_nodes[f] = lista di compact_idx validi per (b, f)
            channel_nodes = []
            for f in range(F):
                raw_f = b * F * T + f * T
                mask_f = flat_mask[raw_f: raw_f + T]
                t_valid = mask_f.nonzero(as_tuple=False).squeeze(1)
                if t_valid.numel() > 0:
                    nodes = compact_idx[raw_f + t_valid].tolist()
                else:
                    nodes = []
                channel_nodes.append(nodes)

            # Filtra canali non vuoti
            nonempty = [f for f in range(F) if len(channel_nodes[f]) > 0]
            if len(nonempty) < 2:
                continue

            # ── Fase A: spanning tree tra canali per garantire connettività ──
            # Collega i canali in ordine casuale con un arco random ciascuno,
            # così ogni canale è raggiungibile dagli altri
            perm = torch.randperm(len(nonempty), generator=rng).tolist()
            shuffled = [nonempty[i] for i in perm]

            for i in range(len(shuffled) - 1):
                f_a = shuffled[i]
                f_b = shuffled[i + 1]

                # scegli un nodo random da ciascun canale
                na = channel_nodes[f_a]
                nb = channel_nodes[f_b]

                pick_a = na[torch.randint(len(na), (1,), generator=rng).item()]
                pick_b = nb[torch.randint(len(nb), (1,), generator=rng).item()]

                edges_cross_src.append(pick_a)
                edges_cross_dst.append(pick_b)

            # ── Fase B: archi extra random (opzionale, aumenta densità) ──
            # Numero di archi extra ~ F/2, tra coppie di canali distinti
            n_extra = max(0, len(nonempty) // 2)
            all_pairs = [
                (nonempty[i], nonempty[j])
                for i in range(len(nonempty))
                for j in range(i + 1, len(nonempty))
            ]
            if n_extra > 0 and len(all_pairs) > 0:
                chosen = torch.randperm(len(all_pairs), generator=rng)[:n_extra].tolist()
                for idx in chosen:
                    f_a, f_b = all_pairs[idx]
                    na = channel_nodes[f_a]
                    nb = channel_nodes[f_b]
                    pick_a = na[torch.randint(len(na), (1,), generator=rng).item()]
                    pick_b = nb[torch.randint(len(nb), (1,), generator=rng).item()]
                    edges_cross_src.append(pick_a)
                    edges_cross_dst.append(pick_b)

    else:
        # Modalità 1, 2, 3
        for b in range(B):
            for f in range(F - 1):
                raw_f  = b * F * T + f * T
                raw_f1 = b * F * T + (f + 1) * T

                mask_f  = flat_mask[raw_f  : raw_f  + T]
                mask_f1 = flat_mask[raw_f1 : raw_f1 + T]

                t_valid_f  = mask_f.nonzero(as_tuple=False)
                t_valid_f1 = mask_f1.nonzero(as_tuple=False)

                if t_valid_f.numel() == 0 or t_valid_f1.numel() == 0:
                    continue

                t_first_f  = t_valid_f[0].item()
                t_last_f   = t_valid_f[-1].item()
                t_first_f1 = t_valid_f1[0].item()
                t_last_f1  = t_valid_f1[-1].item()

                if inter_mode in (1, 3):
                    src_raw = raw_f  + t_last_f
                    dst_raw = raw_f1 + t_first_f1
                    edges_cross_src.append(compact_idx[src_raw].item())
                    edges_cross_dst.append(compact_idx[dst_raw].item())

                if inter_mode in (2, 3):
                    src_raw = raw_f  + t_first_f
                    dst_raw = raw_f1 + t_first_f1
                    edges_cross_src.append(compact_idx[src_raw].item())
                    edges_cross_dst.append(compact_idx[dst_raw].item())

    # 5. Assembla edge_index
    if edges_cross_src:
        cross_src = torch.tensor(edges_cross_src, dtype=torch.long, device=device)
        cross_dst = torch.tensor(edges_cross_dst, dtype=torch.long, device=device)
        all_src = torch.cat([src_temp, cross_src])
        all_dst = torch.cat([dst_temp, cross_dst])
    else:
        all_src = src_temp
        all_dst = dst_temp

    edge_index = torch.stack([all_src, all_dst], dim=0)
    return Data(x=x_valid, edge_index=edge_index, batch=batch_vec)

def select_gnn(args):
    if args.gnn_name == 'GCN':
        gnn = GCN(in_channels=args.hid_dim,
                  out_channels=args.hid_dim,
                  hidden_channels=args.hid_dim,
                  num_layers=args.gnn_layers,
                  dropout=args.dropout,)
    elif args.gnn_name == 'GAT':
        gnn = GAT(in_channels=args.hid_dim,
                  out_channels=args.hid_dim,
                  hidden_channels=args.hid_dim,
                  num_layers=args.gnn_layers,
                  v2=True,
                  dropout=args.dropout)
    else:
        raise ValueError('Unknown gnn {}'.format(args.gnn_name))
    return gnn