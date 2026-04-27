import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Parameters

class SimpleDecoder(nn.Module):
    def __init__(self, args: Parameters):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.hid_dim * 2, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1))
    def forward(self, h, te_pred):
        h = torch.cat([h, te_pred], dim=-1)  # [B, N, L_out, D] -> [B, N, L_out, 2D]
        return self.decoder(h)


def select_decoder(args: Parameters):
    if args.decoder_name == 'simple':
        return SimpleDecoder(args)
    if args.decoder_name == 'film':
        return FiLMDecoder(args)
    if args.decoder_name == 'gated':
        return GatedDecoder(args)
    if args.decoder_name == 'filmSwiglu':
        return FiLMSwiGLUDecoder(args)
    if args.decoder_name == 'gru':
        return GRURecurrentDecoder(args)
    if args.decoder_name == 'crossAttn':
        return CrossAttentionDecoder(args)
    if args.decoder_name == 'INR':
        return INRDecoder(args)
    else:
        raise ValueError(f'Invalid decoder name: {args.decoder_name}')


class FiLMDecoder(nn.Module):
    def __init__(self, args: Parameters):
        super().__init__()
        D = args.hid_dim
        self.film = nn.Linear(D, 2 * D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, 1),
        )
        # init: gamma=1, beta=0 -> all'inizio x = h
        with torch.no_grad():
            self.film.weight.zero_()
            self.film.bias.zero_()
            self.film.bias[:D].fill_(1.0)

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        # h, te_pred: [B, N, L_out, D]
        gamma, beta = self.film(te_pred).chunk(2, dim=-1)
        x = gamma * h + beta
        return self.mlp(x).squeeze(-1)  # [B, N, L_out]


class GatedDecoder(nn.Module):
    def __init__(self, args: Parameters, expansion: int = 2):
        super().__init__()
        D = args.hid_dim
        H = expansion * D
        self.proj_gate = nn.Linear(2 * D, H)
        self.proj_up   = nn.Linear(2 * D, H)
        self.proj_down = nn.Linear(H, D)
        self.head      = nn.Linear(D, 1)

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        # h, te_pred: [B, N, L_out, D]
        x = torch.cat([h, te_pred], dim=-1)                          # [B, N, L_out, 2D]
        x = self.proj_down(F.silu(self.proj_gate(x)) * self.proj_up(x))
        return self.head(x).squeeze(-1)                              # [B, N, L_out]


class FiLMSwiGLUDecoder(nn.Module):
    """
    Drop-in replacement per il tuo MLP a 3 layer.

    Idea:
      - te_pred modula h tramite FiLM (gamma, beta), invece di essere concatenato
      - il blocco interno usa SwiGLU al posto di Linear+ReLU
      - residual + LayerNorm per stabilità

    Input shapes (come nel tuo forward):
      h:       [B, N, L_out, D]  (gia' replicato lungo L_out)
      te_pred: [B, N, L_out, D]
    Output:
      y_hat:   [B, N, L_out]
    """

    def __init__(self, args: Parameters, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        D = args.hid_dim
        H = expansion * D  # dimensione interna SwiGLU

        # FiLM: dal time embedding generiamo (gamma, beta) per modulare h
        self.film = nn.Linear(D, 2 * D)

        # Pre-norm sull'input modulato
        self.norm = nn.LayerNorm(D)

        # SwiGLU block: due proiezioni in parallelo, una con SiLU che fa da gate
        self.proj_gate = nn.Linear(D, H)
        self.proj_up = nn.Linear(D, H)
        self.proj_down = nn.Linear(H, D)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Head di regressione finale
        self.head = nn.Linear(D, 1)

        # Init: gamma vicino a 1, beta a 0 -> all'inizio FiLM e' quasi identita'
        nn.init.zeros_(self.film.weight)
        with torch.no_grad():
            self.film.bias.zero_()
            self.film.bias[:D].fill_(1.0)  # gamma_init = 1

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        # h, te_pred: [B, N, L_out, D]

        # --- FiLM modulation: il tempo modula il contesto ---
        gamma, beta = self.film(te_pred).chunk(2, dim=-1)  # ognuno [B, N, L_out, D]
        x = gamma * h + beta  # [B, N, L_out, D]

        # --- SwiGLU block con residual ---
        residual = x
        x = self.norm(x)
        x = self.proj_down(F.silu(self.proj_gate(x)) * self.proj_up(x))
        x = self.dropout(x) + residual

        # --- Head ---
        y = self.head(x).squeeze(-1)  # [B, N, L_out]
        return y


class GRURecurrentDecoder(nn.Module):
    """
    Decoder ricorrente: stato iniziale = h (contesto della serie),
    input al GRU = embedding del timestamp futuro.

    Input shapes:
      h:       [B, N, L_out, D]  (verra' usato solo lo slice [..., 0, :])
      te_pred: [B, N, L_out, D]
    Output:
      y_hat:   [B, N, L_out]
    """

    def __init__(self, args: Parameters, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers
        self.hid_dim = args.hid_dim

        self.gru = nn.GRU(
            input_size=args.hid_dim,
            hidden_size=args.hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Se n_layers > 1, replichiamo h su tutti i layer
        self.head = nn.Linear(args.hid_dim, 1)

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        B, N, L_out, D = te_pred.shape

        # Appiattisci batch e variabili: il GRU vede (B*N) "serie" indipendenti
        te_seq = te_pred.reshape(B * N, L_out, D)  # [B*N, L_out, D]

        # Stato iniziale: prendiamo h al primo step (e' costante lungo L_out)
        h0 = h[:, :, 0, :].reshape(1, B * N, D).contiguous()  # [1, B*N, D]
        if self.n_layers > 1:
            h0 = h0.expand(self.n_layers, -1, -1).contiguous()  # [n_layers, B*N, D]

        out, _ = self.gru(te_seq, h0)  # [B*N, L_out, D]
        y = self.head(out).squeeze(-1)  # [B*N, L_out]
        return y.reshape(B, N, L_out)


class CrossAttentionDecoder(nn.Module):
    """
    Ogni timestamp futuro interroga il contesto della propria variabile via attention.

    Per default usa 1 token di contesto per variabile (h).
    Se vuoi piu' espressivita', puoi passare un h con piu' "memory tokens"
    (vedi nota sotto): basta che la shape sia [B, N, K, D] con K >= 1.

    Input shapes:
      h:       [B, N, L_out, D]  oppure  [B, N, K, D]   (vedi sotto)
      te_pred: [B, N, L_out, D]
    Output:
      y_hat:   [B, N, L_out]
    """

    def __init__(self, args: Parameters, n_heads: int = 4, dropout: float = 0.0,
                 ffn_expansion: int = 2):
        super().__init__()
        D = args.hid_dim
        H = ffn_expansion * D

        self.norm_q = nn.LayerNorm(D)
        self.norm_kv = nn.LayerNorm(D)

        self.attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # Piccola FFN post-attention con SwiGLU (stile transformer moderno)
        self.norm_ffn = nn.LayerNorm(D)
        self.proj_gate = nn.Linear(D, H)
        self.proj_up = nn.Linear(D, H)
        self.proj_down = nn.Linear(H, D)

        self.head = nn.Linear(D, 1)

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        B, N, L_out, D = te_pred.shape

        # --- Costruzione del "contesto" ---
        # Se h e' [B, N, L_out, D] (replicato come fai ora), basta lo slice al primo
        # step per ottenere un singolo token per variabile.
        # Se invece passi K memory tokens, h e' [B, N, K, D] e li usiamo tutti.
        if h.dim() == 4 and h.shape[2] == L_out:
            kv = h[:, :, :1, :]  # [B, N, 1, D]
        else:
            kv = h  # [B, N, K, D]

        # Appiattiamo (B, N) -> batch del MultiheadAttention
        BN = B * N
        K = kv.shape[2]

        q = te_pred.reshape(BN, L_out, D)  # queries  : un token per timestamp futuro
        kv = kv.reshape(BN, K, D)  # keys/vals: K token di contesto

        # Pre-norm + cross-attention + residual
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        x = q + attn_out  # residual sulla query

        # FFN SwiGLU + residual
        x_n = self.norm_ffn(x)
        x = x + self.proj_down(F.silu(self.proj_gate(x_n)) * self.proj_up(x_n))

        # Head
        y = self.head(x).squeeze(-1)  # [BN, L_out]
        return y.reshape(B, N, L_out)


class INRDecoder(nn.Module):
    """
    Implicit Neural Representation: il decoder e' una funzione continua del tempo
    f(t), parametrizzata punto per punto da h tramite FiLM stratificato.

    Concettualmente: per ogni serie (b, n), impari una piccola rete sinusoidale
    (SIREN-like) che mappa il time embedding al valore predetto, modulata dal
    contesto h. E' molto coerente con l'idea "raw, continuous, no patching".

    Input shapes:
      h:       [B, N, L_out, D]
      te_pred: [B, N, L_out, D]
    Output:
      y_hat:   [B, N, L_out]
    """

    def __init__(self, args: Parameters, n_layers: int = 3, w0: float = 30.0,
                 first_layer: bool = True):
        super().__init__()
        D = args.hid_dim
        self.n_layers = n_layers
        self.w0 = w0
        self.first_layer = first_layer

        # Stack di layer "modulati": ogni layer applica  sin(w0 * (gamma * Wx + beta))
        self.layers = nn.ModuleList([nn.Linear(D, D) for _ in range(n_layers)])
        self.films = nn.ModuleList([nn.Linear(D, 2 * D) for _ in range(n_layers)])

        self.head = nn.Linear(D, 1)

        # Init SIREN: importante per la stabilita' delle attivazioni sinusoidali
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                fan_in = layer.weight.shape[1]
                if i == 0 and first_layer:
                    bound = 1.0 / fan_in
                else:
                    bound = (6.0 / fan_in) ** 0.5 / w0
                layer.weight.uniform_(-bound, bound)
                layer.bias.zero_()

            # FiLM init: gamma ~ 1, beta ~ 0 (parti dall'identita' moltiplicativa)
            for film in self.films:
                film.weight.zero_()
                film.bias.zero_()
                film.bias[:args.hid_dim].fill_(1.0)

    def forward(self, h: torch.Tensor, te_pred: torch.Tensor) -> torch.Tensor:
        # h, te_pred: [B, N, L_out, D]

        x = te_pred  # la "coordinata" e' il time embedding
        for i, (layer, film) in enumerate(zip(self.layers, self.films)):
            gamma, beta = film(h).chunk(2, dim=-1)  # modulazione cond. su h
            x = layer(x)
            x = torch.sin(self.w0 * (gamma * x + beta))  # attivazione sinusoidale

        return self.head(x).squeeze(-1)  # [B, N, L_out]


# # ... tutto identico fino a:
# h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # [B, N, L_out, D]
# time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
# te_pred = self.LearnableTE(time_steps_to_predict)  # [B, N, L_out, D]
#
# # ---- nuovo decoder, niente concat ----
# outputs = self.decoder(h, te_pred)                  # [B, N, L_out]
# outputs = outputs.permute(0, 2, 1).unsqueeze(dim=0) # [1, B, L_out, N]
# out_dict['pred_y'] = outputs
# return out_dict


# class FiLMDecoder(nn.Module):
#     def __init__(self, hid_dim):
#         super().__init__()
#         # genera (gamma, beta) dal time embedding
#         self.film = nn.Linear(hid_dim, 2 * hid_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(hid_dim, hid_dim),
#             nn.GELU(),
#             nn.Linear(hid_dim, 1),
#         )
#
#     def forward(self, h, te_pred):
#         # h: [B, N, L_out, D], te_pred: [B, N, L_out, D]
#         gamma, beta = self.film(te_pred).chunk(2, dim=-1)
#         h = gamma * h + beta            # modulazione condizionata sul tempo
#         return self.mlp(h).squeeze(-1)
#
#
# class GatedDecoder(nn.Module):
#     def __init__(self, hid_dim):
#         super().__init__()
#         self.proj_in = nn.Linear(2 * hid_dim, hid_dim)
#         self.gate = nn.Linear(2 * hid_dim, hid_dim)
#         self.proj_out = nn.Linear(hid_dim, 1)
#
#     def forward(self, h_cat):  # h_cat: [..., 2D]
#         return self.proj_out(F.silu(self.gate(h_cat)) * self.proj_in(h_cat)).squeeze(-1)
#
#
# class CrossAttnDecoder(nn.Module):
#     def __init__(self, hid_dim, n_heads=4):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(hid_dim, n_heads, batch_first=True)
#         self.norm = nn.LayerNorm(hid_dim)
#         self.head = nn.Linear(hid_dim, 1)
#
#     def forward(self, h, te_pred):
#         # h: [B, N, D] (NON replicato), te_pred: [B, N, L_out, D]
#         B, N, L, D = te_pred.shape
#         q = te_pred.reshape(B * N, L, D)
#         kv = h.reshape(B * N, 1, D)  # un singolo "token di contesto" per variabile
#         out, _ = self.attn(q, kv, kv)
#         return self.head(self.norm(out + q)).squeeze(-1).reshape(B, N, L)
#
#
# class INRDecoder(nn.Module):
#     def __init__(self, hid_dim, n_layers=2):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.Linear(hid_dim, hid_dim) for _ in range(n_layers)
#         ])
#         self.films = nn.ModuleList([
#             nn.Linear(hid_dim, 2 * hid_dim) for _ in range(n_layers)
#         ])
#         self.head = nn.Linear(hid_dim, 1)
#
#     def forward(self, h, te_pred):
#         # h modula te_pred a ogni layer
#         x = te_pred
#         for layer, film in zip(self.layers, self.films):
#             gamma, beta = film(h).chunk(2, dim=-1)
#             x = torch.sin(gamma * layer(x) + beta)  # SIREN-style
#         return self.head(x).squeeze(-1)
