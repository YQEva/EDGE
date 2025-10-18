# EDGE DanceDecoder Data Flow

```mermaid
graph TD
    X[Noised Motion Sequence x] -->|Linear| XP[Input Projection]
    XP -->|+ Positional Encoding| XM[Motion Tokens]
    XM -->|Self-Attn + FiLM| D1[Decoder Layer 1]
    D1 -->|Stacked Residual Layers| Dn[Decoder Layer N]
    Dn -->|Linear| OUT[Predicted Motion Update]

    subgraph Conditioning Branch
        CE[Audio Conditioning Embeddings] -->|Linear| CP[Cond Projection]
        CP -->|Abs Pos Encoding| CA[Positional Encoding]
        CA -->|Transformer Encoder| CT[Contextual Tokens]
        CT -->|Classifier-Free Dropout| CTD[Condition Tokens]
    end

    subgraph Time Branch
        T[Diffusion Timestep t] -->|Sinusoidal Pos Emb| TS[Time Sinusoid]
        TS -->|Linear + Mish| TH[Hidden Time Emb]
        TH -->|Linear| TCraw[Time FiLM Vector]
        TH -->|Linear| TT[Time Tokens]
    end

    CTD -->|Concat + LayerNorm| MEM[Decoder Memory]
    TT --> MEM
    MEM -->|Cross-Attn + FiLM| D1

    CTD -->|Mean Pool + MLP| CH[Cond Hidden Vector]
    CH -->|Add| SUM[Combined FiLM Conditioning]
    TCraw -->|Add| SUM
    SUM -->|DenseFiLM blocks| D1
    SUM --> Dn

    XM -->|Cross-Attn + FiLM| Dn
    MEM -->|Cross-Attn + FiLM| Dn
```

This diagram summarizes how motion, music, and diffusion timestep signals flow through the `DanceDecoder` defined in [`model/model.py`](../model/model.py). Motion tokens repeatedly attend to the concatenated conditioning/time memory while FiLM modulation injects the combined conditioning vector into every sub-layer.

## TransformerEncoderLayer Flow

```mermaid
flowchart LR
    subgraph PreNorm["norm_first = True"]
        PN_in[Input src]
        PN_in --> PN_LN1[LayerNorm]
        PN_LN1 --> PN_SA[Self-Attention Block]
        PN_SA --> PN_Add1[Residual Add]
        PN_Add1 --> PN_LN2[LayerNorm]
        PN_LN2 --> PN_FF[Feedforward Block]
        PN_FF --> PN_Add2[Residual Add]
        PN_Add2 --> OUT_pre[Output]
    end

    subgraph PostNorm["norm_first = False"]
        PO_in[Input src]
        PO_in --> PO_SA[Self-Attention Block]
        PO_SA --> PO_Add1[Residual Add]
        PO_Add1 --> PO_LN1[LayerNorm]
        PO_LN1 --> PO_FF[Feedforward Block]
        PO_FF --> PO_Add2[Residual Add]
        PO_Add2 --> PO_LN2[LayerNorm]
        PO_LN2 --> OUT_post[Output]
    end

    subgraph SABlock[Self-Attention Block]
        SA_in[Input] --> ROT{use_rotary?}
        ROT -->|Yes| ROT_qk[Rotate q/k with rotary embedding]
        ROT -->|No| ID_qk[Identity]
        ROT_qk --> MHA[MHA(q,k,v=x)]
        ID_qk --> MHA
        MHA --> DO1[Dropout]
        DO1 --> SA_out[Return]
    end

    subgraph FFBlock[Feedforward Block]
        FF_in[Input] --> FC1[Linear → dim_feedforward]
        FC1 --> ACT[Activation]
        ACT --> DO2[Dropout]
        DO2 --> FC2[Linear → d_model]
        FC2 --> DO3[Dropout]
        DO3 --> FF_out[Return]
    end

    PN_SA -.-> SA_in
    PO_SA -.-> SA_in
    PN_FF -.-> FF_in
    PO_FF -.-> FF_in
```

The encoder layer first decides whether to apply pre-norm (`norm_first=True`) or post-norm (`norm_first=False`) residual ordering. In both cases, the self-attention block optionally rotates queries/keys with the provided rotary embedding before calling PyTorch's multi-head attention and dropping out the result, while the feedforward block applies a linear–activation–dropout sandwich followed by a projection back to the model width and another dropout, matching the implementation in [`TransformerEncoderLayer`](../model/model.py).
