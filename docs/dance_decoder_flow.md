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
