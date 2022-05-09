To Download dataset

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip


Things to try:
    - Apply self-attention instead of attention
    - Weighting between hand-eye and relative pose losses
    - Use convolutional features
    - DeepSFM feature extractor uses no maxpool
    - Add current node to msg_mlp
    - Add non-linearity after edge_he_feat
    - SimpleConvEdgeUpdate to SimpleEdgeUpdate -> No conv