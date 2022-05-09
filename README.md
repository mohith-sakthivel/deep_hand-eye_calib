Data setup instructions:

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip


Things to try:
- [ ] Apply self-attention instead of attention
- [ ] Weighting between hand-eye and relative pose losses
- [ ] Use convolutional features
- [ ] DeepSFM feature extractor uses no maxpool
- [ ] Add target node to msg_mlp
- [ ] Apply attention over messages before aggregation
- [ ] Add noise to robot end-effector pose

Things to note:
- Is non-linearity needed after edge_he_feat
- Original implementation had edge direction invariance in `join_node_edge_feat`