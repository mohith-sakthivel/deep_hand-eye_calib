Data setup instructions:

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip


Things to try:
- [ ] Use pre-trained DeepSFM feature extractor
- [ ] Apply self-attention instead of attention
- [ ] Weighting between hand-eye and relative pose losses
- [ ] Try more sophisticated message passing
    - [ ] Add target node to msg generation
    - [ ] Apply attention over messages before aggregation
    - [ ] Add message as a residual to the node

Experiments to try:
- [ ] Add noise to robot end-effector pose
