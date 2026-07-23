"""ADJSCC: attention-based deep joint source-channel coding for CIFAR-10.

Reproduction of arXiv:2012.00533. Library modules:
  channel  — AWGN + power normalization
  models   — AFModule, Encoder, Decoder, DeepJSCC (attention flag)
  data     — CIFAR-10 loaders
  metrics  — PSNR, SSIM
  engine   — device pick, train loop, eval sweep, checkpoint I/O
"""
