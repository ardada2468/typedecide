# Getting started

Three pages, in order:

1. [Install](install.md). The base install is small; training and export are extras.
2. [Your first dataset](first-dataset.md). What a `Decision` is and what a row looks
   like on disk.
3. [End to end](end-to-end.md). Validate, split, train, evaluate and export, with the
   real output of each command.

You can do the first two pages, and the validate and split steps of the third, on a
laptop with no GPU and no model download. Training and evaluation download model
weights from the Hugging Face Hub and are far more pleasant with a CUDA GPU.

!!! note "Python version"
    `typedecide` requires Python 3.10 or newer.
