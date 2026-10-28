### Setup
If using a Colab runtime, you can ignore `conda` and perform all of the `pip`-based installations.
```
conda create -n py312FinTechA3 python=3.12 -y   # Python will need to be at least 3.11
conda activate py312FinTechA3
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers==5.1.2
pip install 'datasets<3.0'
pip install comet_ml==3.58.8
pip install einops==0.8.1
pip install accelerate==1.15.0
pip install psutil==7.2.2
```

### Files
- `ImplementMe.py`: this is the main file you need to modify, and will be graded in part I.
- `NoModify.py`: do not change anything in this file.
- `TrainGRPO.py`: does GRPO training.
- `Utils.py`: contains utilities used in other files.
- `UtilsBase.py`: contains utilities used in other files.

### A Note on Indexing
In this assignment, there are tensors of many dimensions. Intuitively, a one-dimensional tensor is a vector, a two-dimensional tensor is a matrix, and higher dimensional tensors are just... tensors.

In function documentation, tensor shapes are denoted with capital letters and separated by `x` symbols. Usually each dimension of a tensor will have semantic meaning. For example, a two-dimensional tensor giving a batch of token IDs might have a shape `BSxL`, where `BS` denotes a batch size and `L` a sequence length.

Sometimes there will be a sort of implicit organization along a particular dimension. For example, writing that a tensor has shape `(BS C)xL` would indicate that along the zeroth dimension (of size `(BS C)`), the first `C` elements are in some way grouped together, the next `C` elements are in some way grouped together, and so on.

This notational paradigm is very useful with the [einops](https://einops.rocks/) library.
