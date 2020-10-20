# As of PyTorch 1.7.0, the matrix exponential is supported natively by PyTorch under the name `matrix_exp`

https://pytorch.org/docs/master/generated/torch.matrix_exp.html?highlight=matrix_exp#torch.matrix_exp

## Use the PyTorch native implementation rather than this one in your project if you can as it is faster

Two differentiable implementations of the exponential of matrices in Pytorch.

They implement the papers:

- `expm_taylor.py`:
[Computing the matrix exponential with an optimized Taylor polynomial approximation](https://www.mdpi.com/2227-7390/7/12/1174)

- `expm_pade.py`: [A New Scaling and Squaring Algorithm for the Matrix Exponential](http://eprints.ma.man.ac.uk/1300/1/covered/MIMS_ep2009_9.pdf)

The Taylor implementation should run faster in GPU, as it does not require of a QR decomposition.

The Taylor implementation supports batches of square matrices of shape `(*, n ,n)`.

The Taylor implementation is done entirely in Pytorch.

The Pade implementation requires Scipy. It is itself an adaptation of the implementation of `expm` in Scipy.
