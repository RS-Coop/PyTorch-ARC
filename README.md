# PyTorch ARC: Adaptive Regularization with Cubics

### Authors: [Cooper Simpson](https://rs-coop.github.io/), [Jaden Wang](https://github.com/tholdem)

A PyTorch implementation of the ARC algorithm: a second-order method for non-convex optimization. To that end, we consider a problem of the following form
$$\min_{\mathbf{x}\in \mathbb{R}^n}f(\mathbf{x})$$
where $f:\mathbb{R}^n\to\mathbb{R}$ is a twice continuously differentiable function. Each iteration applies an update of the following form:
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}+\mathbf{s}^{(k)}$$
where the update $\mathbf{s}^{(k)}$ is computed as the minimizer the following cubic sub-problem:
$$\min_{s\in\mathbb{R}^n}\nabla f(\mathbf{x^{(k)}})s+\frac{1}{2}\mathbf{s}^T\mathbf{B}^{(k)}\mathbf{s}+\frac{\sigma_k}{3}\|\mathbf{s}\|^3$$
The matrix $\mathbf{B}^{(k)}$ is an approximation of the Hessian $\nabla^2f(\mathbf{x}^{(k)})$, and $\sigma_k$ is the adaptive regularization term.

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the entry in `CITATION`. See [References](#references) for a full list of publications related to ARC and influencing this package. If any of these are useful to your own work, please cite them individually.

## Installation
This package can be installed via pip. From the terminal, run the following command:
```console
pip install pytorch-arc
```

### Testing

## Usage

```python
from torch_arc import arc
```

## References

### []()
```bibtex
@article{
}
```
