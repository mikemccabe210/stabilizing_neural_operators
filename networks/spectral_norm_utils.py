from enum import Enum, auto

import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F

from typing import Optional

__all__ = ['orthogonal', 'spectral_norm']



class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        depth_separable=False,
        dist_complex = False
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.dist_complex = dist_complex
        self.depth_separable = depth_separable
        if self.dist_complex:
            weight = torch.view_as_complex(weight)
            if self.dim != 0:
                self.dim -= 1
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            if min(weight_mat.shape) < 10:
                self.use_power_iter = False
            else:
                self.use_power_iter = True
                h, w = weight_mat.size()

                u = weight_mat.new_empty(h).normal_(0, 1)
                v = weight_mat.new_empty(w).normal_(0, 1)
                u = F.normalize(u, dim=0, eps=self.eps)
                v = F.normalize(v, dim=0, eps=self.eps)
                if self.dist_complex:
                    u = torch.view_as_real(u)
                    v = torch.view_as_real(v)
         
                self.register_buffer('_u', u)
                self.register_buffer('_v', v)

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
                self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:

        # Precondition
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            if self.dist_complex:
                    u = torch.view_as_complex(self._u)
                    v = torch.view_as_complex(self._v)
            else:
                u = self._u
                v = self._v
            u = F.normalize(torch.mv(weight_mat, v),      # type: ignore[has-type]
                                  dim=0, eps=self.eps, out=u)   # type: ignore[has-type]
            v = F.normalize(torch.mv(weight_mat.t().conj(), u),
                                  dim=0, eps=self.eps, out=v)   # type: ignore[has-type]
            if self.dist_complex:
                self._u = torch.view_as_real(u)
                self._v = torch.view_as_real(v)
            else:
                self._u = u
                self._v = v

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.dist_complex:
            weight = torch.view_as_complex(weight)
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        elif self.depth_separable:
            weight_mat = weight.reshape(weight.shape[0]*weight.shape[1], weight.shape[2]*weight.shape[3])
            sigmas = torch.linalg.norm(weight_mat, dim=1)
            sigma = sigmas.max()
            if self.dist_complex:
                weight = torch.view_as_real(weight)
            if sigma > 1:
                return weight/sigma
            else:
                return weight
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                if self.use_power_iter:
                    self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            if self.use_power_iter:
                u = self._u.clone(memory_format=torch.contiguous_format)
                v = self._v.clone(memory_format=torch.contiguous_format)
                if self.dist_complex:
                    u = torch.view_as_complex(u)
                    v = torch.view_as_complex(v)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
                sigma = torch.dot(u.conj(), torch.mv(weight_mat, v)) 
            else:
                sigma = torch.linalg.svdvals(weight_mat)[0]
            sigma = sigma.abs()
            if self.dist_complex:
                weight = torch.view_as_real(weight)
            if sigma > 1:
                return weight / sigma
            else:
                return weight

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value

def spectral_norm(module: Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None,
                  depth_separable=False,
                  dist_complex=False) -> Module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.


    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-determenistic")
        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> snm
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _SpectralNorm()
            )
          )
        )
        >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps,
        depth_separable=depth_separable, dist_complex=dist_complex))
    return module

