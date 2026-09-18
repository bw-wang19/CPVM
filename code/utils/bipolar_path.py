"""Shared geometry and quadrature for bipolar model merging and attribution.

Write ``delta_plus = theta_plus - theta_base`` and
``delta_star = theta_star - theta_base``. Every path is represented by
``theta(t) = theta_base + c_plus(t) * delta_plus + c_star(t) * delta_star``,
where ``-1 <= t <= 1``. The low/star endpoint is at -1 and the high/plus
endpoint is at +1. Piecewise-linear and quadratic paths pass through the
base at 0; the endpoint-linear path passes through the endpoint midpoint.

For a scalar objective F, parameter attribution is
``C_i = integral_0^1 partial_i F(theta(t(alpha))) * dtheta_i/dalpha d alpha``.
Piecewise-linear attribution uses separate outward branches from t=0.
Endpoint-linear and quadratic attribution instead follow the entire path
from t=-1 to +1 with one fixed high-pole objective and one common result.
At a quadrature point, ``dtheta/dalpha`` is
``tangent_plus * delta_plus + tangent_star * delta_star``. Consequently,
the tangents returned by :func:`branch_points` and :func:`full_path_points`
already include the orientation and derivative ``dt/dalpha``. Callers must not
multiply them by another endpoint delta or scheduling derivative.

``interpolation='parabolic'`` changes branch speed to ``t=+/-alpha**2``;
on a full path it uses ``s=2*alpha-1`` and ``t=sign(s)*s**2``. It does not
select the quadratic geometry. With exact integration, a speed change leaves
attribution unchanged. Finite quadrature may differ. ``k=1`` preserves the
endpoint-gradient times net-displacement baseline: endpoint minus center
for a branch, high endpoint minus low endpoint for the common full path.
It is not a one-node Gauss-Legendre integral or a local endpoint tangent.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


PATH_METHODS = ("piecewise_linear", "endpoint_linear", "quadratic")
SUPPORTED_K = (1, 2, 4, 8, 16, 32, 64)
_POLES = ("high", "low")
_INTERPOLATIONS = ("linear", "parabolic")


@dataclass(frozen=True)
class BipolarPathPoint:
    """One path integration point; tangent coefficients are w.r.t. alpha.

    ``plus`` and ``star`` position the model relative to the common base.
    ``weight`` is a quadrature weight on alpha in [0, 1]. For ``k=1``,
    the tangent fields instead contain the integration path's net displacement
    coefficients, as required by the endpoint-gradient baseline.
    """

    alpha: float
    t: float
    weight: float
    plus: float
    star: float
    tangent_plus: float
    tangent_star: float


def validate_path_method(method: str) -> str:
    """Return a supported path name, or reject a misspelled/unknown name."""

    if not isinstance(method, str) or method not in PATH_METHODS:
        raise ValueError(f"path method must be one of {PATH_METHODS}, got {method!r}")
    return method


def _validate_t(t: float) -> float:
    if isinstance(t, bool) or not isinstance(t, Real):
        raise ValueError("t must be a finite real number in [-1, 1]")
    value = float(t)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("t must be a finite real number in [-1, 1]")
    return value


def path_coefficients(t: float, method: str) -> tuple[float, float]:
    """Return ``(c_plus, c_star)`` positioning theta(t) relative to the base.

    The quadratic is the unique degree-at-most-two parameter curve through
    ``theta_star`` at -1, ``theta_base`` at 0, and ``theta_plus`` at +1.
    It need not remain in the convex hull of these three model vectors.
    """

    method = validate_path_method(method)
    t = _validate_t(t)
    if method == "piecewise_linear":
        return max(t, 0.0), max(-t, 0.0)
    if method == "endpoint_linear":
        return (1.0 + t) / 2.0, (1.0 - t) / 2.0
    return (t * t + t) / 2.0, (t * t - t) / 2.0


def path_tangent_coefficients(
    t: float,
    method: str,
    *,
    branch: str | None = None,
) -> tuple[float, float]:
    """Return ``(dc_plus/dt, dc_star/dt)`` for the chosen path geometry.

    These are derivatives with respect to t, not alpha. At the corner
    ``t=0`` of a piecewise-linear path, ``branch='high'`` selects the
    right-hand derivative and ``branch='low'`` the left-hand derivative.
    A branch is required there because no unique two-sided derivative
    exists. Away from that corner the derivative is independent of branch.
    """

    method = validate_path_method(method)
    t = _validate_t(t)
    if branch is not None and branch not in _POLES:
        raise ValueError(f"branch must be one of {_POLES} or None")
    if method == "piecewise_linear":
        if t > 0.0:
            return 1.0, 0.0
        if t < 0.0:
            return 0.0, -1.0
        if branch == "high":
            return 1.0, 0.0
        if branch == "low":
            return 0.0, -1.0
        raise ValueError("piecewise_linear at t=0 requires branch='high' or 'low'")
    if method == "endpoint_linear":
        return 0.5, -0.5
    return t + 0.5, t - 0.5


def branch_points(
    k: int,
    method: str,
    target_pole: str,
    interpolation: str = "linear",
) -> list[BipolarPathPoint]:
    """Build K integration points from the path center to one endpoint.

    ``high`` follows t=0 to +1; ``low`` follows t=0 to -1. For K>1,
    Gauss-Legendre nodes and weights are mapped to alpha in [0, 1].
    A caller approximates attribution elementwise as
    ``sum(point.weight * grad_F_at_point *
    (point.tangent_plus * delta_plus + point.tangent_star * delta_star))``.
    Summing exact attributions across all coordinates recovers
    ``F(theta_endpoint) - F(theta_center)`` for differentiable F.

    K=1 instead returns the endpoint, weight 1, and net displacement
    coefficients; it intentionally retains the endpoint-gradient baseline
    for both schedules and all three geometries. K is restricted to the
    same supported choices as the existing single-model path integrator.
    """

    if isinstance(k, bool) or not isinstance(k, Integral) or k not in SUPPORTED_K:
        raise ValueError(f"k must be an integer in {SUPPORTED_K}")
    method = validate_path_method(method)
    if target_pole not in _POLES:
        raise ValueError(f"target_pole must be one of {_POLES}")
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {_INTERPOLATIONS}")

    direction = 1.0 if target_pole == "high" else -1.0
    if k == 1:
        plus, star = path_coefficients(direction, method)
        center_plus, center_star = path_coefficients(0.0, method)
        return [
            BipolarPathPoint(
                alpha=1.0,
                t=direction,
                weight=1.0,
                plus=plus,
                star=star,
                tangent_plus=plus - center_plus,
                tangent_star=star - center_star,
            )
        ]

    nodes, weights = np.polynomial.legendre.leggauss(int(k))
    points = []
    for node, weight in zip(nodes, weights):
        alpha = float((node + 1.0) / 2.0)
        if interpolation == "linear":
            t = direction * alpha
            dt_dalpha = direction
        else:
            t = direction * alpha * alpha
            dt_dalpha = direction * 2.0 * alpha
        plus, star = path_coefficients(t, method)
        tangent_plus, tangent_star = path_tangent_coefficients(
            t, method, branch=target_pole
        )
        points.append(
            BipolarPathPoint(
                alpha=alpha,
                t=t,
                weight=float(weight / 2.0),
                plus=plus,
                star=star,
                tangent_plus=tangent_plus * dt_dalpha,
                tangent_star=tangent_star * dt_dalpha,
            )
        )
    return points


def full_path_points(
    k: int,
    method: str,
    interpolation: str = "linear",
) -> list[BipolarPathPoint]:
    """Build one common low-to-high integral on t in [-1, 1].

    The model objective stays fixed along this path. With the two channel
    objectives log P(high) and -log P(low), their sum attributes the entire
    change of log P(high)-log P(low) from theta_star to theta_plus.

    K>1 uses Gauss-Legendre quadrature on alpha in [0, 1]. Linear traversal
    is t=2*alpha-1 with dt/dalpha=2. Parabolic traversal is
    t=sign(s)*s**2, s=2*alpha-1, with dt/dalpha=4*abs(s), including zero
    derivative at s=0. All Jacobian factors are included in the returned
    tangents. K=1 uses the high endpoint gradient and theta_plus-theta_star.
    """

    if isinstance(k, bool) or not isinstance(k, Integral) or k not in SUPPORTED_K:
        raise ValueError(f"k must be an integer in {SUPPORTED_K}")
    method = validate_path_method(method)
    if method == "piecewise_linear":
        raise ValueError("piecewise_linear uses separate branch_points, not a common full path")
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {_INTERPOLATIONS}")

    if k == 1:
        return [BipolarPathPoint(
            alpha=1.0,
            t=1.0,
            weight=1.0,
            plus=1.0,
            star=0.0,
            tangent_plus=1.0,
            tangent_star=-1.0,
        )]

    nodes, weights = np.polynomial.legendre.leggauss(int(k))
    points = []
    for node, weight in zip(nodes, weights):
        s = float(node)
        alpha = (s + 1.0) / 2.0
        if interpolation == "linear":
            t, dt_dalpha = s, 2.0
        else:
            t, dt_dalpha = s * abs(s), 4.0 * abs(s)
        plus, star = path_coefficients(t, method)
        tangent_plus, tangent_star = path_tangent_coefficients(t, method)
        points.append(BipolarPathPoint(
            alpha=alpha,
            t=t,
            weight=float(weight / 2.0),
            plus=plus,
            star=star,
            tangent_plus=tangent_plus * dt_dalpha,
            tangent_star=tangent_star * dt_dalpha,
        ))
    return points
