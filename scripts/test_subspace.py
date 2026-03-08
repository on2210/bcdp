import torch

from bcdp.trace.site import Site, Stream, Position
from bcdp.trace.activation_cache import ActivationCache
from bcdp.subspace.subspace import Subspace
from bcdp.subspace.diff_means import DiffMeans


def make_site(layer: int = 0) -> Site:
    return Site(layer=layer, stream=Stream.RESID_POST, position=Position.QA, index=None)


def assert_close(a, b, atol=1e-6, rtol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError("Tensors not close enough.")


def test_geometry():
    print("Running geometry test...")
    site = make_site()
    d, k = 16, 3

    Q, _ = torch.linalg.qr(torch.randn(d, k))
    ss = Subspace(basis=Q, method="test", anchor_sites=(site,))

    assert ss.d_model == d
    assert ss.k == k

    x = torch.randn(5, d)
    coords = ss.coords(x)
    assert coords.shape == (5, k)

    proj = ss.project(x)
    rem = ss.remove(x)

    # Projecting twice should be stable
    proj2 = ss.project(proj)
    assert_close(proj, proj2)

    # Remove should be orthogonal to subspace
    coords_rem = ss.coords(rem)
    assert_close(coords_rem, torch.zeros_like(coords_rem))

    print("Geometry test passed.\n")


def test_leading_dims():
    print("Running leading-dims test...")
    site = make_site()
    d, k = 12, 2
    Q, _ = torch.linalg.qr(torch.randn(d, k))
    ss = Subspace(basis=Q, method="test", anchor_sites=(site,))

    x = torch.randn(4, 7, d)
    coords = ss.coords(x)
    proj = ss.project(x)
    rem = ss.remove(x)

    assert coords.shape == (4, 7, k)
    assert proj.shape == (4, 7, d)
    assert rem.shape == (4, 7, d)

    print("Leading-dims test passed.\n")


def test_steer():
    print("Running steer test...")
    site = make_site()
    d = 10

    v = torch.randn(d)
    v = v / v.norm()
    ss = Subspace(basis=v[:, None], method="test", anchor_sites=(site,))

    x = torch.zeros(d)
    y = ss.steer(x, alpha=2.0)

    assert_close(y, 2.0 * v)
    print("Steer test passed.\n")


def test_diff_means():
    print("Running DiffMeans test...")
    torch.manual_seed(0)
    site = make_site()

    N, d = 20, 8
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2 :] = 1

    signal = torch.randn(d)
    signal = signal / signal.norm()

    X = torch.randn(N, d) * 0.01
    X[y == 1] += signal
    X[y == 0] -= signal

    cache = ActivationCache(activations={site: X}, labels=y)
    dm = DiffMeans()

    sub = dm.fit(cache, site=site)
    v = sub.basis[:, 0]

    mu_pos = X[y == 1].mean(dim=0)
    mu_neg = X[y == 0].mean(dim=0)
    v_manual = (mu_pos - mu_neg)
    v_manual = v_manual / v_manual.norm()

    cos = torch.dot(v, v_manual).abs().item()
    if cos < 0.999:
        raise AssertionError(f"DiffMeans direction mismatch. cos={cos}")

    print("DiffMeans test passed.\n")


if __name__ == "__main__":
    print("\n=== Subspace Manual Tests ===\n")

    test_geometry()
    test_leading_dims()
    test_steer()
    test_diff_means()

    print("All tests passed successfully.\n")
