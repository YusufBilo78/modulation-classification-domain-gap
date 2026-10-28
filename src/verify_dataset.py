"""
verify_dataset.py -- check the assumptions a training script makes about the
data before it silently trains on garbage.

Six questions, each answered by measurement rather than by reading the paper:

  1. axis order      is X really (n, 1024, 2), and does the transpose to
                     (n, 2, 1024) keep each frame's samples together?
  2. class order     CLASSES is an assumption -- the file stores one-hot
                     integers with no names. Every confusion matrix depends
                     on it.
  3. cell layout     is the file really 24 classes x 26 SNRs x 4096 frames in
                     block order, which is what cell_indices() assumes?
  4. SNR dtype       what the loader is converting from
  5. frame power     is the per-frame normalisation a no-op or not?
  6. step time       an actual ICRNNA training step at 1024 and at 128
                     samples, so epoch estimates come from a measurement
                     rather than from a guess

Run on mains power. On battery this laptop's GPU is capped near 35 W and the
timing in (6) comes out about 5x pessimistic.
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import features
import model_zoo
import radioml

BATCH = 256
EXPECTED_REAL = {"OOK", "4ASK", "8ASK", "BPSK", "AM-DSB-WC", "AM-DSB-SC"}
# verify_classes.py established these two as expected exceptions: RadioML's SSB
# is so narrowband and carrier-dominated that |C20| does not average to zero
# over 1024 samples. The labels are right; the statistic is the wrong tool for
# them. class_spectra.py settled it from the spectra instead.
EXPECTED_AMBIGUOUS = {"AM-SSB-WC", "AM-SSB-SC"}


def rule(title):
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")


def main() -> None:
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")
    X, y, z = ds.X, ds.labels, ds.snrs
    n_cls, n_snr = len(radioml.CLASSES), len(np.unique(z))

    # ---------------------------------------------------------------- 1
    rule("1. Axis order")
    print(f"X {X.shape} {X.dtype}   y {y.shape} {y.dtype}   z {z.shape} {z.dtype}")
    assert X.ndim == 3, "expected 3 axes"
    short = int(np.argmin(X.shape[1:])) + 1
    print(f"shortest non-batch axis: {short} (length {X.shape[short]}) -> I/Q")

    raw = np.asarray(X[:4])                       # (4, 1024, 2)
    tr = np.transpose(raw, (0, 2, 1))             # (4, 2, 1024)
    same = all(np.array_equal(tr[k, c], raw[k, :, c])
               for k in range(4) for c in (0, 1))
    print(f"transpose (0,2,1) keeps each frame's samples together: {same}")

    # A time axis has temporal structure; an I/Q axis of length 2 cannot. If
    # the axes were swapped, lag-1 autocorrelation along the '1024' axis would
    # look like noise.
    #
    # The frame has to be a clean one. X[0] is the first block of the file --
    # class 0 at -20 dB -- which is almost entirely noise, so its correlation
    # is ~0 for reasons that have nothing to do with axis order. Sampling a
    # high-SNR frame instead is the difference between testing the layout and
    # testing the noise floor.
    print()
    print(f"{'class':<8} {'SNR':>5} {'lag-1 autocorr':>15} {'I vs Q corr':>13}")
    for ci, snr in ((3, 30), (12, 30), (3, -20)):
        f2 = np.asarray(X[ds.cell_indices(ci, snr)[0]])
        v = f2[:, 0].astype(np.float64)
        v -= v.mean()
        ac1 = float(np.dot(v[:-1], v[1:]) / np.dot(v, v))
        iqc = float(np.corrcoef(f2[:, 0], f2[:, 1])[0, 1])
        print(f"{radioml.CLASSES[ci]:<8} {snr:>5} {ac1:>15.4f} {iqc:>13.4f}")
    print("High-SNR frames must show strong lag-1 correlation (oversampled "
          "symbols).")
    print("The -20 dB row is the control: noise, so near zero.")

    # ---------------------------------------------------------------- 2
    rule("2. Class order (CLASSES is an assumption, not stored in the file)")
    print("|C20| ~ 1 for a real-valued signal, ~ 0 for a proper complex one.\n")
    hits = misses = 0
    print(f"{'idx':>3} {'class':<11} {'|C20|':>7}  verdict")
    for ci, name in enumerate(radioml.CLASSES):
        rows = ds.cell_indices(ci, 30)[:96]
        frames = np.asarray(X[np.sort(rows)])
        iq = (frames[..., 0] + 1j * frames[..., 1]).astype(np.complex64)
        c20 = float(np.mean([features.cumulants(f)["|C20|"] for f in iq]))
        is_real = c20 > 0.5
        want_real = name in EXPECTED_REAL
        if name in EXPECTED_AMBIGUOUS:
            tag = "known exception"
        elif is_real == want_real:
            tag = "OK"; hits += 1
        else:
            tag = f"MISMATCH (expected {'real' if want_real else 'complex'})"
            misses += 1
        print(f"{ci:>3} {name:<11} {c20:>7.3f}  {tag}")
    testable = len(radioml.CLASSES) - len(EXPECTED_AMBIGUOUS)
    print(f"\n{hits}/{testable} testable classes match; {misses} mismatched")

    # ---------------------------------------------------------------- 3
    rule("3. Cell layout")
    per_cell = X.shape[0] // (n_cls * n_snr)
    print(f"{X.shape[0]:,} frames / ({n_cls} classes x {n_snr} SNRs) "
          f"= {per_cell} per cell")
    ok = True
    for ci in (0, 7, 23):
        for snr in (-20, 0, 30):
            rows = ds.cell_indices(ci, snr)
            good = (len(rows) == per_cell
                    and np.all(y[rows] == ci) and np.all(z[rows] == snr)
                    and rows.max() - rows.min() == per_cell - 1)
            ok &= good
            if not good:
                print(f"  BROKEN cell (class {ci}, {snr} dB)")
    print(f"spot-checked 9 cells: {'contiguous, correctly labelled' if ok else 'BROKEN'}")

    # ---------------------------------------------------------------- 4
    rule("4. SNR values")
    u = np.unique(z)
    print(f"dtype {z.dtype}, {len(u)} distinct: {u.min()} .. {u.max()} dB, "
          f"step {int(u[1] - u[0])}")
    print(f"integer-valued: {np.all(u == u.astype(np.int64))}")

    # ---------------------------------------------------------------- 5
    rule("5. Raw frame power (is normalisation a no-op?)")
    print(f"{'SNR':>5} {'mean power':>12} {'median':>10} {'min':>10} {'max':>10}")
    for snr in (-20, 0, 30):
        rows = np.sort(np.concatenate(
            [ds.cell_indices(ci, snr)[:32] for ci in range(0, 24, 4)]))
        fr = np.asarray(X[rows]).astype(np.float64)
        p = (fr[..., 0] ** 2 + fr[..., 1] ** 2).mean(axis=1)
        print(f"{snr:>5} {p.mean():>12.4f} {np.median(p):>10.4f} "
              f"{p.min():>10.4f} {p.max():>10.4f}")
    print("\nUnit power would make the per-frame normalisation a no-op.")
    ds.close()

    # ---------------------------------------------------------------- 6
    rule("6. ICRNNA step time (measured, not estimated)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f"{torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"batch {BATCH}, fp16 autocast, forward + backward + step\n")

    print(f"{'frame':>7} {'LSTM steps':>11} {'ms/batch':>10} "
          f"{'frames/s':>10} {'VRAM':>8}")
    results = {}
    for length in (128, 1024):
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        model = model_zoo.ICRNNA(24).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        xb = torch.randn(BATCH, 2, length, device=dev)
        yb = torch.randint(0, 24, (BATCH,), device=dev)
        with torch.no_grad():
            # model_zoo.ICRNNA keeps its two conv blocks separate; the
            # colab/ build is the one with a single `.cnn` Sequential.
            t_steps = model.conv2(model.conv1(xb[:1])).shape[-1]

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(5):                       # warm-up: cuDNN autotune
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        vram = (torch.cuda.max_memory_allocated() / 1e9
                if dev.type == "cuda" else 0.0)
        results[length] = BATCH / (ms / 1000)
        print(f"{length:>7} {t_steps:>11} {ms:>10.1f} "
              f"{results[length]:>10,.0f} {vram:>7.1f}G")
        del model, opt, xb, yb
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    ratio = results[128] / results[1024]
    print(f"\n128-sample frames are {ratio:.1f}x cheaper per frame "
          f"(recurrence does not parallelise over time)")

    rule("Epoch estimates from the measurement above")
    print(f"{'setup':<44} {'frames':>10} {'min/epoch':>11} {'epochs/h':>9}")
    plans = [
        ("2018.01A, 24 cls, 256/cell, 1024 samples", 24 * 26 * 256, 1024),
        ("2018.01A, 24 cls, 512/cell, 1024 samples", 24 * 26 * 512, 1024),
        ("2018.01A,  5 cls, 768/cell, 1024 samples", 5 * 26 * 768, 1024),
        ("2018.01A, 200k subsample, 128 crop", 200_000, 128),
        ("2016.10a, 220k frames, 128 samples", 220_000, 128),
    ]
    for name, n, length in plans:
        n_train = int(n * 0.70)
        mins = n_train / results[length] / 60
        print(f"{name:<44} {n_train:>10,} {mins:>11.1f} {60/mins:>9.1f}")
    print("\nTraining only; evaluation adds roughly 15-25% on top.")


if __name__ == "__main__":
    main()
