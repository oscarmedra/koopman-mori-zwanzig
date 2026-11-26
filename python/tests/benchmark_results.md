# Benchmark results (10 systems)

Scenario: ψ observed over the whole duration, states over a short window. RMS errors on ψ, independent test trajectories. Numbers from the last full run (French API version, same algorithms).

| System | Selected dictionary | p | Window: model / floor | End: test model / announced / floor / constant | Coverage ±2 bars | Basins | Accuracy |
|---|---|---|---|---|---|---|---|
| Damped Duffing (ẍ = −0.5ẋ + x − x³) | V, basin_sharp | 19 | 0.262 / 0.261 | 0.246 / 0.250 / 0.211 / 1.000 | 97.5% | 2 | 98.6% |
| Van der Pol (μ=1) | V, harmonic | 19 | 0.143 / 0.142 | 0.279 / 0.283 / 0.178 / 1.330 | 99.6% | 1 | – |
| Conservative Duffing | V, cesaro | 8 | 0.598 / 0.595 | 0.950 / 0.958 / 0.908 / 1.018 | 98.6% | 1 | – |
| Lotka–Volterra | V, harmonic, monomials deg 4 | 19 | 0.134 / 0.129 | 0.362 / 0.381 / 0.265 / 0.491 | 98.7% | 1 | – |
| Pendulum (cos θ, sin θ, ω) | V, harmonic, monomials deg 3, cesaro | 19 | 0.417 / 0.411 | 0.605 / 0.615 / 0.532 / 0.665 | 99.6% | 1 | – |
| Lorenz-63 | V, monomials deg 3 | 15 | 5.493 / 5.442 | 7.992 / 8.085 / 7.932 / 7.976 | 97.3% | 1 | – |
| Viscous Burgers (ν=0.05) | V, quadratic PCA 6 | 19 | 0.159 / 0.155 | 0.032 / 0.032 / 0.017 / 0.507 | 97.2% | 1 | – |
| Kuramoto–Sivashinsky (L=22) | V | 14 | 0.281 / 0.275 | 0.283 / 0.284 / 0.282 / 0.283 | 94.4% | 1 | – |
| Allen–Cahn (ε²=0.36) | V | 19 | 0.488 / 0.485 | 0.881 / 0.839 / 0.616 / 0.999 | 95.7% | 2 | 94.4% |
| Bistable Nagumo (D=0.1, a=0.3) | V, basin_sharp | 2 (unstable) | 0.114 / 0.113 | 1.645 / 1.626 / 0.228 / 0.488 | 94.9% | 2 | 94.0% |
