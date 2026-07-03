# Interpreting RMC Convergence Behavior

## Normal convergence

A healthy RMC run shows agreement factors dropping quickly at the start, then
decreasing ever more slowly toward a plateau. Because RMC is stochastic, small
fluctuations around the plateau are normal and are not a problem. Reaching a
plateau is expected behavior — the question is whether the plateau represents
a physical configuration, not whether the numbers can be pushed lower.

## Flat from the start

If agreement factors barely move from the beginning of the run:

- Constraints may be too tight for any move to be accepted; check the move
  acceptance rate. Very low acceptance means the configuration cannot evolve.
- The starting configuration may already be at the achievable misfit given the
  constraints and data (common when restarting from a converged run).
- Dataset weights (sigmas) may be set so loosely that almost any move is
  accepted and the fit performs a random walk instead of refining.

## Oscillating agreement factors

Sustained oscillation with meaningful amplitude usually indicates competing
constraints or datasets: the configuration is being pulled alternately toward
incompatible targets. Check dataset weights and whether two datasets disagree
(see dataset conflicts). Mild oscillation around a plateau is normal noise.

## Increasing agreement factors

An agreement factor that rises over time for one dataset while another falls
signals a dataset conflict — the fit is trading one dataset against another.
If all datasets rise, check whether constraints were changed mid-run, whether
a restart loaded the wrong configuration, or whether a weight was mistyped.

## Move acceptance

Acceptance rates provide complementary evidence: very high acceptance with
flat R-values means the weights are too loose; very low acceptance means the
configuration is frozen by constraints. Moderate acceptance with plateaued
R-values is the normal converged regime.
