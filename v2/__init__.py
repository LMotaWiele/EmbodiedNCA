"""
v2 — Counterfactual self-maintenance allocation (attention budget).

Extends the v1 self-model to boost-type interventions and allocates a scarce
attention resource via argmax over self-model predictions. Does not modify
v0 dynamics, SimpleNCA, or v1 training data / architecture.
"""
