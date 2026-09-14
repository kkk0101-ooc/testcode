# S3-R5-A1 prospective numeric-validator routing amendment

The first R5 numeric-validation invocation rejected the corrected scalar because it mapped logical arrays only by basename and expected control_mse.npy, while the fresh R5 payload stores each corrected scalar as corrected_control_mse/unit-NNNNN.npy. The record itself has the required float64 dtype, empty axes, and scalar shape. This is a validator routing error; no artifact failed its frozen numeric contract.

A1 recognizes exactly one record in the corrected_control_mse namespace with dtype float64, axes [], and shape [] as the logical control MSE, rejecting ambiguity. It also strengthens independent validation to recompute train-only normalization, every alpha validation MSE and tie selection, every final ridge weight, and every test prediction from sealed features and weights. No source artifact is modified and no cohort-level D2 aggregation is performed.
