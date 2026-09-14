# S3-R5 prospective bit-exact scalar recovery amendment

S3-R4 correctly produced zero-dimensional float64 arrays, but recomputed each control MSE from the retained float32 prediction and occupancy. The original S3-R3 MSE had been calculated before prediction down-casting, so every recomputed value differed by small float rounding (the first difference was about 1.97e-10). S3-R4 therefore has status FAILED_IMPLEMENTATION and is not eligible for S3 completion.

S3-R5 is a fresh CPU-only recovery root. It verifies each S3-R3 terminal and array and rebinds them. For each of 3,424 VALID D2 units it reads the original float64 shape-[1] control MSE, requires exactly one finite element, and writes that same float64 element bit-exactly as a zero-dimensional array. No image/mask/member access, decode, model load, feature extraction, prediction, fitting, alpha selection, or cohort aggregation occurs. All S3-R3 and S3-R4 evidence remains immutable.
