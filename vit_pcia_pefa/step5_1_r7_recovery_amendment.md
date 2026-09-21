# Step 5.1 r7 reporting-only resource-lineage recovery amendment

The final Step-5.1 completion audit found that `step5_1_r6_20260921` started its
cumulative Step-5 total from the extension seal value `12.75807705888834` H20-hours.
That omitted two costs already sealed by `step5_completion_20260921`: the completed
DeiT float64 diagnosis (`0.027427571121693795` H20-hours) and the conservative upper
bound for the interrupted ViT-B diagnosis (`0.2` H20-hours). The r6 Step-5.1 subtotal
of `1.1972829116923434` H20-hours is correct, but its cumulative total is incomplete.

r7 is a reporting-only recovery. It inherits every r6 A0, A1 and C1 artifact and the
complete queue by byte digest. No image, checkpoint, model or GPU is opened. It changes
no intervention, score, baseline, threshold, precision or scientific decision. The
correct conservative cumulative total is
`12.985504630010032 + 1.1972829116923434 = 14.182787541702375` H20-hours, still below
the frozen 64 H20-hour cap. Evaluation remains unread, B1 remains unstarted because the
A1 incremental gate is closed, and Step 6 remains unauthorized.
