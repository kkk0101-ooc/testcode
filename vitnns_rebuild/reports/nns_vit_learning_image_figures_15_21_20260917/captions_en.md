# Learning image figures 15–21 · en

## Figure 15

One internal response is paired with one map in input coordinates. The illustrated linear example uses input difference [1,2;3,4], unit changes 3 and 2, and output change 7. All four maps retain the same input locations. This is an analytic example, not a measured ViT.

## Figure 16

Input-position initialization. The reference is [1,1;1,1] and the input is [2,3;4,5]. Their difference is split into four coordinate images whose sum restores the difference. This is an equivalent mathematical representation for path attribution; the implementation does not forward each sparse seed independently through the nonlinear model.

## Figure 17

Generation in an analytic linear network. For a=x1+x2, b=2x3−x4 and f=2a+0.5b, the learning images of a and b are [1,2;0,0] and [0,0;6,−4]. Their weighted combination is [2,4;3,−2], summing to the output change 7. For nonlinear ViT targets the corresponding coefficients are integrated along the input path.

## Figure 18

ViT operator adaptation. A patch channel with weights [1,0,0,0.5] produces the local contribution map [1,0;0,2] from input differences [1,2;3,4]. Position and bias terms remain in the forward model. Attention requires both dA and dV terms on the full path; the fixed-attention condition suppresses dA. Head and identity components jointly form a complete cut. The drawing is schematic.

## Figure 19

Measured class-conditioned attention-branch learning images and token responses at blocks 0,2,4,6,8,11. The input is the first frozen Y2 cohort sample, ILSVRC2012_val_00007922.JPEG, with clean predicted-class logit target 235. The first two rows show positive and negative-magnitude input-cell contributions, respectively, from sealed 56×56 component arrays. The last row shows RMS across the 768 channels of the corresponding attention branch output at each of the 196 patch tokens; it is obtained by one retained CPU forward with matching target/logit. Input coordinates and internal token indices are different coordinate systems. Each nonnegative panel uses its own P99 scale and nearest display. No correspondence strength or model-wide trend is inferred from this single input.

## Figure 20

Class-logit component learning images for heads 0,5,11 at blocks 0,5,11, on the same image and target as Figure 19. Positive signed input-cell contributions are shown with per-panel P99 scaling. These fixed index subsets do not rank heads by appearance or performance; every head and both signs are included in the attached 12 block pages. Head numbers in different blocks identify different parameter sets.

## Figure 21

Component reconstruction at block 5 for the same class target. The raw residual map plus the sum of all twelve raw head maps reconstructs the total input-cell IG map to numerical precision. Addition is performed before coloring. All panels share sign(m) log1p(9 abs(m)/L)/log(10), where L is the maximum absolute raw value across the four panels. This checks complete-cut accounting, not a new fidelity or causal test.
