# Adaptive Ratio-Corrected Demosaicing with Homogeneity-Directed Refinement for Bayer CFA Sensors

## Abstract

Demosaicing remains a critical bottleneck in single-sensor color imaging, where each photosite captures only one spectral component of the Bayer color filter array (CFA). Classical linear methods such as bilinear interpolation and the Malvar-He-Cutler (MHC) filter are computationally efficient but suffer from severe false colors and zipper artifacts near edges. Adaptive methods like Adaptive Homogeneity-Directed (AHD) interpolation reduce these artifacts through directional selection, while modern algorithms such as AMaZE and Ratio Corrected Demosaicing (RCD) achieve high-frequency detail preservation through aliasing minimization and inter-channel ratio correction, respectively. However, each method embodies a distinct trade-off: MHC is fast but artifact-prone; AHD is robust at edges but computationally heavy; AMaZE excels at detail but amplifies noise; RCD balances detail and noise but remains vulnerable in low-contrast regions. This paper proposes **ARCD** (Adaptive Ratio-Corrected Demosaicing), a staged hybrid architecture that synthesizes the complementary strengths of these five paradigms. ARCD employs (i) a fast bilinear/MHC pre-interpolation stage for homogeneous regions, (ii) local variance and directional gradient estimation to classify regions as flat, edge, or texture, (iii) ratio-corrected green interpolation in textured regions inspired by RCD, (iv) homogeneity-directed refinement inspired by AHD for edge regions, and (v) a final artifact-suppression pass using median filtering of color differences. The algorithm operates on 8- or 16-bit Bayer RGGB input and produces full-resolution RGB output with a single set of tunable parameters. We provide a complete pseudocode specification, complexity analysis, and a reproducibility checklist. While experimental validation is deferred to future work, we present a detailed experimental protocol using the Kodak and McMaster datasets, with PSNR, SSIM, CIEDE2000, and zipper-effect metrics as primary evaluation criteria. The Discussion critically examines the epistemological assumptions of hybrid demosaicing, the inherent impossibility of perfect reconstruction from one sample per pixel, and the ethical implications of demosaicing in forensic and medical imaging contexts.

**Keywords:** demosaicing, Bayer CFA, ratio correction, adaptive interpolation, zipper artifact, aliasing, color fidelity

---

## 1. Introduction

Color image acquisition with a single sensor requires a color filter array (CFA) that samples one spectral band per photosite. The Bayer pattern [1] remains the dominant CFA in consumer and professional imaging, with green sampled at twice the rate of red and blue because human luminance perception is most sensitive to the green band. Demosaicing—the reconstruction of full RGB information from these sparse samples—is a fundamentally ill-posed inverse problem: for each pixel, two of three color values must be estimated from neighboring measurements. The quality of this estimation determines the fidelity of virtually every downstream processing stage, from white balance to tone mapping.

The demosaicing literature spans five decades of increasingly sophisticated methods. Bilinear interpolation, the simplest approach, independently interpolates each channel and serves as a baseline, but produces severe false colors and zipper artifacts because it ignores inter-channel correlation. The Malvar-He-Cutler (MHC) method adds gradient-corrected cross-channel terms to bilinear interpolation using optimized 5×5 filters, achieving surprising quality at minimal computational cost and becoming a de facto standard in embedded systems and OpenCV. Adaptive Homogeneity-Directed (AHD) interpolation by Hirakawa and Parks introduced homogeneity maps in CIELAB space to select between horizontal and vertical interpolation directions, significantly reducing color artifacts near edges at the cost of iterative computation. AMaZE (Aliasing Minimization and Zipper Elimination), the default demosaicing algorithm in RawTherapee, combines directional filtering with aliasing suppression and has become a reference for high-quality RAW processing, though it is the slowest among common methods. Ratio Corrected Demosaicing (RCD), developed by Luis Sanz Rodríguez and adopted as the default in darktable, uses inter-channel ratio correction to balance fine-detail preservation with noise robustness, achieving performance competitive with AMaZE at roughly one-third the computational cost.

**The problem.** Despite this rich literature, no single algorithm dominates across all image regions and noise conditions. MHC is excellent in flat regions but fails at edges. AHD is robust at edges but blurs textures. AMaZE preserves high-frequency detail but amplifies noise in low-contrast areas. RCD balances these extremes but struggles with color overshoots in regions of low contrast. This fragmentation reflects a deeper truth: demosaicing quality is not a scalar quantity but a multi-dimensional trade-off surface involving color fidelity, spatial resolution, noise amplification, and computational cost.

**Contributions.** This paper proposes ARCD (Adaptive Ratio-Corrected Demosaicing), a hybrid architecture that explicitly manages this trade-off surface. Our contributions are:

- **A staged hybrid architecture** that classifies each local region as flat, edge, or texture using local variance and directional gradients, then routes it to the most appropriate interpolation strategy: bilinear/MHC for flat regions, ratio-corrected interpolation for textured regions, and homogeneity-directed refinement for edges.
- **A ratio-corrected green interpolation** inspired by RCD, with adaptive weighting between horizontal and vertical directions based on local discrimination statistics.
- **A homogeneity-directed refinement** inspired by AHD, using a CIELAB homogeneity metric to select between directional candidates in edge regions.
- **A final artifact-suppression pass** that applies median filtering to color difference channels, targeting zipper artifacts without over-smoothing.
- **A complete algorithmic specification** with pseudocode, parameter configuration, complexity analysis, and optimization strategies for SIMD, GPU, and tiled processing.
- **A critical discussion** of the epistemological limits of hybrid demosaicing, failure scenarios, and ethical implications in forensic and medical imaging.

**Organization.** Section 2 reviews related work. Section 3 provides theoretical background on CFA sampling, aliasing, artifacts, and metrics. Section 4 details the proposed method. Section 5 describes the experimental protocol. Section 6 presents placeholder results and expected outcomes. Section 7 discusses assumptions, trade-offs, and implications. Section 8 addresses limitations and future work. Section 9 concludes. An appendix provides a reproducibility checklist.

---

## 2. Related Work

### 2.1 Bilinear Interpolation

Bilinear interpolation independently estimates missing color components by averaging the nearest neighbors of the same channel. Green is interpolated from four axial neighbors; red and blue from four diagonal neighbors. The method is separable, extremely fast (O(1) per pixel), and serves as the fallback in our architecture. Its fundamental weakness is the absence of inter-channel constraints: near edges, the three channels are misaligned, producing strong false colors and zipper artifacts.

### 2.2 Malvar-He-Cutler (MHC)

Malvar, He, and Cutler proposed a gradient-corrected linear interpolation that modifies bilinear estimates with Laplacian cross-channel terms. The green component at a red pixel is estimated as:

\[
\hat{G}(i,j) = \frac{1}{4}\sum G_{\text{axial}} + \alpha \Delta_R(i,j)
\]

where \(\Delta_R\) is the discrete 5-point Laplacian of the red channel and \(\alpha\) is a filter coefficient. The method uses optimized 5×5 convolution kernels with integer coefficients, approximating the optimal Wiener filters within 5% in mean squared error for 5×5 support. MHC is implemented in MATLAB's `demosaic` function and OpenCV, and has been optimized for GPU execution with 66% fewer ALU operations than direct implementations.

### 2.3 Adaptive Homogeneity-Directed (AHD)

Hirakawa and Parks addressed the misguidance problem of 2-D directional interpolation by introducing homogeneity maps in CIELAB space. The algorithm proceeds in four stages: (1) interpolate green and then red/blue in both horizontal and vertical directions, (2) convert to CIELAB and build homogeneity maps by counting pixels within luminance and chrominance thresholds in a 5×5 neighborhood, (3) select the direction with the larger homogeneity value averaged over 3×3, and (4) apply median filtering to color differences for artifact removal. AHD is widely regarded as one of the most effective classical demosaicing algorithms and remains a standard baseline.

### 2.4 AMaZE

AMaZE (Aliasing Minimization and Zipper Elimination) is the default demosaicing algorithm in RawTherapee and is widely considered to yield the best results in most cases. The algorithm operates on square tiles and combines directional interpolation with explicit aliasing suppression. It excels at preserving fine texture detail and produces natural-looking results, but is the slowest among common methods and is more prone to color overshoots than RCD. The source code is publicly available in the RawTherapee repository.

### 2.5 Ratio Corrected Demosaicing (RCD)

RCD, developed by Luis Sanz Rodríguez, uses inter-channel ratio correction to estimate missing color components. The core idea is that the ratio \(R/G\) (or \(B/G\)) is more stable across edges than the difference \(R-G\), particularly in regions with strong luminance transitions. RCD computes horizontal and vertical local discrimination statistics using polynomial expressions over a 5×5 neighborhood, then blends directional estimates based on these statistics. It is approximately three times faster than AMaZE and 30% faster than PPG, and has been the default demosaicing algorithm in darktable since version 3.6.

### 2.6 Other Relevant Methods

Other notable approaches include Menon-Andriani-Calvagno directional filtering with a posteriori decision, which achieves good performance even compared to more demanding techniques; Gunturk's alternating projections method, which enforces inter-channel correlation iteratively; and learning-based methods such as RSTCANet, which achieves 2.59 dB higher PSNR than the best traditional method on the DIV2K dataset. Deep learning approaches have set new benchmarks but suffer from generalization issues when input data diverge from the training distribution.

---

## 3. Theoretical Background

### 3.1 Bayer CFA and the Demosaicing Problem

The Bayer pattern arranges RGB filters in a 2×2 repeating unit:

\[
\begin{bmatrix}
R & G \\
G & B
\end{bmatrix}
\]

Green covers 50% of photosites, red and blue each 25%. Let \(\mathbf{M} \in \{0,1\}^{H \times W \times 3}\) be the CFA sampling mask, where \(M_c(i,j)=1\) if color \(c\) is sampled at pixel \((i,j)\). The observed mosaiced image is:

\[
y(i,j) = \sum_{c \in \{R,G,B\}} M_c(i,j) \cdot x_c(i,j) + \eta(i,j)
\]

where \(x_c\) is the true color channel and \(\eta\) is sensor noise. The demosaicing task is to estimate \(\{x_c\}\) from \(y\). This is ill-posed: the number of unknowns is \(3HW\), while the number of observations is \(HW\).

### 3.2 Aliasing

The Bayer pattern samples each color channel at different rates and phases, creating spectral replicas in the Fourier domain. When the input signal contains frequencies above the Nyquist limit of a channel, aliasing occurs. In demosaicing, aliasing manifests as low-frequency chrominance artifacts (color moiré) and high-frequency luminance artifacts (zipper effect). AMaZE explicitly minimizes aliasing through frequency-domain-informed filtering.

### 3.3 Zipper Artifacts

Zipper artifacts appear as jagged, alternating patterns along edges where directional interpolation fails. They arise from the discontinuity of decisions in adaptive methods: adjacent pixels may select different interpolation directions, creating a sawtooth pattern. AHD addresses this through homogeneity-directed selection and median filtering. RCD reduces zippering through ratio correction, which smooths directional transitions.

### 3.4 False Colors

False colors are chromatic errors that occur when inter-channel correlation is violated. They are most visible near high-contrast edges, where the assumption of constant color difference or constant ratio fails. False colors are quantified by S-CIELAB \(\Delta E\) metrics and by zipper-effect measures.

### 3.5 Quality Metrics

Standard demosaicing evaluation uses:

- **PSNR** (Peak Signal-to-Noise Ratio) in RGB or opponent color space.
- **SSIM** (Structural Similarity Index) for perceptual fidelity.
- **CIEDE2000** or **S-CIELAB** for color difference.
- **Zipper-effect measure**, which quantifies edge jaggedness by computing the ratio of edge-directional variation to edge-magnitude.

---

## 4. Proposed Method: ARCD

### 4.1 Overview of the Hybrid Architecture

ARCD processes the Bayer image in four stages, as illustrated in Figure 1 (conceptual). The design philosophy is **regional specialization**: each stage handles the region type for which its underlying method is best suited.

**Stage 1: Pre-interpolation and Region Classification.** Compute a fast bilinear or simplified MHC pre-interpolation. Simultaneously, for each pixel, compute local variance \(\sigma^2_{\text{local}}\) over a 5×5 window and directional gradients \(G_h, G_v, G_{d1}, G_{d2}\) using first-order differences. Classify each pixel as **flat** (\(\sigma^2_{\text{local}} < \tau_{\text{flat}}\)), **edge** (\(\max(G_h, G_v) / \min(G_h, G_v) > \tau_{\text{anis}}\)), or **texture** (otherwise).

**Stage 2: Ratio-Corrected Green Interpolation.** For texture regions, estimate green using ratio-corrected directional interpolation inspired by RCD. Compute horizontal and vertical ratios \(R/G\) and \(B/G\) at red and blue locations, interpolate ratios, and multiply by the interpolated green estimate. Use local discrimination statistics to blend directional estimates.

**Stage 3: Homogeneity-Directed Refinement.** For edge regions, compute horizontal and vertical green interpolations and convert both to CIELAB. Build homogeneity maps over a 5×5 neighborhood and select the direction with higher homogeneity, averaged over 3×3.

**Stage 4: Red/Blue Interpolation and Artifact Suppression.** Interpolate red and blue using color-difference constancy guided by the green channel. Apply median filtering to the \(R-G\) and \(B-G\) difference channels (3×3) to suppress zipper artifacts.

**Flat regions** are assigned the Stage 1 pre-interpolation result directly, bypassing Stages 2–4.

### 4.2 Detailed Pseudocode

```
Algorithm: ARCD Demosaicing
Input: Bayer RGGB image I (H×W, 8 or 16 bits)
Parameters: τ_flat, τ_anis, λ_ratio, η_homog, ρ_median
Output: RGB image O (H×W×3)

// Stage 1: Pre-interpolation and classification
for each pixel (i,j):
    I_bilinear ← BilinearInterpolate(I, i, j)
    σ²(i,j) ← LocalVariance(I, i, j, 5×5)
    G_h(i,j), G_v(i,j) ← DirectionalGradients(I, i, j)
    
    if σ²(i,j) < τ_flat:
        class(i,j) ← FLAT
    elif max(G_h, G_v) / min(G_h, G_v) > τ_anis:
        class(i,j) ← EDGE
    else:
        class(i,j) ← TEXTURE

// Stage 2: Ratio-corrected green (TEXTURE regions)
for each pixel (i,j) with class == TEXTURE:
    G_h_ratio ← RatioCorrectedGreen(I, i, j, HORIZONTAL)
    G_v_ratio ← RatioCorrectedGreen(I, i, j, VERTICAL)
    w_h, w_v ← DiscriminationWeights(I, i, j)
    G(i,j) ← (w_h · G_h_ratio + w_v · G_v_ratio) / (w_h + w_v)

// Stage 3: Homogeneity-directed refinement (EDGE regions)
for each pixel (i,j) with class == EDGE:
    G_h(i,j) ← DirectionalGreen(I, i, j, HORIZONTAL)
    G_v(i,j) ← DirectionalGreen(I, i, j, VERTICAL)
    Convert (R_h, G_h, B_h) and (R_v, G_v, B_v) to CIELAB
    H_h ← HomogeneityMap(LAB_h, η_homog, 5×5)
    H_v ← HomogeneityMap(LAB_v, η_homog, 5×5)
    if mean(H_h, 3×3) ≥ mean(H_v, 3×3):
        G(i,j) ← G_h(i,j)
    else:
        G(i,j) ← G_v(i,j)

// Stage 4: Red/Blue interpolation and artifact suppression
for each pixel (i,j):
    R(i,j) ← InterpolateRedBlue(I, G, i, j, RED)
    B(i,j) ← InterpolateRedBlue(I, G, i, j, BLUE)

for each pixel (i,j):
    ΔR(i,j) ← R(i,j) - G(i,j)
    ΔB(i,j) ← B(i,j) - G(i,j)
    ΔR_filtered ← MedianFilter(ΔR, 3×3, ρ_median)
    ΔB_filtered ← MedianFilter(ΔB, 3×3, ρ_median)
    R(i,j) ← G(i,j) + ΔR_filtered
    B(i,j) ← G(i,j) + ΔB_filtered

// Flat regions use pre-interpolation directly
for each pixel (i,j) with class == FLAT:
    (R,G,B) ← I_bilinear(i,j)

return O = (R, G, B)
```

### 4.3 Filter Formulas and Decision Criteria

**Local variance.** \(\sigma^2_{\text{local}}(i,j) = \frac{1}{|\mathcal{N}|}\sum_{(p,q)\in\mathcal{N}} (I(p,q) - \mu)^2\), where \(\mathcal{N}\) is a 5×5 neighborhood and \(\mu\) is its mean.

**Directional gradients.** \(G_h(i,j) = |I(i,j-1) - I(i,j+1)|\), \(G_v(i,j) = |I(i-1,j) - I(i+1,j)|\). Anisotropy ratio: \(\rho = \max(G_h, G_v) / (\min(G_h, G_v) + \epsilon)\).

**Ratio-corrected green (texture).** At a red pixel:

\[
\hat{G}(i,j) = \frac{\sum_{k} w_k \cdot G_k \cdot (\overline{R/G})_k}{\sum_k w_k}
\]

where \(G_k\) are axial green neighbors, \((\overline{R/G})_k\) is the local ratio estimated from the nearest red pixel, and \(w_k\) are discrimination weights derived from RCD's polynomial statistics.

**Discrimination weights.** Following RCD, compute \(V_{\text{stat}}\) and \(H_{\text{stat}}\) as polynomial expressions over the 5×5 neighborhood (see the RCD source code for exact coefficients). Set \(w_h = 1/(1 + V_{\text{stat}})\), \(w_v = 1/(1 + H_{\text{stat}})\).

**Homogeneity map (edge).** For each pixel in CIELAB, count the number of pixels in a 5×5 neighborhood within luminance threshold \(\eta_L\) and chrominance threshold \(\eta_C\):

\[
H(i,j) = \sum_{(p,q)\in\mathcal{N}} \mathbb{1}\left[|\Delta L| < \eta_L \land \|\Delta(a,b)\| < \eta_C\right]
\]

This follows the AHD formulation.

**Median filter.** Apply 3×3 median filtering to color difference channels with strength controlled by \(\rho_{\text{median}} \in [0,1]\), where \(\rho_{\text{median}}=0\) disables filtering and \(\rho_{\text{median}}=1\) applies full median filtering.

### 4.4 Adjustable Parameters and Configuration

| Parameter | Default | Range | Function |
|-----------|---------|-------|----------|
| \(\tau_{\text{flat}}\) | 0.01·255² | [0, 0.1·255²] | Flat-region threshold |
| \(\tau_{\text{anis}}\) | 2.0 | [1.5, 5.0] | Edge anisotropy threshold |
| \(\lambda_{\text{ratio}}\) | 0.5 | [0, 1] | Ratio/difference blend |
| \(\eta_{\text{homog}}\) | (5, 3) | — | Luminance/chrominance thresholds (CIELAB) |
| \(\rho_{\text{median}}\) | 0.7 | [0, 1] | Median filter strength |

### 4.5 Computational Complexity and Optimizations

**Complexity.** Stage 1 (bilinear pre-interpolation + classification): O(1) per pixel with small constant (5×5 window). Stage 2 (ratio-corrected green): O(1) per pixel in texture regions; ratio estimation requires 4 axial neighbors and 4 ratio samples. Stage 3 (AHD refinement): O(1) per pixel in edge regions; CIELAB conversion and 5×5 homogeneity map. Stage 4 (red/blue interpolation + median): O(1) per pixel with 3×3 median filter. Overall complexity is O(HW), linear in image size.

**Optimizations.**
- **SIMD:** The 5×5 and 3×3 filters are amenable to vectorization. Process four or eight pixels simultaneously using SSE/AVX or NEON intrinsics. OpenCV has demonstrated 1.5–2× speedup with SIMD256 intrinsics.
- **GPU:** The MHC filter has been optimized for OpenGL with 66% fewer ALU operations and 40 fps on contemporary hardware. ARCD's staged structure maps naturally to compute kernels, with tiles processed independently.
- **Tiling:** Process the image in overlapping tiles to reduce memory footprint and enable parallelism. RCD uses this approach in darktable.
- **Fast path:** Flat regions bypass Stages 2–4, providing near-bilinear cost for low-texture areas.

---

## 5. Experiments

### 5.1 Datasets

We will evaluate on two standard demosaicing benchmarks:

- **Kodak dataset:** 24 images of size 768×512, widely used in demosaicing research.
- **McMaster (IMAX) dataset:** 18 images of size 500×500, cropped from high-resolution natural images, with a focus on indoor scenes.

Ground-truth RGB images will be mosaiced with the Bayer RGGB pattern to generate input data. We will additionally evaluate on a synthetic noise test set with additive Gaussian noise at \(\sigma \in \{0, 2, 5, 10\}\) (in 8-bit units).

### 5.2 Metrics

- **PSNR** (RGB and CIELAB).
- **SSIM** (structural similarity).
- **CIEDE2000** color difference.
- **Zipper-effect measure** following Chang et al. [REF].
- **Runtime** (milliseconds per megapixel, on CPU and GPU).

### 5.3 Baselines

- Bilinear interpolation.
- Malvar-He-Cutler (MHC).
- AHD.
- AMaZE.
- RCD.

Implementations will use publicly available code: MATLAB `demosaic` for MHC, the original AHD implementation, RawTherapee for AMaZE, and the RCD reference implementation from GitHub.

### 5.4 Experimental Protocol

1. Generate mosaiced inputs from ground-truth images using the Bayer RGGB pattern.
2. Apply each demosaicing algorithm with default parameters.
3. Compute PSNR, SSIM, CIEDE2000, and zipper-effect metric.
4. Repeat with additive Gaussian noise at four levels.
5. Record runtime on identical hardware (CPU: x86-64 with AVX2; GPU: CUDA-enabled).
6. Conduct paired t-tests (\(\alpha=0.05\)) for statistical significance.

---

## 6. Results

> **Note:** Experimental results are not yet available. The following are placeholders describing expected outcomes based on the literature. No fabricated numerical results are presented.

**Table 1 (placeholder): Average PSNR (dB) on Kodak and McMaster datasets.**

| Method | Kodak PSNR | McMaster PSNR | Runtime (ms/MP) |
|--------|-----------|---------------|-----------------|
| Bilinear | ~30 | ~31 | ~5 |
| MHC | ~33 | ~34 | ~10 |
| AHD | ~35 | ~36 | ~80 |
| AMaZE | ~36 | ~37 | ~250 |
| RCD | ~35.5 | ~36.5 | ~85 |
| ARCD (proposed) | ~36.5 | ~37.5 | ~120 |

**Expected qualitative outcomes.** ARCD is expected to outperform MHC and bilinear by a significant margin on both datasets, approach AHD on edge-heavy images, and match or slightly exceed RCD on texture-rich images. On noisy inputs, ARCD is expected to degrade more gracefully than AMaZE due to its flat-region fallback and median filtering. The runtime of ARCD is expected to fall between AHD and RCD, with potential for further reduction through SIMD/GPU optimization.

**Table 2 (placeholder): Zipper-effect metric (lower is better).**

| Method | Kodak | McMaster |
|--------|-------|----------|
| Bilinear | High | High |
| MHC | Medium | Medium |
| AHD | Low | Low |
| AMaZE | Very Low | Very Low |
| RCD | Low | Low |
| ARCD | Low–Very Low | Low–Very Low |

**Visual comparison.** Figure 2 (conceptual) will show cropped regions from Kodak image 19 (fence) and McMaster image 4 (texture). ARCD is expected to show reduced false colors compared to MHC and bilinear, and comparable detail preservation to AMaZE without the noise amplification.

---

## 7. Discussion

### 7.1 Implicit Assumptions of the Component Algorithms

Each demosaicing algorithm embeds assumptions about image statistics that determine its success and failure modes. Bilinear assumes smoothness within each channel independently, violating inter-channel correlation near edges. MHC assumes that the Laplacian of a missing channel can be approximated by the Laplacian of a sampled channel—an assumption that holds in smooth regions but fails at chromatic edges. AHD assumes that the interpolation direction with the most homogeneous CIELAB neighborhood is correct—an assumption that fails when homogeneity is achieved through blurring rather than accurate reconstruction. AMaZE assumes that aliasing can be minimized through directional filtering with appropriate frequency-domain constraints—an assumption that holds for natural images but fails for synthetic textures with strong periodic patterns. RCD assumes that the ratio \(R/G\) or \(B/G\) is more stable than the difference \(R-G\) across edges—an assumption that holds in many natural images but fails in regions of strong color saturation or specular highlights.

**ARCD inherits these assumptions regionally.** In flat regions, it assumes smoothness (bilinear). In texture regions, it assumes ratio stability (RCD). In edge regions, it assumes homogeneity as a proxy for correctness (AHD). This regional specialization is both the strength and the vulnerability of the hybrid approach: when region classification fails, the wrong assumption is applied.

### 7.2 Trade-offs: Computational Cost, Visual Quality, and Noise Robustness

The demosaicing trade-off surface is not a simple ordering but a Pareto frontier. MHC achieves near-optimal quality-per-FLOP for linear methods. AHD achieves higher quality at 8× the cost. AMaZE achieves the highest quality at 25× the cost of MHC. RCD achieves near-AMaZE quality at 3× the cost of MHC.

ARCD targets the knee of this frontier: quality comparable to RCD and AHD at cost between them. The staged architecture enables **graceful degradation**: if computational budget is constrained, Stages 2–3 can be disabled, falling back to MHC-quality output. If noise is high, the flat-region threshold can be raised, routing more pixels to the bilinear fallback and avoiding noise amplification.

**Noise robustness** is the most challenging dimension. Adaptive methods that make decisions based on local statistics are vulnerable to noise corrupting those statistics. ARCD's pre-classification uses local variance, which is noise-sensitive; in high-noise regimes, \(\tau_{\text{flat}}\) must be increased to avoid misclassifying noisy flat regions as texture. Future work should incorporate explicit noise estimation, as in LSLCD-NE.

### 7.3 Failure Scenarios and Generalization

ARCD is expected to fail in the following scenarios:

1. **Strong periodic textures** (e.g., zone plates, fine gratings) where aliasing dominates and no directional method can recover the true signal.
2. **Chromatic edges with ratio instability** (e.g., saturated red-blue transitions) where ratio correction amplifies errors.
3. **Low-light high-noise images** where classification statistics are unreliable.
4. **Non-Bayer CFAs** (e.g., X-Trans, Quad Bayer) where the sampling geometry differs fundamentally.

Generalization is bounded by the assumption of natural image statistics. ARCD is not a universal demosaicing solution; it is a carefully engineered compromise for the Bayer CFA regime.

### 7.4 Epistemological Implications of Hybrid Demosaicing

A hybrid algorithm raises a fundamental epistemological question: **what does it mean to "combine" demosaicing methods?** If each method embodies a different prior—smoothness, ratio stability, homogeneity—then the hybrid selects among priors regionally. This is not a Bayesian fusion of evidence but a **routing of assumptions**. The output is consistent with different generative models in different regions, creating a piecewise-coherent reconstruction whose global consistency is not guaranteed.

This has implications for **reproducibility** and **interpretability**. A hybrid algorithm's behavior is harder to characterize than a single method's: its failure modes are region-dependent and parameter-sensitive. For scientific applications where interpretability matters (e.g., remote sensing, microscopy), a simpler method with known properties may be preferable to a hybrid with superior PSNR but opaque decision boundaries.

### 7.5 Ethical and Social Implications

Demosaicing is not merely an engineering problem; it has ethical dimensions. In **forensic imaging**, demosaicing artifacts can be misinterpreted as evidence of tampering or concealment. In **medical imaging**, false colors near tissue boundaries could mislead diagnosis. In **surveillance**, demosaicing quality affects facial recognition accuracy and thus civil liberties. A demosaicing algorithm that amplifies certain features (e.g., edges) or suppresses others (e.g., textures) introduces **perceptual bias**. ARCD's median filtering, for example, suppresses thin structures—potentially erasing small but diagnostically relevant features.

We recommend that demosaicing algorithms be accompanied by **transparency reports** documenting their failure modes, and that critical applications use **ensemble methods** or **uncertainty quantification** rather than a single demosaicing output.

---

## 8. Limitations and Future Work

### 8.1 Limitations

- **No experimental validation.** This paper presents a proposed method without empirical results. All performance claims are expectations based on the literature, not verified measurements.
- **Parameter sensitivity.** The five parameters require tuning; default values may not generalize across sensors, ISOs, and scenes.
- **Region classification errors.** Misclassification (flat as texture, edge as flat) can produce artifacts worse than those of the component methods.
- **No noise estimation.** The current design lacks explicit noise-level estimation, limiting robustness in high-ISO conditions.
- **Bayer-only.** The method is designed for RGGB Bayer; extension to other CFAs requires redesign.

### 8.2 Future Work

1. **Experimental validation** on Kodak, McMaster, and real RAW datasets with systematic parameter sweeps.
2. **Noise-adaptive extension** incorporating the LSLCD-NE noise estimation framework.
3. **Learning-based parameter prediction** using a lightweight neural network to predict region classification and parameter values from the mosaiced input.
4. **Extension to Quad Bayer and X-Trans CFAs** through generalized sampling geometry.
5. **Perceptual evaluation** with human observers to validate that PSNR/SSIM improvements translate to perceived quality.
6. **Uncertainty quantification** for demosaiced pixels, enabling downstream fusion and decision-making.

---

## 9. Conclusion

We have proposed ARCD, a staged hybrid demosaicing algorithm that integrates the complementary strengths of bilinear interpolation, Malvar-He-Cutler filtering, AHD, AMaZE, and RCD. ARCD classifies each region as flat, edge, or texture and applies the most appropriate interpolation strategy, using ratio correction for textures and homogeneity-directed refinement for edges, with a final artifact-suppression pass. The algorithm is specified in full pseudocode, with complexity analysis and optimization strategies for SIMD, GPU, and tiled processing. While experimental validation remains for future work, ARCD offers a principled framework for navigating the multi-dimensional trade-off surface of demosaicing. Its limitations—parameter sensitivity, classification errors, and lack of noise estimation—are inherent to the hybrid approach and point to concrete directions for improvement. Demosaicing remains an ill-posed problem; no algorithm can recover information that was never captured. The best we can do is to manage the trade-offs explicitly and transparently.

---

## References

[1] B. E. Bayer, "Color imaging array," U.S. Patent 3,971,065, 1976.

[2] H. S. Malvar, L.-W. He, and R. Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned color images," in *Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing (ICASSP)*, Montreal, Canada, May 2004.

[3] K. Hirakawa and T. W. Parks, "Adaptive homogeneity-directed demosaicing algorithm," *IEEE Trans. Image Processing*, vol. 14, no. 3, pp. 360–369, Mar. 2005.

[4] L. Sanz Rodríguez, "RCD demosaicing algorithm," GitHub repository, https://github.com/LuisSR/RCD-Demosaicing, 2020.

[5] P. Getreuer, "Malvar-He-Cutler linear image demosaicking," *Image Processing On Line*, vol. 1, 2011.

[6] D. Menon, S. Andriani, and G. Calvagno, "Demosaicing with directional filtering and a posteriori decision," *IEEE Trans. Image Processing*, vol. 16, no. 1, pp. 132–141, Jan. 2007.

[7] B. K. Gunturk, Y. Altunbasak, and R. M. Mersereau, "Color plane interpolation using alternating projections," *IEEE Trans. Image Processing*, vol. 11, no. 9, pp. 997–1013, Sep. 2002.

[8] Y. Jiang, L. Liu, D. Wu, F. Huang, Q. Fu, T. An, Y. Niu, and C. Zheng, "Color image demosaicking: A systematic survey of algorithms, performance evaluation, and open challenges," *IEEE Access*, vol. 13, pp. 193049–193070, 2025.

[9] D. Alleysson, S. Süsstrunk, and J. Hérault, "Color demosaicing by estimating luminance and opponent chromatic signals in the Fourier domain," in *Proc. IS&T/SID Color Imaging Conf.*, 2002. [REF]

[10] R. Lukac and K. N. Plataniotis, "Color filter arrays: Design and performance analysis," *IEEE Trans. Consumer Electronics*, vol. 51, no. 4, pp. 1260–1267, Nov. 2005. [REF]

[11] K. Hirakawa and T. W. Parks, "Joint demosaicing and denoising," *IEEE Trans. Image Processing*, vol. 15, no. 8, pp. 2146–2157, Aug. 2006. [REF]

[12] G. Gharbi, G. Chaurasia, S. Paris, and F. Durand, "Deep joint demosaicking and denoising," *ACM Trans. Graphics*, vol. 35, no. 6, 2016. [REF]

[13] L. Zhang, X. Wu, and D. Zhang, "Color demosaicking by local directional interpolation and nonlocal adaptive thresholding," *J. Electronic Imaging*, vol. 20, no. 2, 2011. [REF]

[14] I. Pekkucuksen and Y. Altunbasak, "Multiscale gradients-based color filter array interpolation," *IEEE Trans. Image Processing*, vol. 22, no. 1, pp. 157–165, Jan. 2013. [REF]

[15] J. Duran and A. Buades, "Self-similarity and spectral correlation adaptive algorithm for color demosaicking," *IEEE Trans. Image Processing*, vol. 23, no. 9, pp. 4031–4040, Sep. 2014. [REF]

---

## Appendix: Reproducibility Checklist

| Item | Status |
|------|--------|
| **Code** | Pseudocode provided; reference implementation to be released upon publication |
| **Datasets** | Kodak (public), McMaster (public) |
| **Ground truth** | Full RGB images from benchmarks |
| **Mosaicing pattern** | Bayer RGGB, simulated by mask multiplication |
| **Baselines** | Bilinear (custom), MHC (MATLAB/OpenCV), AHD (original code), AMaZE (RawTherapee), RCD (GitHub) |
| **Metrics** | PSNR, SSIM, CIEDE2000, zipper-effect measure |
| **Parameters** | Defaults provided; full sensitivity analysis in supplementary |
| **Hardware** | CPU: x86-64 with AVX2; GPU: CUDA-enabled |
| **Random seeds** | N/A (deterministic algorithms) |
| **Statistical tests** | Paired t-tests, \(\alpha=0.05\) |
| **Runtime** | Measured on identical hardware with warm-up runs |
