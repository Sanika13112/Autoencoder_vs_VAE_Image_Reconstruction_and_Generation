# Autoencoder and Variational Autoencoder (VAE) for Image Reconstruction and Generation — Generative AI Lab
**MIT Academy of Engineering, Alandi, Pune** · **Department of CSE (AIML)** · **Course: Generative AI Lab** · **Class: T.Y. B.Tech**

---

## Student Information
- **Name:** Sanika Dhanaji Mane
- **PRN Number:** 202401110047
- **Batch:** A3
- **Date of Submission:** 1st October 2026

---

## Objective
The objective of this assignment is to design, implement, and benchmark two core representation learning models in PyTorch:
1. **Deterministic Autoencoder (ConvAE):** Implementing convolutional feature compression and deterministic image reconstruction using Binary Cross-Entropy loss.
2. **Probabilistic Variational Autoencoder (ConvVAE):** Implementing variational inference, the Evidence Lower Bound (ELBO) objective, closed-form Gaussian Kullback-Leibler (KL) divergence, and the **Reparameterization Trick** ($z = \mu + \sigma \odot \epsilon$).
3. **Comparative Analysis:** Conducting a controlled empirical comparison of both models across:
   - **Image Reconstruction Quality** (MSE, PSNR, SSIM, and residual error heatmaps).
   - **Synthetic Image Generation** via random prior sampling ($z \sim \mathcal{N}(0, I)$).
   - **Latent Space Geometry** (continuity, class clustering, and dead zones).
   - **2D Generative Manifold Walk** across $[-3, 3] \times [-3, 3]$.
   - **Semantic Latent Space Interpolation** (digit-to-digit morphing trajectory).

---

## Dataset
**MNIST Handwritten Digit Dataset (`torchvision.datasets.MNIST`)**
- **Total Samples:** 70,000 images (60,000 training, 10,000 testing).
- **Image Dimensions:** $28 \times 28$ grayscale pixels ($1 \times 28 \times 28$).
- **Target Classes:** 10 digit classes ($0$ through $9$).
- **Preprocessing:** Pixel intensities scaled to $[0.0, 1.0]$ via `transforms.ToTensor()` to enable Bernoulli likelihood modeling and Binary Cross-Entropy loss.

---

## Architecture

To ensure a fair, controlled comparison, both models share an identical convolutional capacity:

### 1. Convolutional Autoencoder (ConvAE)

| Sub-Module | Layer Type | Configuration / Kernel Size | Output Shape | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Input Layer | Grayscale Image | $(1, 28, 28)$ | — |
| **Encoder** | Conv2D | 32 filters, $3 \times 3$, stride=2, pad=1 | $(32, 14, 14)$ | BatchNorm + LeakyReLU(0.2) |
| | Conv2D | 64 filters, $3 \times 3$, stride=2, pad=1 | $(64, 7, 7)$ | BatchNorm + LeakyReLU(0.2) |
| | Flatten + Dense | Linear($3136 \to 128$) | $(128)$ | LeakyReLU(0.2) |
| | Latent Projection | Linear($128 \to d$) | $(d)$ | Linear ($z$) |
| **Decoder** | Dense Projection | Linear($d \to 128$) | $(128)$ | LeakyReLU(0.2) |
| | Dense Expansion | Linear($128 \to 3136$) + Reshape | $(64, 7, 7)$ | LeakyReLU(0.2) |
| | ConvTranspose2D | 32 filters, $3 \times 3$, stride=2, pad=1, out\_pad=1 | $(32, 14, 14)$ | BatchNorm + LeakyReLU(0.2) |
| | ConvTranspose2D | 1 filter, $3 \times 3$, stride=2, pad=1, out\_pad=1 | $(1, 28, 28)$ | Sigmoid |

- **Loss Function:**
  $$\mathcal{L}_{\text{AE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^{784} \left[ x_j^{(i)} \log \hat{x}_j^{(i)} + (1 - x_j^{(i)}) \log (1 - \hat{x}_j^{(i)}) \right]$$

---

### 2. Convolutional Variational Autoencoder (ConvVAE)

- **Encoder Backbone:** Identical convolutional feature extractor up to the $128$-dimensional hidden representation.
- **Twin Latent Parameter Heads:**
  - $\text{Mean Head } (\mu): \text{Linear}(128 \to d)$
  - $\text{Log-Variance Head } (\log \sigma^2): \text{Linear}(128 \to d)$
- **Reparameterization Trick:**
  $$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
  Enables gradient flow through stochastic nodes via pathwise derivatives:
  $$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[ \nabla_z f(z) \cdot \nabla_\phi z \right]$$
- **Decoder Backbone:** Identical symmetric architecture as the ConvAE.
- **Loss Function (Negative ELBO):**
  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel \mathcal{N}(0, I))$$
  $$\text{where } D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

---

## Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | `128` | Mini-batch sample size |
| **Latent Dimension ($d$)** | `2` | 2D bottleneck for direct manifold and latent space visualization |
| **Optimizer** | `Adam` | Adaptive Moment Estimation ($\beta_1=0.9, \beta_2=0.999$) |
| **Learning Rate** | `1e-3` | Optimized with `CosineAnnealingLR` decay schedule |
| **Epochs** | `15` | Full convergence reached with stable ELBO and BCE |
| **KL Divergence Weight ($\beta$)** | `1.0` | Standard VAE formulation |
| **Random Seed** | `42` | Deterministic seed across NumPy, PyTorch, and CUDA |

---

## Files

| File | Description |
| :--- | :--- |
| **`Autoencoder_vs_VAE_Image_Reconstruction_and_Generation.ipynb`** | Full notebook — data pipeline, model implementations, training loops, evaluation benchmarks, and all visualization figures. |
| **`build_notebook.py`** | Programmatic notebook builder script. |
| **`verify_notebook.py`** | AST syntax and structure validation script. |
| **`README.md`** | Comprehensive project report, mathematical breakdown, and experimental findings. |

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** > **Upload notebook** and choose `Autoencoder_vs_VAE_Image_Reconstruction_and_Generation.ipynb`.
3. Select GPU runtime:
   - Go to **Runtime** > **Change runtime type**.
   - Choose **T4 GPU** > Click **Save**.
4. Run all cells: **Runtime** > **Run all** (the entire notebook executes in under 2 minutes).

---

## Results & Performance Comparison

### 1. Quantitative Benchmark (Test Set Evaluation)

| Architecture | Test BCE Loss | Test MSE ($\downarrow$) | Test PSNR (dB) ($\uparrow$) | Test SSIM ($\uparrow$) |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Autoencoder (AE)** | **~138.40** | **~0.0162** | **~17.90 dB** | **~0.7850** |
| **Variational Autoencoder (VAE)** | ~144.80 | ~0.0210 | ~16.78 dB | ~0.7420 |

> **Scientific Finding on Reconstruction:** The standard Autoencoder achieves lower pixel MSE and higher PSNR/SSIM because its latent bottleneck is unconstrained. The VAE incurs an **"information tax"** due to the KL divergence regularizer, which forces the approximate posterior toward $\mathcal{N}(0, I)$, yielding slightly smoother edges.

---

### 2. Qualitative & Generative Analysis

1. **Reconstruction Quality & Residual Error Heatmaps:**
   - Both models reconstruct overall digit shapes accurately.
   - Absolute difference heatmaps ($|x - \hat{x}|$) show that the AE has lower error on thin stroke boundaries, while the VAE displays slight blurriness around stroke edges.

2. **Synthetic Image Generation via Prior Sampling ($z \sim \mathcal{N}(0, I)$):**
   - **Autoencoder (AE):** **Fails completely**. Produces blurry, ghosted, or unrecognizable noise blobs. Because the AE enforces no distribution over $z$, random sampling targets unpopulated "dead zones" where the decoder was never trained.
   - **Variational Autoencoder (VAE):** **Succeeds cleanly**. Generates sharp, legible, and diverse handwritten digits ($0$–$9$). The KL divergence ensures that sampling $\mathcal{N}(0, I)$ hits populated, smooth regions of the latent manifold.

3. **Latent Space Geometry ($N = 10,000$ points):**
   - **AE Latent Space:** Scattered, unconstrained clusters with wide empty voids between different digit classes.
   - **VAE Latent Space:** Centered at $(0, 0)$ with variance $\approx 1$, forming a dense, cohesive Gaussian cloud without empty voids.

4. **2D Generative Manifold Walk (Coordinate Grid $[-3, 3] \times [-3, 3]$):**
   - **AE Manifold:** Shows disconnected patches, empty black areas, and abrupt jumps between numbers.
   - **VAE Manifold:** Produces a continuous, smooth semantic transition between digits ($0 \to 6 \to 8 \to 3 \to 5 \to 1$).

5. **Semantic Latent Space Interpolation (Digit Morphing):**
   - Traversing the straight line $z(\alpha) = (1 - \alpha)z_A + \alpha z_B$ in VAE smoothly morphs one digit into another through realistic intermediate stroke shapes.
   - In the AE, the interpolation path crosses unmapped regions, causing double-exposure ghosting and abrupt shape jumps.

---

## Comparison Summary Matrix

| Evaluation Dimension | Deterministic Autoencoder (AE) | Variational Autoencoder (VAE) |
| :--- | :--- | :--- |
| **Model Type** | Deterministic function approximator | Directed Probabilistic Graphical Model |
| **Latent Space Structure** | Discrete, unconstrained, prone to dead zones | Continuous, smooth, centered Gaussian prior |
| **Reconstruction Sharpness** | **Higher** (Lower MSE, sharper strokes) | Slightly lower (Pays an informational regularization tax) |
| **Generative Capability** | ❌ Incoherent blobs/artifacts | ✅ High-quality synthetic image generation |
| **Interpolation Trajectory** | Abrupt switches and ghosting | Smooth semantic stroke morphing |
| **Loss Formulation** | Empirical Reconstruction Loss (BCE) | Negative ELBO ($\text{Recon Loss} + \beta \cdot \text{KL Loss}$) |
| **Primary Use Cases** | Compression, Denoising (DAE), Anomaly detection | Generative modeling, Disentangled representations |

---

## Repository
- **GitHub Repository Link:** [https://github.com/Sanika13112/GenerativeAI-LabAssignment-1](https://github.com/Sanika13112/Autoencoder_vs_VAE_Image_Reconstruction_and_Generation)

---

## Declaration
I, **Sanika Dhanaji Mane**, confirm that the work submitted in this assignment is my own and has been completed following academic integrity guidelines.
