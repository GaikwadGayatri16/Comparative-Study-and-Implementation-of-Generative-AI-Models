# Assignment–02: Comparative Study and Practical Analysis of Generative AI Models

**Course Outcomes:** CO3 and CO4  
**Assignment Type:** Group Assignment  
**Implementation Environment:** Google Colab, PyTorch and Hugging Face Diffusers

This project implements and compares four generative AI model families through image reconstruction and image generation experiments:

- **Autoencoder (AE)** — reconstructs input images through an encoder–decoder architecture.
- **Variational Autoencoder (VAE)** — learns a regularized probabilistic latent space for reconstruction and sampling.
- **Deep Convolutional Generative Adversarial Network (DCGAN)** — generates MNIST-like images using an adversarial generator–discriminator pair.
- **Stable Diffusion v1.5** — generates text-conditioned images using a pretrained latent diffusion pipeline.

The notebook includes model implementations, training, generated/reconstructed outputs, quantitative reconstruction evaluation, inference timings, comparisons and a conclusion based on the recorded run.

---

## Group Members

| Name | PRN | Documented Contribution |
|---|---|---|
| Suyash Hadole | 202401110055 | AE and VAE implementation, reconstruction analysis and MSE evaluation |
| Gayatri Gaikwad | 202502110002 | DCGAN implementation, training analysis and generated-sample evaluation |
| Om Channawar | 202401110044 | Stable Diffusion experiment, comparative analysis and report documentation |

---

## Objectives

1. Study the working principles and architectures of AE, VAE, DCGAN and Diffusion Models.
2. Reconstruct MNIST images using AE and VAE and compare reconstruction quality using visual inspection and Mean Squared Error (MSE).
3. Generate images using DCGAN and a pretrained Stable Diffusion model.
4. Compare the advantages, limitations, applications and outputs of the four model families.
5. Record experimental results, observations and group learning outcomes.

---

## Experimental Setup

| Parameter | Configuration |
|---|---|
| Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Diffusion Library | Hugging Face Diffusers |
| Dataset | MNIST Handwritten Digits |
| Training Subset | 10,000 images |
| Test Dataset | 10,000 images |
| Batch Size | 128 |
| Random Seed | 42 |
| AE Epochs | 6 |
| VAE Epochs | 6 |
| DCGAN Epochs | 8 |
| AE/VAE Latent Dimension | 16 |
| DCGAN Noise Dimension | 100 |
| Diffusion Model | Stable Diffusion v1.5 |
| Image Resolution | 512 × 512 |
| Denoising Steps | 12 |
| Guidance Scale | 7.5 |

AE and VAE use MNIST images normalized to `[0, 1]`, matching their Sigmoid decoder. DCGAN uses images normalized to `[-1, 1]`, matching its Tanh generator output.

Stable Diffusion is used for pretrained inference, not trained from scratch.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Google Colab
- Hugging Face Diffusers
- Stable Diffusion v1.5
- MNIST Dataset

---

## How to Run the Project

### Option 1: Google Colab (Recommended)

1. Download the Jupyter Notebook from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload `Assignment02_GenAI_Optimized.ipynb`.
4. Select **Runtime → Change runtime type → T4 GPU**.
5. Execute the notebook using **Runtime → Run all**.
6. Allow the MNIST dataset and Stable Diffusion model to download.
7. Verify that all four experiments execute successfully.
8. Check the generated figures and `results.json` inside the `outputs/` directory.

### Option 2: Local Environment

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib diffusers transformers accelerate safetensors
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the assignment notebook and execute the cells sequentially.

**Note:** A CUDA-capable NVIDIA GPU is recommended for Stable Diffusion. Internet access is required for first-time dataset and pretrained model downloads.

---

## Model Implementations

### 1. Autoencoder (AE)

- Uses an encoder and decoder architecture.
- Compresses input images into a 16-dimensional latent representation.
- Reconstructs images using the decoder.
- Evaluated using Mean Squared Error (MSE).

### 2. Variational Autoencoder (VAE)

- Learns the mean and log-variance of a latent distribution.
- Uses the reparameterization trick.
- Combines reconstruction loss and KL divergence.
- Generates new samples using random latent vectors.

### 3. Deep Convolutional GAN (DCGAN)

- Consists of a generator and discriminator.
- Generator creates images from random noise.
- Discriminator distinguishes real images from generated images.
- Trained using adversarial learning.

### 4. Stable Diffusion v1.5

- Uses a pretrained latent diffusion architecture.
- Generates images from text prompts.
- Uses a text encoder, U-Net denoiser, scheduler and VAE decoder.
- Produces 512 × 512 images using 12 denoising steps.

---

## Experimental Results

The following values are taken from the executed notebook. They represent this particular experimental run and are not universal performance benchmarks.

| Experiment | Recorded Result |
|---|---:|
| AE Training Time | 2.22 seconds |
| VAE Training Time | 2.29 seconds |
| AE Test MSE | **0.03215** |
| VAE Test MSE | **0.03984** |
| DCGAN Training Time | 12.55 seconds |
| DCGAN Generation Time | 2.32 ms for 16 images |
| Stable Diffusion Generated Images | 3 |
| Stable Diffusion Mean Generation Time | **3.17 seconds per image** |

### Stable Diffusion Generation Timings

| Prompt | Generation Time |
|---|---:|
| Handwritten Digit | 3.96 seconds |
| Photorealistic Portrait | 2.77 seconds |
| Red Sports Car | 2.78 seconds |

### Results Analysis

- **Autoencoder:** Achieved a lower reconstruction MSE than VAE in this experimental run.
- **Variational Autoencoder:** Produced reconstructed images and demonstrated probabilistic latent-space sampling.
- **DCGAN:** Generated MNIST-like images, although some samples contained distorted or ambiguous digits due to limited training.
- **Stable Diffusion:** Generated detailed prompt-conditioned images. The portrait and sports-car prompts were represented clearly, while the handwritten-digit prompt did not accurately produce the requested digit 7.

Reconstruction MSE is suitable for comparing AE and VAE, but it does not directly measure realism, diversity or prompt adherence in generated images.

---

## Output Files

The notebook automatically creates an `outputs/` directory containing:

| File Name | Description |
|---|---|
| `fig_training_losses.png` | AE and VAE training losses |
| `fig_reconstruction_ae_vae.png` | Original and reconstructed MNIST images |
| `fig_mse_comparison.png` | AE and VAE reconstruction MSE |
| `fig_latent_samples.png` | VAE and AE latent-space samples |
| `fig_dcgan_training_progress.png` | DCGAN training progress |
| `fig_dcgan_final_samples.png` | Final DCGAN generated images |
| `fig_dcgan_losses.png` | Generator and discriminator losses |
| `fig_stable_diffusion.png` | Stable Diffusion generated images |
| `results.json` | Experimental configuration and measured results |

---

## Conclusion

This assignment demonstrates the practical implementation and comparative analysis of four generative AI model families.

Autoencoder and Variational Autoencoder were used for image reconstruction, while DCGAN and Stable Diffusion were used for image generation.

In the recorded experiment, AE achieved a lower reconstruction MSE than VAE. VAE demonstrated the advantage of a structured latent space, while DCGAN generated dataset-specific handwritten-digit samples. Stable Diffusion produced detailed text-conditioned images with higher computational requirements.

The experiments demonstrate that different generative models have different architectures, objectives, strengths and limitations. Their performance depends on the dataset, model configuration, training duration and computational resources.

---

## References

1. G. E. Hinton and R. R. Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks," Science, 2006. [Research Paper](https://doi.org/10.1126/science.1127647)

2. D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," ICLR, 2014. [Research Paper](https://arxiv.org/abs/1312.6114)

3. I. Goodfellow et al., "Generative Adversarial Nets," NeurIPS, 2014. [Research Paper](https://arxiv.org/abs/1406.2661)

4. A. Radford, L. Metz and S. Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," 2015. [Research Paper](https://arxiv.org/abs/1511.06434)

5. Y. LeCun, C. Cortes and C. J. C. Burges, "The MNIST Database of Handwritten Digits." [Dataset](http://yann.lecun.com/exdb/mnist/)

6. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR, 2022. [Research Paper](https://arxiv.org/abs/2112.10752)

7. Hugging Face. [Diffusers Documentation](https://huggingface.co/docs/diffusers/)

8. Hugging Face. [Stable Diffusion v1.5 Model Card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

---

