use tch::{Tensor, Device, Kind};
use crate::model::MultiResUNet;

pub fn noise_schedule(t: i64, device: Device) -> (Tensor, Tensor, Tensor) {
    let timesteps = Tensor::arange(0, t, (Kind::Int64, device)); // Create timesteps
    let min_beta = 1e-4;
    let max_beta = 0.02;

    // Cosine beta schedule
    let cos_betas = (timesteps.clone() / (t - 1) as f64).cos();
    let betas = min_beta + (max_beta - min_beta) * (cos_betas - 1.0).abs();

    let alphas = 1.0 - &betas;
    let alpha_bar = alphas.cumprod(0, Kind::Float);

    (betas, alphas, alpa_bar)
}

pub fn dynamic_guidance(t: i64, max_guidance: f64, min_guidance: f64, total_timesteps: i64) -> f64 {
    let progress = (t as f64) / (total_timesteps as f64);
    let scale = max_guidance - (max_guidance - min_guidance) * progress;
    scale
}

pub fn generate_image(
    model: &MultiResUNet,
    ctx: &Tensor,
    betas: &Tensor,
    alphas: &Tensor,
    alpha_bar: &Tensor,
    guidance_scale: f64,
    text_embeddings: Option<&Tensor>
) -> Tensor {
    // Initialize with random noise
    let mut x = Tensor::randn(&[1, 3, 128, 128], (Kind::Float, ctx.device()));
    let uctx = Tensor::zeros_like(ctx); // Use empty context for baseline
    let guidance = guidance_scale; // Fine-tune the guidance scale

    // For conditional generation (e.g., text-to-image), use the embeddings or context if provided
    let conditioned_x = if let Some(embeddings) = text_embeddings {
        model.apply_conditioning(&x, embeddings)
    } else {
        x // Use noise as the baseline if no condition is provided
    };

    // Diffusion process
    for t in (0..1000).rev() { // Reverse diffusion
        let eps_c = model.forward(&conditioned_x, ctx); // Forward pass with context
        let eps_u = model.forward(&conditioned_x, &uctx); // Forward pass with unconditioned context
        let eps = &eps_u + guidance * (&eps_c - &eps_u); // Apply guidance

        x = (&x - &betas.get(t) * eps) / alphas.get(t).sqrt(); // Update the image
    }

    x.clamp(0.0, 1.0) // Ensure pixel values are between 0 and 1
}

impl MultiResUNet {
    pub fn apply_conditioning(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        // Example: Concatenate the image (x) with the conditioning (ctx)
        let (b, c, h, w) = x.size4().unwrap();
        let conditioned_input = x.cat(&[ctx.view([b, c, 1, 1]); ctx.size(0)], 1); // Concatenate across channels
        self.forward(&conditioned_input, ctx) // Forward pass with the conditioned input
    }
}
