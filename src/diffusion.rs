use tch::{Tensor, Device, Kind};
use crate::model::MultiResUNet;

pub fn noise_schedule(t: i64, device: Device) -> (Tensor, Tensor, Tensor) {
    let betas = Tensor::linspace(1e-4, 0.02, t, (Kind::Float, device));
    let alphas = 1.0 - &betas;
    let alpha_bar = alphas.cumprod(0, Kind::Float);
    (betas, alphas, alpha_bar)
}

pub fn generate_image(model: &MultiResUNet, ctx: &Tensor, betas: &Tensor, alphas: &Tensor, alpha_bar: &Tensor) -> Tensor {
    let mut x = Tensor::randn(&[1,3,128,128], (Kind::Float, ctx.device()));
    let uctx = Tensor::zeros_like(ctx);
    let guidance = 7.5;

    for t in (0..1000).rev() {
        let eps_c = model.forward(&x, ctx);
        let eps_u = model.forward(&x, &uctx);
        let eps = &eps_u + guidance * (&eps_c - &eps_u);
        x = (&x - &betas.get(t) * eps) / alphas.get(t).sqrt();
    }
    x.clamp(0.0,1.0)
}
