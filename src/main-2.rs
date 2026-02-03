

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use image::io::Reader as ImageReader;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::vision::image::save;
use rust_bert::pipelines::clip::{CLIPModel, CLIPTokenizer};

// =========================
// CONFIG
// =========================
const IMG_SIZE: i64 = 128;
const BATCH_SIZE: usize = 4;
const EPOCHS: usize = 50;
const LR: f64 = 1e-4;
const T: usize = 1000;
const SAVE_EVERY: usize = 500;
const GUIDANCE_SCALE: f64 = 7.5;

const SAMPLES_DIR: &str = "samples";
const IMAGES_DIR: &str = "data/images";
const CAPTIONS_FILE: &str = "data/captions.json";

// =========================
// SETUP DIRS + CAPTIONS
// =========================
fn setup_dirs() {
    fs::create_dir_all(SAMPLES_DIR).unwrap();
    fs::create_dir_all(IMAGES_DIR).unwrap();
}

fn generate_captions(img_dir: &str, captions_file: &str) {
    if !Path::new(captions_file).exists() {
        let mut captions: HashMap<String, String> = HashMap::new();
        for entry in fs::read_dir(img_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "png" || ext == "jpg" || ext == "jpeg" {
                    if let Some(fname) = path.file_name() {
                        captions.insert(fname.to_string_lossy().to_string(), "A dog".to_string());
                    }
                }
            }
        }
        let file = fs::File::create(captions_file).unwrap();
        serde_json::to_writer_pretty(file, &captions).unwrap();
        println!("Generated captions for {} images.", captions.len());
    } else {
        println!("Captions file exists, using it.");
    }
}

// =========================
// DATASET
// =========================
pub struct DogDataset {
    keys: Vec<String>,
    captions: HashMap<String, String>,
    img_dir: String,
}

impl DogDataset {
    pub fn new(img_dir: &str, captions_file: &str) -> Self {
        let file_content = fs::read_to_string(captions_file).unwrap();
        let captions: HashMap<String, String> = serde_json::from_str(&file_content).unwrap();
        let keys = captions.keys().cloned().collect();
        DogDataset { keys, captions, img_dir: img_dir.to_string() }
    }

    pub fn len(&self) -> usize { self.keys.len() }

    pub fn get(&self, idx: usize) -> (Tensor, String) {
        let fname = &self.keys[idx];
        let path = Path::new(&self.img_dir).join(fname);

        let img = ImageReader::open(path).unwrap()
            .decode().unwrap()
            .resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, image::imageops::FilterType::Nearest)
            .to_rgb8();

        let img_tensor = Tensor::of_data_size(&img.as_raw(), &[IMG_SIZE, IMG_SIZE, 3], Kind::Uint8)
            .to_kind(Kind::Float) / 255.0;
        let img_tensor = img_tensor.permute(&[2, 0, 1]);

        let caption = self.captions[fname].clone();

        (img_tensor, caption)
    }
}

// =========================
// DATA LOADER
// =========================
pub struct DataLoader<'a> {
    dataset: &'a DogDataset,
    batch_size: usize,
    indices: Vec<usize>,
    current: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a DogDataset, batch_size: usize) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        indices.shuffle(&mut rand::thread_rng());
        DataLoader { dataset, batch_size, indices, current: 0 }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        self.indices.shuffle(&mut rand::thread_rng());
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Tensor, Vec<String>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() { return None; }
        let end = (self.current + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current..end];

        let mut imgs = Vec::new();
        let mut captions = Vec::new();
        for &i in batch_indices {
            let (img, cap) = self.dataset.get(i);
            imgs.push(img);
            captions.push(cap);
        }
        self.current = end;
        Some((Tensor::stack(&imgs, 0), captions))
    }
}

// =========================
// CROSS ATTENTION
// =========================
pub struct CrossAttention {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    out: nn::Linear,
    heads: i64,
    scale: f64,
}

impl CrossAttention {
    pub fn new(vs: &nn::Path, dim: i64, ctx_dim: i64, heads: i64) -> Self {
        let q = nn::linear(vs / "q", dim, dim, Default::default());
        let k = nn::linear(vs / "k", ctx_dim, dim, Default::default());
        let v = nn::linear(vs / "v", ctx_dim, dim, Default::default());
        let out = nn::linear(vs / "out", dim, dim, Default::default());
        CrossAttention { q, k, v, out, heads, scale: (dim / heads) as f64 }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let (b, n, c) = x.size3().unwrap();
        let h = self.heads;
        let c_h = c / h;
        let q = self.q.forward(x).view([b, n, h, c_h]);
        let k = self.k.forward(ctx).view([b, -1, h, c_h]);
        let v = self.v.forward(ctx).view([b, -1, h, c_h]);

        let attn = q.unsqueeze(3).matmul(&k.unsqueeze(2)) / self.scale;
        let attn = attn.softmax(-1, Kind::Float);
        let out = attn.matmul(&v.unsqueeze(2)).view([b, n, c]);
        self.out.forward(&out)
    }
}

// =========================
// UNET BLOCK
// =========================
pub struct UNetBlock {
    conv: nn::Conv2D,
    attn: CrossAttention,
}

impl UNetBlock {
    pub fn new(vs: &nn::Path, in_ch: i64, out_ch: i64, ctx_dim: i64) -> Self {
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv = nn::conv2d(vs / "conv", in_ch, out_ch, 3, conv_cfg);
        let attn = CrossAttention::new(&(vs / "attn"), out_ch, ctx_dim, 4);
        UNetBlock { conv, attn }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let x = self.conv.forward(x);
        let (b, c, h, w) = x.size4().unwrap();
        let mut x_flat = x.view([b, c, h * w]).permute(&[0, 2, 1]);
        x_flat = self.attn.forward(&x_flat, ctx);
        x_flat.permute(&[0, 2, 1]).view([b, c, h, w])
    }
}

// =========================
// MULTI-RES UNET
// =========================
pub struct MultiResUNet {
    down1: UNetBlock,
    down2: UNetBlock,
    mid: UNetBlock,
    up1: nn::ConvTranspose2D,
    up2: nn::Conv2D,
}

impl MultiResUNet {
    pub fn new(vs: &nn::Path, ctx_dim: i64) -> Self {
        let down1 = UNetBlock::new(&(vs / "down1"), 3, 64, ctx_dim);
        let down2 = UNetBlock::new(&(vs / "down2"), 64, 128, ctx_dim);
        let mid = UNetBlock::new(&(vs / "mid"), 128, 128, ctx_dim);
        let up1 = nn::conv_transpose2d(&(vs / "up1"), 128, 64, 2, Default::default());
        let up2 = nn::conv2d(&(vs / "up2"), 64, 3, 3, nn::ConvConfig { padding: 1, ..Default::default() });
        MultiResUNet { down1, down2, mid, up1, up2 }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let d1 = self.down1.forward(x, ctx);
        let d2 = self.down2.forward(&d1, ctx);
        let m = self.mid.forward(&d2, ctx);
        let mut u = self.up1.forward(&m);
        if u.size()[2..] != d1.size()[2..] {
            let size = &[d1.size()[2], d1.size()[3]];
            u = u.upsample_bilinear2d(size, false, None, None);
        }
        self.up2.forward(&(u + d1))
    }
}

// =========================
// MAIN
// =========================
fn main() {
    setup_dirs();
    generate_captions(IMAGES_DIR, CAPTIONS_FILE);

    let device = Device::cuda_if_available();

    // Load CLIP
    let mut clip_model = CLIPModel::new(Default::default()).unwrap();
    clip_model.model.set_device(device);

    // Dataset & loader
    let dataset = DogDataset::new(IMAGES_DIR, CAPTIONS_FILE);
    let mut loader = DataLoader::new(&dataset, BATCH_SIZE);

    // Model
    let vs = nn::VarStore::new(device);
    let model = MultiResUNet::new(&vs.root(), 512);
    let mut opt = nn::Adam::default().build(&vs, LR).unwrap();

    // Diffusion schedule
    let betas = Tensor::linspace(1e-4, 0.02, T as i64, (Kind::Float, device));
    let alphas = 1.0 - &betas;
    let alpha_bar = alphas.cumprod(0, Kind::Float);

    let mut step = 0;
    for epoch in 0..EPOCHS {
        loader.reset();
        for (imgs, captions) in loader.by_ref() {
            let imgs = imgs.to_device(device);

            // Encode text using CLIP
            let ctx = clip_model.encode_text(captions.clone()).unwrap();

            // Sample timestep & noise
            let t = Tensor::randint(0, T as i64, &[imgs.size()[0]], (Kind::Int64, device));
            let noise = Tensor::randn_like(&imgs);
            let noisy = alpha_bar.i((t,)).sqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1) * &imgs
                + (1.0 - alpha_bar.i((t,))).sqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1) * &noise;

            let pred = model.forward(&noisy, &ctx);
            let loss = (pred - noise).pow(2).mean(Kind::Float);

            opt.backward_step(&loss);

            if step % SAVE_EVERY == 0 {
                let img_save = pred.clamp(0.0, 1.0).get(0);
                save(&img_save, format!("{}/step_{}.png", SAMPLES_DIR, step)).unwrap();
            }

            step += 1;
            println!("Epoch {}/{} | Step {} | Loss {:.4}", epoch + 1, EPOCHS, step, f64::from(&loss));
        }
    }

    println!("Training finished!");
}
