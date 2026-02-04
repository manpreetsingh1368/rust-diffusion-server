use tch::{nn, nn::ModuleT, Tensor, Kind, Device};

pub struct CrossAttention {
    pub q: nn::Linear,
    pub k: nn::Linear,
    pub v: nn::Linear,
    pub out: nn::Linear,
    pub heads: i64,
    pub scale: f64,
}

impl CrossAttention {
    pub fn new(vs: &nn::Path, dim: i64, ctx_dim: i64, heads: i64) -> Self {
        let q = nn::linear(vs, dim, dim, Default::default());
        let k = nn::linear(vs, ctx_dim, dim, Default::default());
        let v = nn::linear(vs, ctx_dim, dim, Default::default());
        let out = nn::linear(vs, dim, dim, Default::default());
        Self { q, k, v, out, heads, scale: 1.0 / ((dim / heads) as f64).sqrt() }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let (b, n, c) = x.size3().unwrap();
        let h = self.heads;
        let q = self.q.forward(x).view([b, n, h, c / h]);
        let k = self.k.forward(ctx).view([b, -1, h, c / h]);
        let v = self.v.forward(ctx).view([b, -1, h, c / h]);
        let attn = q.matmul(&k.transpose(-2, -1)) * self.scale;
        let attn = attn.softmax(-1, Kind::Float);
        let out = attn.matmul(&v);
        self.out.forward(&out.view([b, n, c]))
    }
}

pub struct UNetBlock {
    pub conv1: nn::Conv2D,
    pub conv2: nn::Conv2D,
    pub attn: CrossAttention,
}

impl UNetBlock {
    pub fn new(vs: &nn::Path, in_ch: i64, out_ch: i64, ctx_dim: i64) -> Self {
        let conv1 = nn::conv2d(vs, in_ch, out_ch, 3, Default::default());
        let conv2 = nn::conv2d(vs, out_ch, out_ch, 3, Default::default());
        let attn = CrossAttention::new(&(vs / "attn"), out_ch, ctx_dim, 8);
        Self { conv1, conv2, attn }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let mut x = x.apply(&self.conv1).relu();
        x = x.apply(&self.conv2).relu();
        let (b, c, h, w) = x.size4().unwrap();
        let x_flat = x.view([b, c, h*w]).transpose(1, 2);
        let x_attn = self.attn.forward(&x_flat, ctx);
        x_attn.transpose(1,2).view([b,c,h,w])
    }
}

pub struct MultiResUNet {
    pub down1: UNetBlock,
    pub down2: UNetBlock,
    pub mid: UNetBlock,
    pub up1: nn::ConvTranspose2D,
    pub up2: nn::Conv2D,
}

impl MultiResUNet {
    pub fn new(vs: &nn::Path, ctx_dim: i64) -> Self {
        let down1 = UNetBlock::new(&(vs / "down1"), 3, 128, ctx_dim);
        let down2 = UNetBlock::new(&(vs / "down2"), 128, 256, ctx_dim);
        let mid = UNetBlock::new(&(vs / "mid"), 256, 256, ctx_dim);
        let up1 = nn::conv_transpose2d(&(vs / "up1"), 256, 128, 2, Default::default());
        let up2 = nn::conv2d(&(vs / "up2"), 128, 3, 3, Default::default());
        Self { down1, down2, mid, up1, up2 }
    }

    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Tensor {
        let d1 = self.down1.forward(x, ctx);
        let d2 = self.down2.forward(&d1.max_pool2d_default(2), ctx);
        let m = self.mid.forward(&d2.max_pool2d_default(2), ctx);
        let u = self.up1.forward(&m) + d2;
        self.up2.forward(&(u + d1))
    }
}
