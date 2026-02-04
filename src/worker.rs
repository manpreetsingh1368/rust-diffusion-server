mod model;
mod diffusion;
mod dataset;

use tonic::{transport::Server, Request, Response, Status};
use tch::{nn, Device, Tensor, Kind, nn::VarStore};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::signal::unix::{signal, SignalKind};
use log::{info, error};
use env_logger;

use model::{MultiResUNet, inject_lora};
use diffusion::{generate_image, noise_schedule};
use dataset::DogDataset;

pub mod gpu_proto {
    tonic::include_proto!("gpu");
}

use gpu_proto::gpu_service_server::{GpuService, GpuServiceServer};
use gpu_proto::{PromptRequest, PromptResponse};

// Shared worker state
#[derive(Clone)]
struct WorkerServer {
    model: Arc<RwLock<MultiResUNet>>,
    device: Device,
}

#[tonic::async_trait]
impl GpuService for WorkerServer {
    async fn prompt(
        &self,
        request: Request<PromptRequest>,
    ) -> Result<Response<PromptResponse>, Status> {
        let prompt = request.into_inner().prompt;
        info!("Generating image for prompt: {}", prompt);

        let model_guard = self.model.read().await;
        let ctx = Tensor::zeros(&[1, 512], (Kind::Half, self.device));
        let (betas, alphas, alpha_bar) = noise_schedule(1000, self.device);

        let image = generate_image(&*model_guard, &ctx, &betas, &alphas, &alpha_bar);
        let fname = format!("samples/{}.png", prompt.replace(" ", "_"));
        tch::vision::image::save(&image, &fname).unwrap();

        Ok(Response::new(PromptResponse {
            response: format!("Generated: {}", fname),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    let device = Device::Cuda(0);

    // Initialize model and inject LoRA
    let vs = VarStore::new(device);
    let mut model = MultiResUNet::new(&vs.root(), 512);
    inject_lora(&mut model);

    let model = Arc::new(RwLock::new(model));

    // Load pre-trained model if exists
    if let Err(e) = load_model(&vs, &model).await {
        error!("Failed to load model: {}", e);
    } else {
        info!("Successfully loaded pre-trained model.");
    }

    // Start async training task
    let train_model = model.clone();
    tokio::spawn(async move {
        let dataset = DogDataset::new("data/images", "data/captions.json", 128);
        let batch_size = 4;
        let lr = 1e-4;

        let mut opt = nn::AdamW::default().build(&vs, lr).unwrap();

        for epoch in 0..50 {
            for (imgs, tokens) in dataset.iter_batch(batch_size) {
                let imgs = imgs.to_device(device).to_kind(Kind::Half);
                let tokens = tokens.to_device(device);

                let ctx = Tensor::zeros(&[batch_size, 32, 512], (Kind::Half, device));
                let (betas, alphas, alpha_bar) = noise_schedule(1000, device);

                let mut model_guard = train_model.write().await;
                let noisy = &imgs + Tensor::randn_like(&imgs) * 0.1;

                let pred = model_guard.forward(&noisy, &ctx);
                let loss = (pred - noisy).pow_tensor_scalar(2).mean(Kind::Float);

                opt.backward_step(&loss);
            }
            info!("Epoch {} finished", epoch);

            // Save model periodically
            if epoch % 10 == 0 {
                if let Err(e) = save_model(&vs, &train_model).await {
                    error!("Error saving model: {}", e);
                }
            }
        }
    });

    // Start gRPC server for inference
    let server = WorkerServer { model, device };

    // Graceful shutdown handling
    let mut sigterm = signal(SignalKind::terminate()).unwrap();
    let server_future = Server::builder()
        .add_service(GpuServiceServer::new(server))
        .serve("0.0.0.0:50052".parse()?);

    tokio::select! {
        _ = sigterm.recv() => {
            info!("Received SIGTERM, shutting down gracefully...");
        },
        result = server_future => {
            if let Err(e) = result {
                error!("Server failed: {}", e);
            }
        }
    }

    Ok(())
}

// Function to save the model
async fn save_model(vs: &VarStore, model: &Arc<RwLock<MultiResUNet>>) -> Result<(), Box<dyn std::error::Error>> {
    let model_guard = model.read().await;
    let model_path = "saved_model.ot";
    model_guard.save(vs, model_path)?;
    info!("Model saved at {}", model_path);
    Ok(())
}

// Function to load the model
async fn load_model(vs: &VarStore, model: &Arc<RwLock<MultiResUNet>>) -> Result<(), Box<dyn std::error::Error>> {
    let model_guard = model.write().await;
    let model_path = "saved_model.ot";
    if std::path::Path::new(model_path).exists() {
        model_guard.load(vs, model_path)?;
        info!("Model loaded from {}", model_path);
        Ok(())
    } else {
        Err("No pre-trained model found.".into())
    }
}
