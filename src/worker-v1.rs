mod model;
mod diffusion;
mod dataset;
use tonic::{transport::Server, Request, Response, Status};
use tch::{nn, Device, VarStore, Tensor, Kind};
use std::sync::Arc;
use model::MultiResUNet;
use diffusion::{generate_image, noise_schedule};

pub mod gpu_proto {
    tonic::include_proto!("gpu");
}

use gpu_proto::gpu_service_server::{GpuService, GpuServiceServer};
use gpu_proto::{PromptRequest, PromptResponse};

#[derive(Clone)]
struct WorkerServer {
    model: Arc<MultiResUNet>,
}

#[tonic::async_trait]
impl GpuService for WorkerServer {
    async fn send_prompt(
        &self,
        request: Request<PromptRequest>,
    ) -> Result<Response<PromptResponse>, Status> {
        let prompt = request.into_inner().prompt;
        println!("Generating image for prompt: {}", prompt);

        let ctx = Tensor::zeros(&[1,512], (Kind::Float, Device::Cuda(0)));
        let (betas, alphas, alpha_bar) = noise_schedule(1000, Device::Cuda(0));

        let image = generate_image(&self.model, &ctx, &betas, &alphas, &alpha_bar);
        let fname = format!("samples/{}.png", prompt.replace(" ", "_"));
        tch::vision::image::save(&image, &fname).unwrap();

        Ok(Response::new(PromptResponse {
            status: format!("Generated: {}", fname),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vs = VarStore::new(Device::Cuda(0));
    let model = Arc::new(MultiResUNet::new(&vs.root(), 512));

    let server = WorkerServer { model };
    Server::builder()
        .add_service(GpuServiceServer::new(server))
        .serve("0.0.0.0:50052".parse()?)
        .await?;

    Ok(())
}
