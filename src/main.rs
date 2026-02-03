use tokio::sync::mpsc;
use tonic::{transport::Server, Request, Response, Status};

pub mod proto {
    tonic::include_proto!("gpu");
}

use proto::gpu_service_server::{GpuService, GpuServiceServer};
use proto::{PromptRequest, PromptResponse};

#[derive(Debug)]
struct Job {
    prompt: String,
}

#[derive(Debug)]
struct GpuWorker {
    tx: mpsc::Sender<Job>,
}

// #[tonic::async_trait]
//impl GpuService for GpuWorker {
//    async fn prompt(
//        &self,
//        request: Request<PromptRequest>,
//    ) -> Result<Response<PromptResponse>, Status> {
 //       let prompt = request.into_inner().prompt;

//        let job = Job { prompt };

  //      self.tx
    //        .send(job)
      //      .await
        //    .map_err(|_| Status::internal("Worker queue closed"))?;

        //Ok(Response::new(PromptResponse {
           // status: "queued".to_string(),
        //}))
   // }
// }
// updated code for prompt 
#[tonic::async_trait]
impl GpuService for GpuWorker {
    async fn prompt(
        &self,
        request: Request<PromptRequest>,
    ) -> Result<Response<PromptResponse>, Status> {
        let prompt = request.into_inner().prompt;

        let job = Job { prompt };

        self.tx
            .send(job)
            .await
            .map_err(|_| Status::internal("Worker queue closed"))?;

        Ok(Response::new(PromptResponse {
            
            response: "queued".to_string(), // ✅ Use 'response' here
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- job queue ----
    let (tx, mut rx) = mpsc::channel::<Job>(100);

    // ---- worker task ----
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            // Placeholder for diffusion + model inference
            println!("Processing prompt: {}", job.prompt);
        }
    });

    // ---- gRPC server ----
    let addr = "[::1]:50051".parse()?;
    let service = GpuWorker { tx };

    println!("GPU gRPC server listening on {}", addr);

    Server::builder()
        .add_service(GpuServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
