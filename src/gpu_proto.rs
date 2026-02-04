#[tonic::async_trait]
pub trait GpuService: Send + Sync {
    async fn prompt(
        &self,
        request: tonic::Request<PromptRequest>,
    ) -> Result<tonic::Response<PromptResponse>, tonic::Status>;

    async fn start_training(
        &self,
        request: tonic::Request<TrainingRequest>, // This  match the gRPC method signature
    ) -> Result<tonic::Response<TrainingResponse>, tonic::Status>;
}
