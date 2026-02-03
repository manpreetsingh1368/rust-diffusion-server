use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};
use serde::Deserialize;
use tokio::sync::mpsc::{Sender, Receiver, channel};  // NEW: Async queue for tasks


#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

#[post("/generate")]
async fn generate_prompt(req: web::Json<PromptRequest>) -> impl Responder {
    let prompt = &req.prompt;

    // Call GPU worker via gRPC
    let worker_url = "http://127.0.0.1:50052"; // worker gRPC endpoint
    // Example with REST for simplicity:
    // let client = reqwest::Client::new();
    // let res = client.post(worker_url).json(&req.0).send().await;

    // TODO: implement gRPC client call to WorkerServer.send_prompt(prompt)
    // Here we just return the URL placeholder
    let image_url = format!("https://myserver.com/samples/{}.png", prompt.replace(" ", "_"));

    HttpResponse::Ok().json(serde_json::json!({
        "prompt": prompt,
        "image_url": image_url
    }))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(generate_prompt)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
