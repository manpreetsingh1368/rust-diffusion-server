# 🚀 VAE-RUST

> **A Rust-powered gRPC image generation server**
>
> Turning prompts into images through a fast, safe, production-style backend.

---

## ✨ What Is This?

**VAE-RUST** is a **pure Rust gRPC server** that accepts text prompts and returns **generated images**.

It is built as an **AI inference service**, not a script and not a notebook — but a real backend you can run on a VPS, behind an API, or inside a larger system.

Think of it as:

> 🧠 *The infrastructure that turns image-generation models into a service.*

---

## 🖼️ What It Does (Right Now)

- Listens on **port 50051** via gRPC
- Accepts a text prompt from a client
- Generates an image inside the Rust server
- Returns the image as **PNG bytes**
- Responds with a clear execution status

End-to-end flow:

```
Client
  ↓ gRPC
Rust Server (VAE-RUST)
  ↓
Image Generation
  ↓
PNG Bytes
```

Even if the current image logic is simple or placeholder, the **full production pipeline is real and working**.

---

## ❌ What This Project Is NOT

- ❌ Not a model training framework
- ❌ Not a research playground
- ❌ Not a PyTorch / TensorFlow replacement
- ❌ Not focused on backpropagation or datasets

Training happens elsewhere.

**This project is about serving models reliably.**

---

## 🔥 Why This Is Interesting

Most AI demos stop at:

```bash
python generate.py
```

VAE-RUST answers the harder question:

> *How do I run image generation as a real service?*

That includes:
- Networking
- Concurrency
- Binary data handling
- API design
- Long-running processes
- Production-ready structure

This is the side of AI most tutorials never show.

---

## 🧩 How It Works

### gRPC API
Defined in `proto/gpu.proto`:

- **Package:** `gpu`
- **Service:** `GpuService`
- **Method:** `Prompt`
- **Request:** text prompt
- **Response:** image bytes (`PNG`) + status

### Server

- Written in **Rust**
- Uses `tonic` for gRPC
- Async execution via **Tokio**
- Designed for future GPU workers & batching

---

## ▶️ Running the Server

```bash
cargo run
```

You should see:
```
GPU gRPC server listening on [::1]:50051
```

---

## 🖼️ Example: Image Generation Code (Rust)

Below is a **simplified example** of the image-generation logic used by the server.
This code generates a 256×256 PNG image and returns it as bytes in the gRPC response.

```rust
use image::{RgbImage, ImageOutputFormat};
use rand::Rng;
use std::io::Cursor;

// Create an empty RGB image
let mut img = RgbImage::new(256, 256);
let mut rng = rand::thread_rng();

// Fill the image with random colors
for pixel in img.pixels_mut() {
    *pixel = image::Rgb([
        rng.gen(),
        rng.gen(),
        rng.gen(),
    ]);
}

// Encode the image as PNG into a byte buffer
let mut buffer = Vec::new();
img.write_to(&mut Cursor::new(&mut buffer), ImageOutputFormat::Png)?;

// Return image bytes via gRPC
Ok(Response::new(PromptResponse {
    image: buffer,
    status: "completed".to_string(),
}))
```

This example proves that:
- The server can generate images internally
- Binary image data can be returned over gRPC
- The pipeline is ready for real ML-based image generation

The random pixel logic can later be replaced with:
- Diffusion models
- VAEs
- Neural image generators

---
Here's how you can create a gRPC client to call the prompt and start_traning.

use tonic::transport::Channel;
use gpu_proto::gpu_service_client::GpuServiceClient;
use gpu_proto::{PromptRequest, TrainingRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = GpuServiceClient::connect("http://[::1]:50051").await?;

    // Inference request (image generation)
    let prompt_request = tonic::Request::new(PromptRequest {
        prompt: "A dog running in a park".into(),
    });

    let response = client.prompt(prompt_request).await?;
    println!("Response: {:?}", response.into_inner());

    // Training request
    let training_request = tonic::Request::new(TrainingRequest {
        batch_size: 4,
        epochs: 10,
    });

    let response = client.start_training(training_request).await?;
    println!("Response: {:?}", response.into_inner());

    Ok(())
}
Run the Server: First, run the gRPC server:

cargo run


Run the Client: In a separate terminal, run the client:

cargo run --bin client

Expected Workflow:

Start the Server: The server will listen for gRPC requests on port 50051.

Generate Image (Inference): The client sends a prompt to generate an image, and the server generates and saves the image to disk.

Start Training: The client sends a request to start training. The server will start the training process in the background using the specified batch size and number of epochs.
```bash
cargo run
```

You should see:
```
GPU gRPC server listening on [::1]:50051
```

---

## 📡 Calling the Server

Using `grpcurl`:

```bash
grpcurl -plaintext \
  -proto proto/gpu.proto \
  -d '{"prompt":"test"}' \
  localhost:50051 gpu.GpuService/Prompt
```

The response includes:
- Base64-encoded PNG image data
- A `status` field (`queued`, `completed`, etc.)

---

## 🛠️ What You Can Build With This

- Image generation API
- Backend for a web or mobile app
- Internal GPU inference service
- Foundation for diffusion / VAE servers
- Learning project for AI infrastructure in Rust

This is the same architectural pattern used by **real AI platforms**.

---

## 🦀 Why Rust?

Rust brings:

- Memory safety without garbage collection
- High-performance async networking
- Predictable latency
- Fearless concurrency

All of which are ideal for **AI inference backends**.

---

## 🧭 Future Direction

This project is intentionally extensible:

- Plug in real diffusion or VAE models
- Add GPU acceleration
- Implement request batching
- Introduce job queues
- Enable gRPC reflection (dev-only)
- Add authentication and rate limiting

The foundation is already here.

---

## ✅ Project Status

**Working and evolving.**

The server runs, accepts prompts, and returns generated images.

This repository represents a **solid base**, not a finished product.

---

## 📜 License

MIT
