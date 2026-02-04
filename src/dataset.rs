use tch::{Tensor, Device, Kind};
use std::path::Path;
use std::fs;

pub struct DogDataset {
    image_paths: Vec<String>,   // Placeholder for image file paths
    caption_paths: Vec<String>, // Placeholder for caption file paths
    batch_size: usize,
}

impl DogDataset {
    // Constructor to initialize the dataset with paths to images and captions
    pub fn new(image_dir: &str, caption_file: &str, batch_size: usize) -> Self {
        let image_paths = fs::read_dir(image_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().map(|ext| ext == "jpg" || ext == "png").unwrap_or(false))
            .map(|entry| entry.path().to_str().unwrap().to_string())
            .collect::<Vec<String>>();

        // Just a placeholder for captions, assuming each image has a corresponding caption
        let caption_paths = image_paths.iter().map(|_| "A random dog image".to_string()).collect::<Vec<String>>();

        DogDataset {
            image_paths,
            caption_paths,
            batch_size,
        }
    }

    // Method to simulate the loading of a batch of images and captions
    pub fn iter_batch(&self, batch_size: usize) -> Vec<(Tensor, Tensor)> {
        let mut batches = Vec::new();
        let num_batches = self.image_paths.len() / batch_size;

        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;

            let image_batch = self.image_paths[start..end]
                .iter()
                .map(|_| Tensor::randn(&[3, 128, 128], (Kind::Float, Device::Cuda(0)))) // Simulate random images
                .collect::<Vec<Tensor>>();

            let caption_batch = self.caption_paths[start..end]
                .iter()
                .map(|_| Tensor::randn(&[1, 512], (Kind::Float, Device::Cuda(0)))) // Simulate random captions (embeddings)
                .collect::<Vec<Tensor>>();

            let image_batch_tensor = Tensor::stack(&image_batch, 0); // Stack images along the batch dimension
            let caption_batch_tensor = Tensor::stack(&caption_batch, 0); // Stack captions along the batch dimension

            batches.push((image_batch_tensor, caption_batch_tensor));
        }

        batches
    }
}
