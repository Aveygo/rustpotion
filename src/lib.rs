use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use regex::Regex;

use reqwest::blocking::get;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use rayon::prelude::*;

#[derive(Deserialize)]
struct Tokenizer {
    model: ModelVocab,
}

#[derive(Deserialize)]
struct ModelVocab {
    vocab: HashMap<String, i32>, 
}

fn word2tok(word: String, vocab: &HashMap<String, i32>) -> Result<Vec<i32>, ()> {
    let mut tokens = Vec::new();
    let mut word = word;
    let mut i = word.len();
    let mut continuing = false;

    while !word.is_empty() {
        let target = if continuing {
            format!("##{}", &word[..i])
        } else {
            word[..i].to_string()
        };

        match vocab.get(&target) {
            Some(&candidate) => {
                tokens.push(candidate);
                word = word[i..].to_string();
                i = word.len();
                continuing = true;
            },
            None => {
                if i == 0 {
                    return Err(());
                }
                i -= 1;
            }
        }
    }

    Ok(tokens)
}

fn setencen2tok(sentence:&str, vocab:&HashMap<String, i32>) -> Result<Vec<i32>, ()> {
    let re = Regex::new(r"[\w'-]+|[.,!?;]").unwrap();

    let mut all_tokens = vec![];
    let lower = sentence.to_lowercase();
    let words:Vec<&str> = re.find_iter(&lower).map(|mat| mat.as_str()).collect();
    
    for word in words {
        all_tokens.extend(word2tok(word.to_string(), vocab)?)
    }
    return Ok(all_tokens)

}

fn u8_to_f32_vec(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(TryInto::try_into)
        .map(Result::unwrap)
        .map(f32::from_le_bytes)
        .collect()
}

fn norm(v: &Vec<f32>) -> f32 {
    let sum_of_squares: f32 = v.iter().map(|&x| x * x).sum();
    sum_of_squares.sqrt()
}


fn download_file(url: &str, destination: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let response = get(url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download file: {}", response.status()).into());
    }
    
    let mut file = File::create(destination)?;
    let content = response.bytes()?;
    file.write_all(&content)?;
    
    Ok(())
}

pub struct RustPotion {
    embeddings: Vec<f32>, // flattened 2d array: (vocab, dimensions)
    dimensions: usize,
    vocab: HashMap<String, i32>
}

#[derive(Debug)]
pub enum PotionKind {
    BASE8M,
    BASE4M,
    BASE2M,
}

impl std::fmt::Display for PotionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn load_embeddings(safetensors_dst: &Path) -> (Vec<f32>, usize) {
    let file = std::fs::read(safetensors_dst).unwrap();
    let emb_tensors = SafeTensors::deserialize(&file).unwrap();
    let embeddings = emb_tensors.tensor("embeddings").unwrap();
    (u8_to_f32_vec(embeddings.data()), embeddings.shape()[1])
}

fn load_vocab(tokenizer_dst: &Path) -> HashMap<String, i32> {
    let file = std::fs::read_to_string(tokenizer_dst).expect("Unable to read file");
    let tokenizer: Tokenizer = serde_json::from_str(&file).unwrap();
    tokenizer.model.vocab
}

impl RustPotion {
    pub fn new(model_kind: PotionKind, working_directory: &Path) -> Self {

        let (safetensors_url, tokenizer_url) = match model_kind {
            PotionKind::BASE8M => ("https://huggingface.co/minishlab/potion-base-8M/resolve/main/model.safetensors".to_string(), "https://huggingface.co/minishlab/potion-base-8M/raw/main/tokenizer.json".to_string()),
            PotionKind::BASE4M => ("https://huggingface.co/minishlab/potion-base-4M/resolve/main/model.safetensors".to_string(), "https://huggingface.co/minishlab/potion-base-4M/raw/main/tokenizer.json".to_string()),
            PotionKind::BASE2M => ("https://huggingface.co/minishlab/potion-base-2M/resolve/main/model.safetensors".to_string(), "https://huggingface.co/minishlab/potion-base-2M/raw/main/tokenizer.json".to_string()),
        };

        
        let working_directory = working_directory.join(model_kind.to_string());
        if !working_directory.exists() {
            std::fs::create_dir_all(&working_directory).unwrap();
        }

        let safetensors_dst = working_directory.join("model.safetensors");
        let tokenizer_dst = working_directory.join("tokenizer.json");

        if !tokenizer_dst.exists() {
            download_file(&safetensors_url, &safetensors_dst).unwrap();
            download_file(&tokenizer_url, &tokenizer_dst).unwrap();
        }

        let (embeddings, dimensions) = load_embeddings(&safetensors_dst);
        let vocab = load_vocab(&tokenizer_dst);

        Self {
            embeddings,
            dimensions,
            vocab
        }
    }

    pub fn encode(&self, sentence: &str) -> Vec<f32> {

        let tokens = setencen2tok(sentence, &self.vocab).unwrap();
        let mut out_array = vec![0.0f32; self.dimensions];

        for token in tokens.iter() {
            let tmp_arr = self.embeddings[*token as usize * self.dimensions .. (*token as usize * self.dimensions) + self.dimensions].to_vec();
            for i in 0..tmp_arr.len() {
                out_array[i] += tmp_arr[i]
            }
        }

        for i in 0..out_array.len() {
            out_array[i] /= tokens.len() as f32;
        }
    
        let n = norm(&out_array);

        for i in 0..out_array.len() {
            out_array[i] /=  n;
        }

        out_array
    }

    pub fn encode_many(&self, sentences: Vec<String>) -> Vec<Vec<f32>> {
        sentences.par_iter() 
        .map(|sentence| self.encode(sentence))
        .collect()
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_encode() {
        let encoder = RustPotion::new(PotionKind::BASE2M, &Path::new("models/"));
        println!("{:?}", encoder.encode("Hello, World!"));
        assert_eq!(4, 4);
    }
}
