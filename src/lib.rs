use anyhow::anyhow;
use ndarray::{Array1, Array2, ArrayBase, Axis, Ix2, OwnedRepr};
use std::array::TryFromSliceError;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_INPUT_LIMIT: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum BgeError {
    #[error(
        "Number of tokens in the input exceed the model limit. Limit: {}, got: {}",
        MODEL_INPUT_LIMIT,
        0
    )]
    LargeInput(usize),
    #[error(transparent)]
    OnnxRuntimeError(#[from] ort::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub struct Bge {
    tokenizer: Tokenizer,
    model: ort::Session,
}

impl Bge {
    /// Creates a new instance of `Bge` by loading a tokenizer and a model from the specified file paths.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_file_path` - A path to the file containing the tokenizer configuration.
    /// * `model_file_path` - A path to the ONNX model file.
    ///
    /// # Returns
    ///
    /// If successful, returns an `Ok(Self)` containing a new instance of `Bge`. On failure, returns an `Err(anyhow::Error)`
    /// detailing the error encountered during the loading process.
    ///
    /// # Errors
    ///
    /// This function can fail if:
    /// - The paths provided do not point to valid files.
    /// - The tokenizer or model file cannot be correctly parsed or loaded, possibly due to format issues or
    ///   compatibility problems.
    ///
    /// # Examples
    ///
    /// ```
    /// let bge = bge::Bge::from_files("path/to/tokenizer.json", "path/to/model.onnx");
    /// match bge {
    ///     Ok(instance) => println!("Bge instance created successfully."),
    ///     Err(e) => eprintln!("Failed to create Bge instance: {}", e),
    /// }
    /// ```
    pub fn from_files<P>(tokenizer_file_path: P, model_file_path: P) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let tokenizer = Tokenizer::from_file(tokenizer_file_path.as_ref().to_str().unwrap())
            .map_err(|e| anyhow!(e))?;
        let model = ort::Session::builder()?.commit_from_file(model_file_path)?;
        Ok(Self { tokenizer, model })
    }

    /// Generates embeddings for a given input text using the model.
    ///
    /// This method tokenizes the input text, performs necessary preprocessing,
    /// and then runs the model to produce embeddings. The embeddings are normalized
    /// before being returned.
    ///
    /// # Arguments
    ///
    /// * `input` - The text input for which embeddings should be generated.
    ///
    /// # Returns
    ///
    /// If successful, returns a `Result` containing a fixed-size array of `f32` elements representing
    /// the generated embeddings. On failure, returns a `BgeError` detailing the nature of the error.
    ///
    /// # Errors
    ///
    /// This method can return an error in several cases:
    /// - `BgeError::LargeInput` if the input text produces more tokens than the model can accept.
    /// - `BgeError::OnnxRuntimeError` for errors related to running the ONNX model.
    /// - `BgeError::Other` for all other errors, including issues with tokenization or tensor extraction.
    ///
    /// # Examples
    ///
    /// ```
    /// # let bge = bge::Bge::from_files("path/to/tokenizer.json", "path/to/model.onnx").unwrap();
    /// let embeddings = bge.create_embeddings("This is a sample text.");
    /// match embeddings {
    ///     Ok(embeds) => println!("Embeddings: {:?}", embeds),
    ///     Err(e) => eprintln!("Error generating embeddings: {}", e),
    /// }
    /// ```
    pub fn create_embeddings(&self, input: &str) -> Result<[f32; 384], BgeError> {
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| BgeError::Other(anyhow!(e)))?;
        let encoding_ids = encoding.get_ids();
        let tokens_count = encoding_ids.len();

        if tokens_count > MODEL_INPUT_LIMIT {
            return Err(BgeError::LargeInput(tokens_count));
        }

        let input_ids =
            Array1::from_vec(encoding_ids.iter().map(|v| *v as i64).collect()).insert_axis(Axis(0));
        let attention_mask: ArrayBase<OwnedRepr<i64>, Ix2> = Array2::ones([1, tokens_count]);
        let token_type_ids: ArrayBase<OwnedRepr<i64>, Ix2> = Array2::zeros([1, tokens_count]);

        let inputs = ort::inputs! {
            "input_ids" => input_ids.view(),
            "attention_mask" => attention_mask.view(),
            "token_type_ids" => token_type_ids.view(),
        }
        .map_err(BgeError::OnnxRuntimeError)?;

        let outputs = self.model.run(inputs).map_err(BgeError::OnnxRuntimeError)?;

        let output = outputs["last_hidden_state"]
            .try_extract_tensor()
            .map_err(BgeError::OnnxRuntimeError)?;
        let view = output.view();

        let slice = view.rows().into_iter().next().unwrap().to_slice().unwrap();
        let mut res: [f32; 384] = slice
            .try_into()
            .map_err(|e: TryFromSliceError| BgeError::Other(e.into()))?;
        normalize(&mut res);
        Ok(res)
    }
}

fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
    if norm != 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_data;

    #[test]
    fn it_works() {
        let bge = Bge::from_files("assets/tokenizer.json", "assets/model.onnx").unwrap();
        let res = bge.create_embeddings("Some input text to generate embeddings for.");

        assert_eq!(res.unwrap(), test_data::TEST_EMBEDDING_RESULT);
    }
}
