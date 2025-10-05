use rand::Rng;

#[cfg(test)]
pub mod test;

pub struct Network {
    sizes: Vec<usize>,           // number of neurons in each layer
    weights: Vec<Vec<f64>>,      // flat weight vectors per layer: sizes[l] * sizes[l+1]
    biases: Vec<Vec<f64>>,       // biases per layer (sizes[l+1])
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for l in 0..sizes.len() - 1 {
            let n_in = sizes[l];
            let n_out = sizes[l + 1];

            // flat weight vector of length n_in * n_out
            let weight_vec: Vec<f64> = (0..(n_in * n_out))
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            weights.push(weight_vec);

            // bias vector of length n_out, random between -1 and 1
            let bias_vec: Vec<f64> = (0..n_out)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            biases.push(bias_vec);
        }

        Network { sizes, weights, biases }
    }

    pub fn pass(&self, input: &Vec<f64>) -> Vec<f64> {
        assert!(input.len() == self.sizes[0], "Input size does not match network input layer size");
        let mut activation = input.clone();

        for l in 0..self.weights.len() {
            let n_in = self.sizes[l];
            let n_out = self.sizes[l + 1];
            let mut next_activations = vec![0.0; n_out];

            for j in 0..n_out {
                let mut z = 0.0;
                for i in 0..n_in {
                    z += self.weights[l][j * n_in + i] * activation[i];
                }
                z += self.biases[l][j];
                next_activations[j] = 1.0 / (1.0 + (-z).exp());
            }

            activation = next_activations;
        }

        activation
    }
}
