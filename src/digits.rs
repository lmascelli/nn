use image::{GenericImageView, ImageReader};
use nn::{Matrix, Memory, NN};
use std::io;

fn create_output(digit: usize) -> Vec<f32> {
    let mut ret = vec![0f32; 10];
    ret[digit] = 1f32;
    ret
}

fn create_training_set(
    dataset_dir: &str,
    max_entries_per_digit: usize,
) -> io::Result<(Memory, Matrix)> {
    const TRAINING_ENTRY_LEN: usize = 28 * 28 + 10;
    let memory = Memory::new(TRAINING_ENTRY_LEN * 10 * max_entries_per_digit);
    let mut matrix = Matrix::alloc(&memory, TRAINING_ENTRY_LEN, 10 * max_entries_per_digit);

    let mut digit_column = 0;

    for digit in 0..10 {
        let digit_folder = format!("{dataset_dir}/{digit}/{digit}");
        for digit_index in 0..max_entries_per_digit {
            let filepath = format!("{digit_folder}/{digit_index}.png");
            let img = ImageReader::open(&filepath)
                .expect("Failed to open")
                .decode()
                .expect("Failed to decode");
            let width = img.width();
            let height = img.height();
            for i in 0..width {
                for j in 0..height {
                    *matrix.get_mut((j * height + i) as usize, digit_column) =
                        img.get_pixel(i, j).0[3] as f32 / 255f32;
                }
            }
            let output = create_output(digit);
            for (i, v) in output.iter().enumerate() {
                *matrix.get_mut(28 * 28 + i, digit_column) = *v;
            }
            digit_column += 1;
        }
    }

    Ok((memory, matrix))
}

pub fn main() -> io::Result<()> {
    // let dataset_dir = "/home/leonardo/kit/nn/dataset";
    // let (memory, mut training_matrix) = create_training_set(dataset_dir, 100)?;
    // training_matrix.shuffle();
    // training_matrix.save("Training.mat");
    let (memory, matrix) = Matrix::load("Training.mat").expect("Failed to load matrix");
    const N_SAMPLES: usize = 10;

    let training_inputs = matrix.sub_matrix(0, 0, 28 * 28, N_SAMPLES);
    let training_outputs = matrix.sub_matrix(28 * 28, 0, 10, N_SAMPLES);

    let mut nn = NN::new(&[28 * 28, 512, 128, 10]);
    for iteration in 0..100 {
        for i in 0..N_SAMPLES {
            nn.back_prop(
                &training_inputs.sub_matrix(0, i, 28*28, 1),
                &training_outputs.sub_matrix(0, i, 10, 1),
            );
            println!("{iteration}, {i} -> {}", nn.cost(
                &training_inputs.sub_matrix(0, i, 28*28, 1),
                &training_outputs.sub_matrix(0, i, 10, 1),
            ));
            nn.learn(1e-3);
        }
    }
    nn.set_input(&matrix.sub_matrix(0, 0, 28 * 28, 1).as_vec()[..]);
    nn.forward();
    println!("{:?}", nn.get_output());
    Ok(())
}
