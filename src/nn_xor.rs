use nn::{Matrix, Memory, NN};

pub fn main() {
    #[rustfmt::skip]
    let train_vec = vec![
        0f32, 0f32, 0f32,
        0f32, 1f32, 1f32,
        1f32, 0f32, 1f32,
        1f32, 1f32, 0f32,
    ];

    let train_memory = Memory::from(train_vec);
    let train = Matrix::from_memory(&train_memory, 4, 3);
    let ti = train.sub_matrix(0, 0, 4, 2);
    let to = train.sub_matrix(0, 2, 4, 1);

    let mut xor = NN::new(&[2, 2, 1]);
    println! {"training input = {ti}"};
    println! {"training output = {to}"};

    let eps = 1e-1;
    let rate = 1e-1;
    let cycles = 1e4 as usize;

    for i in 0..cycles {
        xor.finite_diff(eps, &ti, &to);
        xor.learn(rate);
    }

    println!("Training results:");
    for i in 0..train.rows {
        let input = train.sub_matrix(i, 0, 1, 2).as_vec();
        xor.set_input(&input);
        xor.forward();
        let y = xor.get_output();
        println!("{input:?} = {y:?}");
    }

    println!("\nNetwork weights: {xor}");
}
