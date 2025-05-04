use nn::{Matrix, Memory, NN};

pub fn main() {
    #[rustfmt::skip]
    let train_vec = vec![
        0f32, 0f32, 1f32, 1f32,
        0f32, 1f32, 0f32, 1f32,
        0f32, 1f32, 1f32, 0f32,
    ];

    let train_memory = Memory::from(train_vec);
    let train = Matrix::from_memory(&train_memory, 3, 4);
    let ti = train.sub_matrix(0, 0, 2, 4);
    let to = train.sub_matrix(2, 0, 1, 4);

    let mut xor = NN::new(&[2, 2, 1]);
    println! {"training input = {ti}"};
    println! {"training output = {to}"};

    let eps = 1e-1;
    let rate = 1e-1;
    let cycles = 1e5 as usize;

    for _ in 0..cycles {
        xor.finite_diff(eps, &ti, &to);
        xor.learn(rate);
    }

    let mut sum = 0f32;

    println!("Training results:");
    for i in 0..train.cols {
        let input = train.sub_matrix(0, i, 2, 1).as_vec();
        xor.set_input(&input);
        xor.forward();
        let y = xor.get_output();
        sum += y[0];
        println!("{input:?} = {y:?}");
    }

    println!("\nNetwork weights: {xor}");

    println!("SUM IS: {sum}");
}
