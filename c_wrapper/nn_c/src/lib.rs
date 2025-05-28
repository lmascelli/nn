use nn::{Matrix, NN};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn matrix_from_ptr(ptr: *mut f32, rows: usize, cols: usize) -> Matrix {
    Matrix::from_ptr(ptr, rows, cols)
}

#[unsafe(no_mangle)]
pub extern "C" fn matrix_print(matrix: *mut Matrix) {
    unsafe { println!("{}", &*matrix) }
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_create(n_layers: usize, layers: *const usize) -> *mut NN {
    Box::into_raw(Box::new(NN::new(unsafe {
        std::slice::from_raw_parts(layers, n_layers)
    })))
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_destroy(nn: *mut NN) {
    let _ = unsafe { Box::from_raw(nn) };
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_forward(nn: *mut NN) {
    unsafe { (*nn).forward() };
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_back_prop(nn: *mut NN, ti: *const Matrix, to: *const Matrix) {
    unsafe { (*nn).back_prop(&*ti, &*to) };
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_learn(nn: *mut NN, rate: f32) {
    unsafe { (*nn).learn(rate) };
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_set_input(nn: *mut NN, input: *const f32, len: usize) {
    unsafe { (*nn).set_input(std::slice::from_raw_parts(input, len)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn nn_get_output(nn: *mut NN, output: *mut f32) {
    unsafe {
        let nn_output = (*nn).get_output();
        for (i, value) in nn_output.iter().enumerate() {
            (*output.add(i)) = *value;
        }
    };
}
