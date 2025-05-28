#![allow(dead_code)]

use std::cell::Cell;
use std::fs::File;
use std::io::{Read, Result, Write};
use std::path::Path;

fn as_u8_slice<T>(value: &T) -> &[u8] {
    let ptr = value as *const T as *const u8;
    let size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, size) }
}

fn as_mut_u8_slice<T>(value: &mut T) -> &mut [u8] {
    let ptr = value as *mut T as *mut u8;
    let size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(ptr, size) }
}

fn vec_as_u8_slice<T>(v: &Vec<T>) -> &[u8] {
    let ptr = v.as_ptr() as *const u8;
    let len = v.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn vec_as_mut_u8_slice<T>(v: &mut Vec<T>) -> &mut [u8] {
    let ptr = v.as_ptr() as *const u8 as *mut u8;
    let len = v.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

mod math {
    pub fn kronecker(i: usize, j: usize) -> f32 {
        if i == j { 1f32 } else { 0f32 }
    }

    pub fn sigmoid(x: f32) -> f32 {
        1f32 / (1f32 + (-x).exp())
    }
}

struct RandomGenerator {
    rng: rand::rngs::ThreadRng,
}

impl RandomGenerator {
    fn new() -> Self {
        Self { rng: rand::rng() }
    }

    fn next_f32(&mut self) -> f32 {
        use rand::RngCore;
        self.rng.next_u32() as f32 / u32::MAX as f32
    }

    fn next_index(&mut self, top_limit: usize) -> usize {
        use rand::RngCore;
        (self.rng.next_u64() as f64 / u64::MAX as f64 * top_limit as f64) as usize
    }
}

pub struct Memory {
    pub(crate) data: Vec<f32>,
    pub(crate) size: Cell<usize>,
    capacity: usize,
}

impl Memory {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0f32; capacity],
            size: Cell::new(0),
            capacity,
        }
    }

    pub fn from(memory: Vec<f32>) -> Self {
        let size = memory.len();
        Self {
            data: memory,
            size: Cell::new(size),
            capacity: size,
        }
    }

    fn request(&self, size: usize) -> Option<*mut f32> {
        if self.size.get() + size <= self.capacity {
            let ret = unsafe { self.data.as_ptr().add(self.size.get()) };
            self.size.set(self.size.get() + size);
            Some(ret as *mut f32)
        } else {
            None
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut f32 {
        self.data.as_ptr() as *mut f32
    }
}

#[repr(C)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    es: *mut f32,
}

impl Matrix {
    pub fn get(&self, i: usize, j: usize) -> &f32 {
        debug_assert!(
            i < self.rows,
            "[ERROR]: Matrix::get() {i} exceeds {} rows",
            self.rows
        );

        debug_assert!(
            j < self.cols,
            "[ERROR]: Matrix::get() {j} exceeds {} cols",
            self.cols
        );
        unsafe { &*self.es.add(i * self.stride + j) }
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut f32 {
        debug_assert!(
            i < self.rows,
            "[ERROR]: Matrix::get_mut() {i} exceeds {} rows",
            self.rows
        );
        debug_assert!(
            j < self.cols,
            "[ERROR]: Matrix::get_mut() {j} exceeds {} cols",
            self.cols
        );
        unsafe { &mut *self.es.add(i * self.stride + j) }
    }

    pub fn alloc(memory: &Memory, rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            stride: cols,
            es: memory
                .request(rows * cols)
                .expect("Out of bounds of memory"),
        }
    }

    pub fn save<P>(&self, filepath: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let mut file = File::create(filepath)?;
        file.write_all(as_u8_slice(&self.rows))?;
        file.write_all(as_u8_slice(&self.cols))?;
        let v = self.as_vec();
        file.write_all(vec_as_u8_slice(&v))?;
        Ok(())
    }

    pub fn load<P>(filepath: P) -> Result<(Memory, Self)>
    where
        P: AsRef<Path>,
    {
        let mut file = File::open(filepath)?;
        let mut rows = 0;
        let mut cols = 0;
        file.read_exact(as_mut_u8_slice(&mut rows))?;
        file.read_exact(as_mut_u8_slice(&mut cols))?;

        let mut memory = Memory::new(rows * cols);
        file.read_exact(vec_as_mut_u8_slice(&mut memory.data))?;
        let matrix = Matrix::alloc(&memory, rows, cols);

        Ok((memory, matrix))
    }

    pub fn copy(&mut self, other: &Matrix) {
        debug_assert!(
            self.rows == other.rows,
            "[ERROR]: Matrix::copy() matrices have different rows"
        );
        debug_assert!(
            self.cols == other.cols,
            "[ERROR]: Matrix::copy() matrices have different cols"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) = *other.get(i, j);
            }
        }
    }

    pub fn sub_matrix(&self, i: usize, j: usize, rows: usize, cols: usize) -> Self {
        debug_assert!(
            i + rows <= self.rows,
            "[ERROR]: Matrix::sub_matrix() sub matrix exceeds rows of parent one"
        );
        debug_assert!(
            j + cols <= self.cols,
            "[ERROR]: Matrix::sub_matrix() sub matrix exceeds cols of parent one"
        );

        Self {
            rows,
            cols,
            stride: self.stride,
            es: unsafe { self.es.add(i * self.stride + j) },
        }
    }

    pub fn row(&self, i: usize) -> Self {
        self.sub_matrix(i, 0, 1, self.cols)
    }

    pub fn col(&self, i: usize) -> Self {
        self.sub_matrix(0, i, self.rows, 1)
    }

    pub fn from_memory(memory: &Memory, rows: usize, cols: usize) -> Self {
        debug_assert!(
            memory.size.get() == rows * cols,
            "[ERROR]: Matrix::from_memory() cannot interpret memory as a {rows}x{cols} matrix"
        );
        Self {
            es: memory.as_ptr(),
            rows,
            cols,
            stride: cols,
        }
    }

    pub fn from_ptr(ptr: *mut f32, rows: usize, cols: usize) -> Self {
        Self {
            es: ptr,
            rows,
            cols,
            stride: cols,
        }
    }

    pub fn as_vec(&self) -> Vec<f32> {
        let mut ret = vec![0f32; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                ret[i * self.cols + j] = *self.get(i, j);
            }
        }
        ret
    }

    pub fn zeros(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) = 0f32;
            }
        }
    }

    pub fn eye(&mut self) {
        assert!(
            self.rows == self.cols,
            "[ERROR]: Matrix::eye() this is not a square matrix"
        );
        for i in 0..self.rows {
            *self.get_mut(i, i) = 1f32;
        }
    }

    pub fn random(&mut self) {
        let mut rng = RandomGenerator::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) = rng.next_f32();
            }
        }
    }

    pub fn transpose(&mut self, m: &Matrix) {
        assert!(
            self.rows == m.cols && self.cols == m.rows,
            "[ERROR]: Matrix::transpose() matrix have different dimensions"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) = *m.get(j, i);
            }
        }
    }

    pub fn upper_triangle(&mut self) {
        assert!(
            self.rows == self.cols,
            "[ERROR]: Matrix::upper_triangle() this is not a square matrix"
        );

        self.zeros();

        for i in 0..self.rows {
            for j in i..self.cols {
                *self.get_mut(i, j) = 1f32;
            }
        }
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = *self.get(i, j);
                *self.get_mut(i, j) = math::sigmoid(val);
            }
        }
    }

    pub fn add(&mut self, other: &Matrix) {
        assert!(
            self.rows == other.rows,
            "[ERROR]: Matrix::sum() matrices must have the same number of rows"
        );

        assert!(
            self.cols == other.cols,
            "[ERROR]: Matrix::sum() matrices must have the same number of cols"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) += *other.get(i, j);
            }
        }
    }

    pub fn sum(&mut self, m1: &Matrix, m2: &Matrix) {
        assert!(
            self.rows == m1.rows && self.rows == m2.rows,
            "[ERROR]: Matrix::sum() matrices must have the same number of rows"
        );

        assert!(
            self.cols == m1.cols && self.cols == m2.cols,
            "[ERROR]: Matrix::sum() matrices must have the same number of cols"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut(i, j) = m1.get(i, j) + m2.get(i, j);
            }
        }
    }

    pub fn dot(&mut self, m1: &Matrix, m2: &Matrix) {
        assert!(
            self.rows == m1.rows && self.cols == m2.cols,
            "[ERROR]: Matrix::dot() matrices has wrong outer dimensions"
        );
        assert!(
            m1.cols == m2.rows,
            "[ERROR]: Matrix::dot() matrices has wrong inner dimensions"
        );

        self.zeros();

        for i in 0..self.rows {
            for j in 0..self.cols {
                for k in 0..m1.cols {
                    *self.get_mut(i, j) += m1.get(i, k) * m2.get(k, j);
                }
            }
        }
    }

    pub fn shuffle(&mut self) {
        let mut rng = RandomGenerator::new();
        for i in (1..self.cols - 1).rev() {
            let swap_index = rng.next_index(i);
            for j in 0..self.rows {
                let temp = *self.get(j, i);
                *self.get_mut(j, i) = *self.get(j, swap_index);
                *self.get_mut(j, swap_index) = temp;
            }
        }
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for i in 0..self.rows {
            write!(f, "  ")?;
            for j in 0..self.cols {
                write!(f, "{} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

pub trait ActivationFunction {
    fn activation(&self, x: f32) -> f32;
    fn activation_layer(&self, m: &mut Matrix);
    fn derivative(&self, x: f32) -> f32;
}

pub struct SigmoidActivation {}
impl ActivationFunction for SigmoidActivation {
    fn activation(&self, x: f32) -> f32 {
        math::sigmoid(x)
    }

    fn activation_layer(&self, m: &mut Matrix) {
        m.sigmoid();
    }

    fn derivative(&self, x: f32) -> f32 {
        x * (1f32 - 0.95 * x)
    }
}

pub struct ReluActivation {
    pub relu_zero: f32,
}

impl ActivationFunction for ReluActivation {
    fn activation(&self, x: f32) -> f32 {
        if x > 0f32 { x } else { x * self.relu_zero }
    }

    fn activation_layer(&self, m: &mut Matrix) {
        for i in 0..m.rows {
            for j in 0..m.cols {
                if *m.get(i, j) < 0f32 {
                    *m.get_mut(i, j) = *m.get(i, j) * self.relu_zero;
                }
            }
        }
    }

    fn derivative(&self, x: f32) -> f32 {
        if x >= 0f32 { 1f32 } else { self.relu_zero }
    }
}

pub struct NN {
    n_layers: usize,
    nn_memory: Memory,
    computation_memory: Memory,
    w: Vec<Matrix>,
    b: Vec<Matrix>,
    a: Vec<Matrix>,
    gw: Vec<Matrix>,
    gb: Vec<Matrix>,
    ga: Vec<Matrix>,
}

impl NN {
    pub fn new(layers: &[usize]) -> Self {
        let n_layers = layers.len() - 1;
        debug_assert!(n_layers > 0);
        let mut nn_memory_size = 0;
        let mut computation_memory_size = layers[0] * 2;
        for i in 1..layers.len() {
            nn_memory_size += layers[i - 1] * layers[i] + layers[i];
            computation_memory_size += layers[i - 1] * layers[i] + 3 * layers[i];
        }

        let nn_memory = Memory::new(nn_memory_size);
        let computation_memory = Memory::new(computation_memory_size);

        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut a = Vec::new();
        let mut gw = Vec::new();
        let mut gb = Vec::new();
        let mut ga = Vec::new();

        a.push(Matrix::alloc(&computation_memory, layers[0], 1));
        ga.push(Matrix::alloc(&computation_memory, layers[0], 1));
        for i in 1..layers.len() {
            w.push(Matrix::alloc(&nn_memory, layers[i], layers[i - 1]));
            b.push(Matrix::alloc(&nn_memory, layers[i], 1));
            a.push(Matrix::alloc(&computation_memory, layers[i], 1));
            gw.push(Matrix::alloc(&computation_memory, layers[i], layers[i - 1]));
            gb.push(Matrix::alloc(&computation_memory, layers[i], 1));
            ga.push(Matrix::alloc(&computation_memory, layers[i], 1));
        }

        for i in 0..n_layers {
            w[i].random();
            b[i].random();
        }

        Self {
            n_layers,
            nn_memory,
            computation_memory,
            w,
            b,
            a,
            gw,
            gb,
            ga,
        }
    }

    pub fn set_input(&mut self, input: &[f32]) {
        debug_assert!(
            self.a[0].rows == input.len(),
            "[ERROR]: NN::set_input() input len different from NN input"
        );
        for i in 0..input.len() {
            *self.a[0].get_mut(i, 0) = input[i];
        }
    }

    pub fn get_output(&mut self) -> Vec<f32> {
        self.a[self.n_layers].as_vec()
    }

    pub fn forward(&mut self, af: &impl ActivationFunction) {
        for i in 0..self.n_layers {
            let (a_prev, a_next) = self.a.split_at_mut(i + 1);
            a_next[0].dot(&self.w[i], &a_prev[i]);
            a_next[0].add(&self.b[i]);
            af.activation_layer(&mut a_next[0]);
        }
    }

    pub fn cost(&mut self, ti: &Matrix, to: &Matrix, af: &impl ActivationFunction) -> f32 {
        assert!(ti.cols == to.cols);
        assert!(to.rows == self.a[self.n_layers].rows);

        let mut cost = 0f32;

        for i in 0..ti.cols {
            self.a[0].copy(&ti.col(i));
            self.forward(af);

            for j in 0..to.rows {
                let d = self.a[self.n_layers].get(j, 0) - to.get(j, i);
                cost += d * d;
            }
        }

        cost / ti.cols as f32
    }

    pub fn finite_diff(&mut self, eps: f32, ti: &Matrix, to: &Matrix, af: &impl ActivationFunction) {
        let mut saved;
        let cost = self.cost(ti, to, af);

        for i in 0..self.n_layers {
            for j in 0..self.w[i].rows {
                for k in 0..self.w[i].cols {
                    saved = *self.w[i].get(j, k);
                    *self.w[i].get_mut(j, k) += eps;
                    *self.gw[i].get_mut(j, k) = (self.cost(ti, to, af) - cost) / eps;
                    *self.w[i].get_mut(j, k) = saved;
                }
            }
            for j in 0..self.b[i].rows {
                for k in 0..self.b[i].cols {
                    saved = *self.b[i].get(j, k);
                    *self.b[i].get_mut(j, k) += eps;
                    *self.gb[i].get_mut(j, k) = (self.cost(ti, to, af) - cost) / eps;
                    *self.b[i].get_mut(j, k) = saved;
                }
            }
        }
    }

    pub fn back_prop(&mut self, ti: &Matrix, to: &Matrix, af: &impl ActivationFunction) {
        debug_assert!(ti.rows == self.a[0].rows); // input is compatible with the network
        debug_assert!(to.rows == self.a[self.n_layers].rows); // output is compatible with the network
        debug_assert!(ti.cols == to.cols); // the number of trials in consistent

        // clear the backprop data
        for i in 0..self.n_layers {
            self.gw[i].zeros();
            self.gb[i].zeros();
            self.ga[i].zeros();
        }

        for i in 0..ti.cols {
            // forward the network
            self.set_input(&ti.sub_matrix(0, i, ti.rows, 1).as_vec());
            self.forward(af);
            for i in 0..self.n_layers {
                self.ga[i].zeros();
            }

            // propagate the error in the output layer
            for n in 0..self.a[self.n_layers].rows {
                let an = *self.a[self.n_layers].get(n, 0);
                let err = 2f32 * (an - *to.get(n, i)) * af.derivative(an);
                *self.ga[self.n_layers].get_mut(n, 0) = err;
            }

            let mut layer_index = self.n_layers - 1;
            loop {
                for nact_index in 0..self.a[layer_index + 1].rows {
                    let gap = *self.ga[layer_index + 1].get(nact_index, 0);

                    // update the biases
                    *self.gb[layer_index].get_mut(nact_index, 0) += gap;

                    for cact_index in 0..self.a[layer_index].rows {
                        // update the gradients
                        *self.gw[layer_index].get_mut(nact_index, cact_index) +=
                            gap * *self.a[layer_index].get(cact_index, 0);

                        // propagate the error unless we hit the last
                        // layer because propagating the error to the
                        // input values is useless
                        if layer_index != 0 {
                            *self.ga[layer_index].get_mut(cact_index, 0) += gap
                                * *self.w[layer_index].get(nact_index, cact_index)
                                * af.derivative(*self.a[layer_index].get(cact_index, 0));
                        }
                    }
                }
                if layer_index == 0 {
                    break;
                }
                layer_index -= 1;
            }
        }
    }

    pub fn learn(&mut self, rate: f32) {
        for i in 0..self.n_layers {
            for j in 0..self.w[i].rows {
                for k in 0..self.w[i].cols {
                    *self.w[i].get_mut(j, k) -= rate * *self.gw[i].get(j, k);
                }
            }
            for j in 0..self.b[i].rows {
                for k in 0..self.b[i].cols {
                    *self.b[i].get_mut(j, k) -= rate * *self.gb[i].get(j, k);
                }
            }
        }
    }

    pub fn save<P>(&self, filepath: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let mut file = File::create(filepath)?;
        file.write_all(as_u8_slice(&self.n_layers))?;
        for i in 0..=self.n_layers {
            file.write_all(as_u8_slice(&self.a[i].rows))?;
        }
        for i in 0..self.n_layers {
            file.write_all(vec_as_u8_slice(&self.w[i].as_vec()))?;
        }
        for i in 0..self.n_layers {
            file.write_all(vec_as_u8_slice(&self.b[i].as_vec()))?;
        }
        Ok(())
    }

    pub fn load<P>(filepath: P) -> Result<NN>
    where
        P: AsRef<Path>,
    {
        let mut file = File::open(filepath)?;
        let mut n_layers: usize = 0;
        file.read_exact(as_mut_u8_slice(&mut n_layers))?;
        let mut layers = vec![0usize; n_layers + 1];
        file.read_exact(vec_as_mut_u8_slice(&mut layers))?;

        let mut nn_memory_size = 0;
        let mut computation_memory_size = layers[0] * 2;
        for i in 1..layers.len() {
            nn_memory_size += layers[i - 1] * layers[i] + layers[i];
            computation_memory_size += layers[i - 1] * layers[i] + 3 * layers[i];
        }

        let mut nn_memory = Memory::new(nn_memory_size);
        file.read_exact(vec_as_mut_u8_slice(&mut nn_memory.data))?;
        let computation_memory = Memory::new(computation_memory_size);

        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut a = Vec::new();
        let mut gw = Vec::new();
        let mut gb = Vec::new();
        let mut ga = Vec::new();

        a.push(Matrix::alloc(&computation_memory, layers[0], 1));
        ga.push(Matrix::alloc(&computation_memory, layers[0], 1));

        for i in 1..layers.len() {
            w.push(Matrix::alloc(&nn_memory, layers[i], layers[i - 1]));
        }

        for i in 1..layers.len() {
            b.push(Matrix::alloc(&nn_memory, layers[i], 1));
            a.push(Matrix::alloc(&computation_memory, layers[i], 1));
            gw.push(Matrix::alloc(&computation_memory, layers[i], layers[i - 1]));
            gb.push(Matrix::alloc(&computation_memory, layers[i], 1));
            ga.push(Matrix::alloc(&computation_memory, layers[i], 1));
        }

        Ok(Self {
            n_layers,
            nn_memory,
            computation_memory,
            w,
            b,
            a,
            gw,
            gb,
            ga,
        })
    }
}

impl std::fmt::Display for NN {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for i in 0..self.n_layers {
            writeln!(f, "  w[{i}] = {{")?;
            for r in 0..self.w[i].rows {
                write!(f, "    ")?;
                for c in 0..self.w[i].cols {
                    write!(f, "{} ", self.w[i].get(r, c))?;
                }
                writeln!(f)?;
            }
            writeln!(f, "  }}")?;

            writeln!(f, "  b[{i}] = {{")?;
            for r in 0..self.b[i].rows {
                write!(f, "    ")?;
                for c in 0..self.b[i].cols {
                    write!(f, "{} ", self.b[i].get(r, c))?;
                }
                writeln!(f)?;
            }
            writeln!(f, "  }}")?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn memory_allocation() {
        let memory = Memory::new(9);
        let mut matrix = Matrix::alloc(&memory, 3, 3);
        matrix.eye();
        println!("Memory allocation: {}", matrix);
    }

    #[test]
    fn matrix_sum() {
        let memory = Memory::new(27);
        let mut m1 = Matrix::alloc(&memory, 3, 3);
        m1.eye();
        let mut m2 = Matrix::alloc(&memory, 3, 3);
        m2.eye();
        let mut m3 = Matrix::alloc(&memory, 3, 3);
        m3.sum(&m1, &m2);
        println!("Matrix sum: {}", m3);
        for i in 0..m3.rows {
            for j in 0..m3.cols {
                assert_eq!(*m3.get(i, j), math::kronecker(i, j) * 2f32);
            }
        }
    }

    #[test]
    fn matrix_dot() {
        let memory = Memory::new(27);
        let mut m1 = Matrix::alloc(&memory, 3, 3);
        m1.upper_triangle();
        let mut m2 = Matrix::alloc(&memory, 3, 3);
        m2.transpose(&m1);
        let mut m3 = Matrix::alloc(&memory, 3, 3);
        m3.dot(&m2, &m1);
        println!("Matrix dot: {}", m3);
        assert_eq!(*m3.get(0, 0), 1f32);
        assert_eq!(*m3.get(1, 0), 1f32);
        assert_eq!(*m3.get(2, 0), 1f32);
        assert_eq!(*m3.get(0, 1), 1f32);
        assert_eq!(*m3.get(0, 2), 1f32);
        assert_eq!(*m3.get(1, 1), 2f32);
        assert_eq!(*m3.get(1, 2), 2f32);
        assert_eq!(*m3.get(2, 1), 2f32);
        assert_eq!(*m3.get(2, 2), 3f32);
    }

    #[test]
    fn upper_triangle_transposed() {
        let memory = Memory::new(18);
        let mut matrix = Matrix::alloc(&memory, 3, 3);
        let mut transposed = Matrix::alloc(&memory, 3, 3);
        matrix.upper_triangle();
        transposed.transpose(&matrix);
        println!("Transposed of an upper triangle: {}", transposed);
        assert_eq!(*transposed.get(0, 0), 1f32);
        assert_eq!(*transposed.get(0, 1), 0f32);
        assert_eq!(*transposed.get(0, 2), 0f32);
        assert_eq!(*transposed.get(1, 0), 1f32);
        assert_eq!(*transposed.get(1, 1), 1f32);
        assert_eq!(*transposed.get(1, 2), 0f32);
        assert_eq!(*transposed.get(2, 0), 1f32);
        assert_eq!(*transposed.get(2, 1), 1f32);
        assert_eq!(*transposed.get(2, 2), 1f32);
    }

    #[test]
    fn matrix_from_memory() {
        #[rustfmt::skip]
        let vec_memory = vec![
            1f32, 0f32, 0f32,
            0f32, 1f32, 0f32,
            0f32, 0f32, 1f32];
        let memory = Memory::from(vec_memory);
        let matrix = Matrix::from_memory(&memory, 3, 3);

        assert_eq!(*matrix.get(0, 0), 1f32);
        assert_eq!(*matrix.get(0, 1), 0f32);
        assert_eq!(*matrix.get(0, 2), 0f32);
        assert_eq!(*matrix.get(1, 0), 0f32);
        assert_eq!(*matrix.get(1, 1), 1f32);
        assert_eq!(*matrix.get(1, 2), 0f32);
        assert_eq!(*matrix.get(2, 0), 0f32);
        assert_eq!(*matrix.get(2, 1), 0f32);
        assert_eq!(*matrix.get(2, 2), 1f32);
    }

    #[test]
    fn matrix_row_col_copy() {
        let memory = Memory::new(32);
        let mut matrix1 = Matrix::alloc(&memory, 3, 3);
        let mut matrix2 = Matrix::alloc(&memory, 1, 3);
        let mut matrix3 = Matrix::alloc(&memory, 3, 1);

        matrix1.eye();
        matrix2.copy(&matrix1.row(0));
        assert_eq!(*matrix2.get(0, 0), 1f32);
        assert_eq!(*matrix2.get(0, 1), 0f32);
        assert_eq!(*matrix2.get(0, 2), 0f32);
        matrix3.copy(&matrix1.col(0));
        assert_eq!(*matrix3.get(0, 0), 1f32);
        assert_eq!(*matrix3.get(1, 0), 0f32);
        assert_eq!(*matrix3.get(2, 0), 0f32);

        matrix2.copy(&matrix1.row(1));
        assert_eq!(*matrix2.get(0, 0), 0f32);
        assert_eq!(*matrix2.get(0, 1), 1f32);
        assert_eq!(*matrix2.get(0, 2), 0f32);
        matrix3.copy(&matrix1.col(1));
        assert_eq!(*matrix3.get(0, 0), 0f32);
        assert_eq!(*matrix3.get(1, 0), 1f32);
        assert_eq!(*matrix3.get(2, 0), 0f32);

        matrix2.copy(&matrix1.row(2));
        assert_eq!(*matrix2.get(0, 0), 0f32);
        assert_eq!(*matrix2.get(0, 1), 0f32);
        assert_eq!(*matrix2.get(0, 2), 1f32);
        matrix3.copy(&matrix1.col(2));
        assert_eq!(*matrix3.get(0, 0), 0f32);
        assert_eq!(*matrix3.get(1, 0), 0f32);
        assert_eq!(*matrix3.get(2, 0), 1f32);
    }

    #[test]
    fn sub_matrices() {
        #[rustfmt::skip]
        let vec_memory = vec![
            1f32,  2f32,  3f32,  4f32,
            5f32,  6f32,  7f32,  8f32,
            9f32,  10f32, 11f32, 12f32,
            13f32, 14f32, 15f32, 16f32,
        ];
        let memory = Memory::from(vec_memory);
        let matrix1 = Matrix::from_memory(&memory, 4, 4);
        let matrix2 = matrix1.sub_matrix(1, 1, 3, 3);
        assert_eq!(*matrix2.get(0, 0), 6f32);
        assert_eq!(*matrix2.get(0, 1), 7f32);
        assert_eq!(*matrix2.get(0, 2), 8f32);
        assert_eq!(*matrix2.get(1, 0), 10f32);
        assert_eq!(*matrix2.get(1, 1), 11f32);
        assert_eq!(*matrix2.get(1, 2), 12f32);
        assert_eq!(*matrix2.get(2, 0), 14f32);
        assert_eq!(*matrix2.get(2, 1), 15f32);
        assert_eq!(*matrix2.get(2, 2), 16f32);

        let matrix3 = matrix2.sub_matrix(1, 1, 2, 2);
        assert_eq!(*matrix3.get(0, 0), 11f32);
        assert_eq!(*matrix3.get(0, 1), 12f32);
        assert_eq!(*matrix3.get(1, 0), 15f32);
        assert_eq!(*matrix3.get(1, 1), 16f32);
    }

    #[test]
    fn matrix_as_vec() {
        #[rustfmt::skip]
        let memory_vec = vec![
            1f32, 2f32, 3f32,
            4f32, 5f32, 6f32,
            7f32, 8f32, 9f32,
        ];
        let memory = Memory::from(memory_vec);
        let matrix = Matrix::from_memory(&memory, 3, 3);

        assert_eq!(
            matrix.as_vec(),
            &[1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32]
        );
    }

    #[test]
    fn set_input_get_output() {
        let mut nn = NN::new(&[3, 3, 3]);
        nn.set_input(&[1.1, 1.2, 1.3]);
        assert_eq!(*nn.a[0].get(0, 0), 1.1);
        assert_eq!(*nn.a[0].get(1, 0), 1.2);
        assert_eq!(*nn.a[0].get(2, 0), 1.3);
        *nn.a[nn.n_layers].get_mut(0, 0) = 1.0;
        *nn.a[nn.n_layers].get_mut(1, 0) = 2.0;
        *nn.a[nn.n_layers].get_mut(2, 0) = 3.0;
        let output = nn.get_output();
        assert_eq!(output[0], 1.);
        assert_eq!(output[1], 2.);
        assert_eq!(output[2], 3.);
    }

    #[test]
    fn matrix_from_ptr() {
        let mut v = vec![1f32, 2f32, 3f32, 4f32];
        let matrix = Matrix::from_ptr(v.as_mut_ptr(), 2, 2);
        assert_eq!(*matrix.get(0, 0), 1f32);
        assert_eq!(*matrix.get(0, 1), 2f32);
        assert_eq!(*matrix.get(1, 0), 3f32);
        assert_eq!(*matrix.get(1, 1), 4f32);
    }

    #[test]
    fn matrix_save_load() {
        let mut v = vec![1f32, 2f32, 3f32, 4f32];
        let matrix = Matrix::from_ptr(v.as_mut_ptr(), 2, 2);
        matrix.save("test.mat").expect("Failed to save the matrix");
        let (_memory, loaded_matrix) = Matrix::load("test.mat").expect("Failed to load matrix");
        assert_eq!(matrix.rows, loaded_matrix.rows);
        assert_eq!(matrix.cols, loaded_matrix.cols);
        for i in 0..matrix.rows {
            for j in 0..loaded_matrix.cols {
                assert_eq!(*matrix.get(i, j), *loaded_matrix.get(i, j));
            }
        }
    }

    #[test]
    fn nn_save_load() {
        let layout = &[3, 2, 3];
        let nn = NN::new(layout);
        nn.save("NN.mat").expect("Failed to save the NN");
        let nn_loaded = NN::load("NN.mat").expect("Failed to load the NN");
        assert_eq!(nn.n_layers, nn_loaded.n_layers);
        assert_eq!(nn.w.len(), nn_loaded.w.len());
        for i in 0..nn.w.len() {
            assert_eq!(nn.w[i].rows, nn_loaded.w[i].rows);
            assert_eq!(nn.w[i].cols, nn_loaded.w[i].cols);
            for j in 0..nn.w[i].rows {
                for k in 0..nn.w[i].cols {
                    assert_eq!(*nn.w[i].get(j, k), *nn_loaded.w[i].get(j, k));
                }
            }
        }
        assert_eq!(nn.b.len(), nn_loaded.b.len());
        for i in 0..nn.b.len() {
            assert_eq!(nn.b[i].rows, nn_loaded.b[i].rows);
            assert_eq!(nn.b[i].cols, nn_loaded.b[i].cols);
            for j in 0..nn.b[i].rows {
                for k in 0..nn.b[i].cols {
                    assert_eq!(*nn.b[i].get(j, k), *nn_loaded.b[i].get(j, k));
                }
            }
        }
        assert_eq!(nn.a.len(), nn_loaded.a.len());
        for i in 0..nn.a.len() {
            assert_eq!(nn.a[i].rows, nn_loaded.a[i].rows);
            assert_eq!(nn.a[i].cols, nn_loaded.a[i].cols);
            for j in 0..nn.a[i].rows {
                for k in 0..nn.a[i].cols {
                    assert_eq!(*nn.a[i].get(j, k), *nn_loaded.a[i].get(j, k));
                }
            }
        }
        assert_eq!(nn.gw.len(), nn_loaded.gw.len());
        for i in 0..nn.gw.len() {
            assert_eq!(nn.gw[i].rows, nn_loaded.gw[i].rows);
            assert_eq!(nn.gw[i].cols, nn_loaded.gw[i].cols);
        }
        assert_eq!(nn.gw.len(), nn_loaded.gw.len());
        for i in 0..nn.gb.len() {
            assert_eq!(nn.gb[i].rows, nn_loaded.gb[i].rows);
            assert_eq!(nn.gb[i].cols, nn_loaded.gb[i].cols);
        }
        for i in 0..nn.ga.len() {
            assert_eq!(nn.ga[i].rows, nn_loaded.ga[i].rows);
            assert_eq!(nn.ga[i].cols, nn_loaded.ga[i].cols);
        }
    }
}
