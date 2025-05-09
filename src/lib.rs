#![allow(dead_code)]

use std::cell::Cell;

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
}

pub struct Memory {
    data: Vec<f32>,
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
        let mut computation_memory_size = layers[0];
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
        for i in 1..layers.len() {
            w.push(Matrix::alloc(&nn_memory, layers[i], layers[i-1]));
            b.push(Matrix::alloc(&nn_memory, layers[i], 1));
            a.push(Matrix::alloc(&computation_memory, layers[i], 1));
            gw.push(Matrix::alloc(&computation_memory, layers[i], layers[i-1]));
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

    pub fn forward(&mut self) {
        for i in 0..self.n_layers {
            let (a_prev, a_next) = self.a.split_at_mut(i + 1);
            a_next[0].dot(&self.w[i], &a_prev[i]);
            a_next[0].add(&self.b[i]);
            a_next[0].sigmoid();
        }
    }

    pub fn cost(&mut self, ti: &Matrix, to: &Matrix) -> f32 {
        assert!(ti.cols == to.cols);
        assert!(to.rows == self.a[self.n_layers].rows);

        let mut cost = 0f32;

        for i in 0..ti.cols {
            self.a[0].copy(&ti.col(i));
            self.forward();

            for j in 0..to.rows {
                let d = self.a[self.n_layers].get(j, 0) - to.get(j, i);
                cost += d * d;
            }
        }

        cost / ti.cols as f32
    }

    pub fn finite_diff(&mut self, eps: f32, ti: &Matrix, to: &Matrix) {
        let mut saved;
        let cost = self.cost(ti, to);

        for i in 0..self.n_layers {
            for j in 0..self.w[i].rows {
                for k in 0..self.w[i].cols {
                    saved = *self.w[i].get(j, k);
                    *self.w[i].get_mut(j, k) += eps;
                    *self.gw[i].get_mut(j, k) = (self.cost(ti, to) - cost) / eps;
                    *self.w[i].get_mut(j, k) = saved;
                }
            }
            for j in 0..self.b[i].rows {
                for k in 0..self.b[i].cols {
                    saved = *self.b[i].get(j, k);
                    *self.b[i].get_mut(j, k) += eps;                    
                    *self.gb[i].get_mut(j, k) = (self.cost(ti, to) - cost) / eps;
                    *self.b[i].get_mut(j, k) = saved;
                }
            }
        }
    }

    pub fn back_prop_one_trial(&mut self, ti: &Matrix, to: &Matrix) {
        debug_assert!(ti.cols == 1 && to.cols == 1);
        debug_assert!(ti.rows == self.a[0].rows);
        debug_assert!(to.rows == self.a[self.n_layers].rows);
        
        // compute initial error
        for 
        
        for layer_index in (0..(self.n_layers-1)).rev() {
            
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

        assert_eq!(matrix.as_vec(), &[1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32]);
    }
}
