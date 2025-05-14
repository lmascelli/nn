use nn::{Matrix, Memory};

struct Xor {
    _memory: Memory,
    a0: Matrix,
    w1: Matrix,
    b1: Matrix,
    a1: Matrix,
    w2: Matrix,
    b2: Matrix,
    a2: Matrix,
}

impl Xor {
    fn new() -> Self {
        let memory = Memory::new(256);
        let a0 = Matrix::alloc(&memory, 1, 2);
        let mut w1 = Matrix::alloc(&memory, 2, 2);
        let mut b1 = Matrix::alloc(&memory, 1, 2);
        let a1 = Matrix::alloc(&memory, 1, 2);
        let mut w2 = Matrix::alloc(&memory, 2, 1);
        let mut b2 = Matrix::alloc(&memory, 1, 1);
        let a2 = Matrix::alloc(&memory, 1, 1);

        w1.random();
        b1.random();
        w2.random();
        b2.random();

        Self {
            _memory: memory,
            a0,
            w1,
            b1,
            a1,
            w2,
            b2,
            a2,
        }
    }

    fn forward(&mut self) {
        self.a1.dot(&self.a0, &self.w1);
        self.a1.add(&self.b1);
        self.a1.sigmoid();

        self.a2.dot(&self.a1, &self.w2);
        self.a2.add(&self.b2);
        self.a2.sigmoid();
    }

    fn cost(&mut self, ti: &Matrix, to: &Matrix) -> f32 {
        assert!(ti.rows == to.rows);
        let mut c = 0f32;

        for i in 0..ti.rows {
            // println!("Forwarding {} expecting {}", ti.row(i), to.row(i));
            self.a0.copy(&ti.row(i));
            self.forward();

            for j in 0..to.cols {
                let d = self.a2.get(0, j) - to.get(i, j);
                c += d * d;
            }
        }

        c / ti.rows as f32
    }
}

fn finite_diff(m: &mut Xor, grad: &mut Xor, eps: f32, ti: &Matrix, to: &Matrix) {
    let mut saved;
    let cost = m.cost(ti, to);

    for i in 0..m.w1.rows {
        for j in 0..m.w1.cols {
            saved = *m.w1.get(i, j);
            *m.w1.get_mut(i, j) += eps;
            *grad.w1.get_mut(i, j) = (m.cost(ti, to) - cost) / eps;
            *m.w1.get_mut(i, j) = saved;
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.cols {
            saved = *m.b1.get(i, j);
            *m.b1.get_mut(i, j) += eps;
            *grad.b1.get_mut(i, j) = (m.cost(ti, to) - cost) / eps;
            *m.b1.get_mut(i, j) = saved;
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.cols {
            saved = *m.w2.get(i, j);
            *m.w2.get_mut(i, j) += eps;
            *grad.w2.get_mut(i, j) = (m.cost(ti, to) - cost) / eps;
            *m.w2.get_mut(i, j) = saved;
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.cols {
            saved = *m.b2.get(i, j);
            *m.b2.get_mut(i, j) += eps;
            *grad.b2.get_mut(i, j) = (m.cost(ti, to) - cost) / eps;
            *m.b2.get_mut(i, j) = saved;
        }
    }
}

fn learn(m: &mut Xor, grad: &mut Xor, rate: f32) {
    for i in 0..m.w1.rows {
        for j in 0..m.w1.cols {
            *m.w1.get_mut(i, j) -= rate * *grad.w1.get(i, j);
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.cols {
            *m.b1.get_mut(i, j) -= rate * *grad.b1.get(i, j);
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.cols {
            *m.w2.get_mut(i, j) -= rate * *grad.w2.get(i, j);
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.cols {
            *m.b2.get_mut(i, j) -= rate * *grad.b2.get(i, j);
        }
    }
}

fn main() {
    let train_vec = vec![
        0f32, 0f32, 0f32, 0f32, 1f32, 1f32, 1f32, 0f32, 1f32, 1f32, 1f32, 0f32,
    ];
    let train_memory = Memory::from(train_vec);
    let train = Matrix::from_memory(&train_memory, 4, 3);
    let mut xor = Xor::new();
    let mut grad = Xor::new();
    let ti = train.sub_matrix(0, 0, 4, 2);
    let to = train.sub_matrix(0, 2, 4, 1);
    println! {"ti = {ti}"};
    println! {"ti = {to}"};
    println!("cost = {}", xor.cost(&ti, &to));
    for _ in 0..10000 {
        finite_diff(&mut xor, &mut grad, 1e-1, &ti, &to);
        learn(&mut xor, &mut grad, 1e-1);
    }

    for i in 0..train.rows {
        xor.a0.copy(&train.sub_matrix(i, 0, 1, 2));
        xor.forward();
        let y = *xor.a2.get(0, 0);
        println!("{} ^ {} = {y}", xor.a0.get(0, 0), xor.a0.get(0, 1));
    }
}
