#![feature(generic_arg_infer)]
use rand::distributions::Distribution;
use rand::SeedableRng;
use std::io::BufRead;
use tensrus::matrix::Matrix;
use tensrus::tensor::Tensor;
use tensrus::distribution::Multinomial;

const BOUNDARY: char = '.';

fn main() {
    let names = read_names();
    let bigrams = get_bigrams(&names).normalize_rows();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut idx = 0;
    for _ in 0..20 {
        loop {
            let v = bigrams.row(idx).unwrap().normalize();
            let d = Multinomial::from(&v);
            idx = d.sample(&mut rng);
            if idx == 0 {
                print!("\n");
                break;
            }

            let c = itos(idx);
            print!("{}", c);
        }
    }
}

fn stoi(c: char) -> usize {
    if c == BOUNDARY {
        0
    } else {
        let i = (c as usize) - ('a' as usize) + 1;
        if i > 26 {
            panic!()
        }
        i
    }
}

fn itos(i: usize) -> char {
    if i == 0 {
        BOUNDARY
    } else {
        (i as u8 - 1 + ('a' as u8)) as char
    }
}

fn read_names() -> Vec<String> {
    let file = std::fs::File::open("./data/names.txt").unwrap();
    let lines = std::io::BufReader::new(file).lines();

    let mut names = Vec::new();
    for line in lines.flatten() {
        names.push(line);
    }

    names
}

fn get_bigrams(names: &[String]) -> Matrix<f32, 27, 27> {
    let mut bigrams = Matrix::zeros();
    for name in names {
        let mut cur = BOUNDARY;
        let mut chars = name.chars().peekable();
        while let Some(next) = chars.peek() {
            bigrams = bigrams.set(&[stoi(cur), stoi(*next)], |n| n + 1.0);
            cur = chars.next().unwrap();
        }
        bigrams = bigrams.set(&[stoi(cur), stoi(BOUNDARY)], |n| n + 1.0);
    }

    bigrams
}
