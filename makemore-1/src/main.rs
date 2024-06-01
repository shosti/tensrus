#![feature(generic_arg_infer)]
use std::io::BufRead;
use tensrus::matrix::Matrix;
use tensrus::tensor::Tensor;

const BOUNDARY: char = '.';

fn main() {
    let names = read_names();
    let prob_matrix = get_bigrams(&names).map(|_, n| n + 1.0).normalize_rows();
    let mut log_likelihood = 0.0;
    let mut n = 0;
    for name in &["andrejq".to_string()] {
        let mut cur = BOUNDARY;
        let mut chars = name.chars().chain(vec![BOUNDARY].into_iter()).peekable();
        while let Some(next) = chars.peek() {
            let prob = prob_matrix[&[stoi(cur), stoi(*next)]];
            let logprob = prob.ln();
            log_likelihood += logprob;
            n += 1;
            cur = chars.next().unwrap();
        }
    }
    let nll = -log_likelihood;
    println!("{:.4}", nll / n as f32);
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

// fn itos(i: usize) -> char {
//     if i == 0 {
//         BOUNDARY
//     } else {
//         (i as u8 - 1 + ('a' as u8)) as char
//     }
// }

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
