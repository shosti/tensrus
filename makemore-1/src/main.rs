#![feature(generic_arg_infer)]
use std::io::BufRead;
use tensrus::matrix::Matrix;
use tensrus::tensor::Tensor;

const BOUNDARY: char = '.';

fn main() {
    let names = read_names();
    let (ins, outs) = get_training_set(&names[..10]);
    println!("INS: {:?}", ins);
    println!("OUTS: {:?}", outs);
    // let prob_matrix = get_bigrams(&names).map(|_, n| n + 1.0).normalize_rows();
    // let mut log_likelihood = 0.0;
    // let mut n = 0;
    // for name in &["andrejq".to_string()] {
    //     let mut cur = BOUNDARY;
    //     let mut chars = name.chars().chain(vec![BOUNDARY].into_iter()).peekable();
    //     while let Some(next) = chars.peek() {
    //         let prob = prob_matrix[&[stoi(cur) as usize, stoi(*next) as usize]];
    //         let logprob = prob.ln();
    //         log_likelihood += logprob;
    //         n += 1;
    //         cur = chars.next().unwrap();
    //     }
    // }
    // let nll = -log_likelihood;
    // println!("{:.4}", nll / n as f32);
}

fn stoi(c: char) -> i32 {
    if c == BOUNDARY {
        0
    } else {
        let i = (c as i32) - ('a' as i32) + 1;
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
            bigrams = bigrams.set(&[stoi(cur) as usize, stoi(*next) as usize], |n| n + 1.0);
            cur = chars.next().unwrap();
        }
        bigrams = bigrams.set(&[stoi(cur) as usize, stoi(BOUNDARY) as usize], |n| n + 1.0);
    }

    bigrams
}

fn get_training_set(names: &[String]) -> (Vec<i32>, Vec<i32>) {
    let mut ins = Vec::new();
    let mut outs = Vec::new();
    for name in names {
        let mut cur = BOUNDARY;
        let mut chars = name.chars().chain(vec![BOUNDARY].into_iter()).peekable();
        while let Some(next) = chars.peek() {
            ins.push(stoi(cur));
            outs.push(stoi(*next));
            cur = chars.next().unwrap();
        }
    }

    (ins, outs)
}
