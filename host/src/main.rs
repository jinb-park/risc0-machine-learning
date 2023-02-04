use machine_learning_core::{inference, MnistData, MnistModel};
use machine_learning_methods::{ML_INFERENCE_ID, ML_INFERENCE_PATH};
use risc0_zkvm::serde::{from_slice, to_vec};
use risc0_zkvm::Prover;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn read_dim2_data(filename: &'static str) -> Vec<Vec<i32>> {
    let file = File::open(filename).expect("shold be able to read file");
    let lines = BufReader::new(file).lines();
    let mut res = Vec::new();
    for line in lines.flatten() {
        let v = line
            .split(',')
            .filter_map(|s| s.parse::<i32>().ok())
            .collect();
        res.push(v);
    }
    res
}

fn read_dim1_data(filename: &'static str) -> Vec<i32> {
    let file = File::open(filename).expect("shold be able to read file");
    let lines = BufReader::new(file).lines();
    let mut res = Vec::new();
    for line in lines.flatten() {
        res.push(line.parse::<i32>().unwrap());
    }
    res
}

fn main() {
    // Make the prover.
    let method_code = std::fs::read(ML_INFERENCE_PATH)
        .expect("Method code should be present at the specified path; did you use the correct *_PATH constant?");
    let mut prover = Prover::new(&method_code, ML_INFERENCE_ID).expect(
        "Prover should be constructed from valid method source code and corresponding method ID",
    );

    // 1. read data and model
    let x_data = read_dim2_data("data/x_test.csv");
    let y_data = read_dim1_data("data/y_test.csv");
    let w1_data = read_dim2_data("data/w1_1d_40_16.csv");
    let w2_data = read_dim2_data("data/w2_1d_16_10.csv");
    let test_num: usize = 2;

    // 2. build input data
    let model = MnistModel {
        w1: w1_data,
        w2: w2_data,
    };
    let data0 = MnistData {
        x: x_data[0].to_vec(),
    };
    let data1 = MnistData {
        x: x_data[1].to_vec(),
    };

    // 3. simulate inference on host
    let mut success = 0;
    for (pos, d) in x_data.iter().enumerate() {
        if pos >= test_num {
            break;
        }

        match inference(&model, d) {
            Ok(pred) => {
                if let Some(answer) = y_data.get(pos) {
                    if (pred as i32) == *answer {
                        success += 1;
                    }
                    println!("[host] prediction: {}, answer: {}", pred, *answer);
                }
            }
            Err(e) => {
                println!("[host] error: {}", e);
            }
        }
    }
    println!("[host] success: {}", success);

    // 4. run inference on guest
    success = 0;
    let mut start = Instant::now();
    prover.add_input_u32_slice(&to_vec(&model).unwrap());
    prover.add_input_u32_slice(&to_vec(&data0).unwrap());
    prover.add_input_u32_slice(&to_vec(&data1).unwrap());

    let receipt = prover.run().expect("code should be provable");

    let pred: (i32, i32) = from_slice(&receipt.journal).expect("from_slice error");

    if pred.0 == y_data[0] {
        success += 1;
    }
    if pred.1 == y_data[1] {
        success += 1;
    }

    println!("[guest] prediction: {}, answer: {}", pred.0, y_data[0]);
    println!("[guest] prediction: {}, answer: {}", pred.1, y_data[1]);
    println!("[guest] prove() time elapsed: {:?}", start.elapsed());

    // verify()
    start = Instant::now();
    receipt.verify(ML_INFERENCE_ID).expect(
        "Code you have proven should successfully verify; did you specify the correct method ID?",
    );
    println!("[guest] verify() time elapsed: {:?}", start.elapsed());
    println!("[guest] success: {}", success);
}
