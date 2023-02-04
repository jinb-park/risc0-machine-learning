// TODO: Update the name of the method loaded by the prover. E.g., if the method is `multiply`, replace `METHOD_NAME_ID` with `MULTIPLY_ID` and replace `METHOD_NAME_PATH` with `MULTIPLY_PATH`
use machine_learning_methods::{METHOD_NAME_ID, METHOD_NAME_PATH};
use risc0_zkvm::Prover;
// use risc0_zkvm::serde::{from_slice, to_vec};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::time::{Duration, Instant};
use risc0_zkvm::serde::{from_slice, to_vec};
use machine_learning_core::{MnistModel, MnistData, MnistResult, inference};

fn read_dim2_data(filename: &'static str) -> Vec<Vec<i32>> {
    let file = File::open(filename).expect("shold be able to read file");
    let lines = BufReader::new(file).lines();
    let mut res = Vec::new();
    for line in lines {
        if line.is_ok() {
            let v = line.unwrap().split(",")
                                        .filter_map(|s| s.parse::<i32>().ok())
                                        .collect();
            res.push(v);
        }
    }
    return res;
}

fn read_dim1_data(filename: &'static str) -> Vec<i32> {
    let file = File::open(filename).expect("shold be able to read file");
    let lines = BufReader::new(file).lines();
    let mut res = Vec::new();
    for line in lines {
        if line.is_ok() {
            res.push(line.unwrap().parse::<i32>().unwrap());
        }
    }
    return res;
}

fn main() {
    // Make the prover.
    let method_code = std::fs::read(METHOD_NAME_PATH)
        .expect("Method code should be present at the specified path; did you use the correct *_PATH constant?");
    let mut prover = Prover::new(&method_code, METHOD_NAME_ID).expect(
        "Prover should be constructed from valid method source code and corresponding method ID",
    );

    // TODO: Implement communication with the guest here

    // 1. read data and model
    let x_data = read_dim2_data("x_test.csv");
    let y_data = read_dim1_data("y_test.csv");
    let w1_data = read_dim2_data("w1_1d_40_16.csv");
    let w2_data = read_dim2_data("w2_1d_16_10.csv");

    // 2. build input data
    let model = MnistModel {
        w1: w1_data,
        w2: w2_data,
    };
    let data = MnistData {
        x: x_data.to_vec(),
    };

    // 3. simulate inference on host
    let mut success = 0;
    let res = inference(&model, &data);
    match res {
        Ok(pred) => {
            for (p, a) in pred.iter().zip(y_data.iter()) {
                if (*p as i32) == *a {
                    success += 1;
                }
                println!("[host] prediction: {}, answer: {}", *p, *a);
            }
        },
        Err(e) => {
            println!("[host] error: {}", e);
        }
    }
    println!("[host] success: {}", success);
    
    // 4. run inference on guest
    success = 0;
    let mut start = Instant::now();
    prover.add_input_u32_slice(&to_vec(&model).unwrap());
    prover.add_input_u32_slice(&to_vec(&data).unwrap());

    let receipt = prover.run().expect("code should be provable");

    let guest_res: MnistResult = from_slice(&receipt.journal).expect(
        "Journal output should deserialize into the same types (& order) that it was written",
    );
    if guest_res.res == 0 {
        for (pos, a) in y_data.iter().enumerate() {
            if (pos < guest_res.pred.len()) && ((guest_res.pred[pos] as i32) == *a) {
                success += 1;
            }
            println!("[guest] prediction: {}, answer: {}", guest_res.pred[pos], *a);
        }
    } else {
        println!("[guest] error");
    }
    println!("[guest] prove() time elapsed: {:?}", start.elapsed());

    // verify()
    start = Instant::now();
    receipt.verify(METHOD_NAME_ID).expect(
        "Code you have proven should successfully verify; did you specify the correct method ID?",
    );
    println!("[guest] verify() time elapsed: {:?}", start.elapsed()); 
    println!("[guest] success: {}", success);
}
