#![no_main]
#![no_std]

use risc0_zkvm::guest::env;
use machine_learning_core::{MnistModel, MnistData, inference};

risc0_zkvm::guest::entry!(main);

pub fn main() {
    let model: MnistModel = env::read();
    let data0: MnistData = env::read();
    let data1: MnistData = env::read();

    let call_inference = |m: &MnistModel, d: &MnistData| -> i32 {
        match inference(m, &d.x) {
            Ok(pred) => pred as i32,
            Err(_) => -1,
        }
    };

    let res: (i32, i32) = (call_inference(&model, &data0), call_inference(&model, &data1));
    env::commit(&res);
}
