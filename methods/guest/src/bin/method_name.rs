// TODO: Rename this file to change the name of this method from METHOD_NAME

#![no_main]
#![no_std]

use risc0_zkvm::guest::env;
use machine_learning_core::{MnistModel, MnistData, MnistResult, inference};

risc0_zkvm::guest::entry!(main);

pub fn main() {
    let model: MnistModel = env::read();
    let data: MnistData = env::read();

    let mut res: MnistResult = MnistResult {
        n: data.x.len(),
        pred: [0 as usize; 16],
        res: 0,
    };

    match inference(&model, &data) {
        Ok(pred_vec) => {
            for (pos, p) in pred_vec.iter().enumerate() {
                if pos >= 16 {
                    res.res = -1;
                    break;
                }
                res.pred[pos] = *p;
            }
        },
        Err(_) => {
            res.res = -1;
        },
    }
    
    env::commit(&res);
}
