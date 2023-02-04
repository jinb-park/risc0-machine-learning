// TODO: Rename this file to change the name of this method from METHOD_NAME

#![no_main]
#![no_std]

use risc0_zkvm::guest::env;
use machine_learning_core::{MnistModel, MnistData, inference};

risc0_zkvm::guest::entry!(main);

pub fn main() {
    //let model: MnistModel = env::read();
    let data: MnistData = env::read();

    let mut res: i32 = -1;
    /*
    match inference(&model, &data.x) {
        Ok(pred) => { res = pred as i32; }
        Err(_) => {},
    } */
    
    res = data.y[0][1];
    env::commit(&res);
}
