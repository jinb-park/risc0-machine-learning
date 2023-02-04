// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language govesrning permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MnistModel {
    pub w1: Vec<Vec<i32>>, // 1st weight layer: (40, 16)
    pub w2: Vec<Vec<i32>>, // 2nd weight layer: (16, 10)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MnistData {
    pub x: Vec<Vec<i32>>, // multiple x data where each one is of (1, 40)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MnistResult {
    pub n: usize,          // how many predictions are in
    pub pred: [usize; 16], // stores n predictions
    pub res: i32,          // 0 only if everyting goes well
}

fn relu(data: i32) -> i32 {
    if data < 0 {
        return 0;
    } else {
        return data;
    }
}

fn inference_data(model: &MnistModel, x: &Vec<i32>) -> Result<usize, &'static str> {
    // w1: matrix multiplication
    let mut w1_v = Vec::new();

    if x.len() != model.w1.len() {
        return Err("model-data length inconsistent");
    }

    let w1_cols = model.w1.get(0).map(|v| { v.len() }).unwrap();
    let w2_cols = model.w2.get(0).map(|v| { v.len() }).unwrap();

    for col in 0..w1_cols {
        let mut v: i32 = 0;

        for (i, a) in x.iter().enumerate() {
            if let Some(w1_row) = model.w1.get(i) {
                if let Some(x) = w1_row.get(col) {
                    v += a * x;
                }
            }
        }
        w1_v.push(v);
    }
    // length check
    if w1_v.len() != w1_cols {
        return Err("weight1 length error");
    }
    // relu
    for v in w1_v.iter_mut() {
        *v = relu(*v);
    }

    // w2. matrix multiplication
    let mut w2_v: Vec<i32> = Vec::new();
    for col in 0..w2_cols {
        let mut v: i32 = 0;

        for (i, a) in w1_v.iter().enumerate() {
            if let Some(w2_row) = model.w2.get(i) {
                if let Some(x) = w2_row.get(col) {
                    v += a * x;
                }
            }
        }
        w2_v.push(v);
    }
    // length check
    if w2_v.len() != w2_cols {
        return Err("weight2 length error");
    }

    // argmax
    let mut idx = 0;
    let mut max = -9999999;
    for (pos, v) in w2_v.iter().enumerate() {
        if *v > max {
            max = *v;
            idx = pos;
        }
    }
    
    return Ok(idx);
}

pub fn inference(model: &MnistModel, data: &MnistData) -> Result<Vec<usize>, &'static str> {
    let mut res = Vec::new();

    for x in data.x.iter() {
        match inference_data(model, x) {
            Ok(p) => { res.push(p) },
            Err(e) => { return Err(e) },
        }
    }

    return Ok(res);
}
