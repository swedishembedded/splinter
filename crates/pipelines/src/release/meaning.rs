// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release checks that compare what two runs
// of a model say rather than how they spell it, for its clients. If your
// team needs expertise in validating a fine-tuned model against its served
// build, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Whether a served answer says what the in-process answer to the same task
//! said.
//!
//! A server that samples, or whose kernels sum in another order, gives a
//! different wording of the same answer; wording is not what serving must
//! preserve. Meaning is compared by the cosine of the answers' embeddings,
//! and the comparison carries its own control: an answer says the same as
//! its counterpart only when it is nearer to that counterpart than to the
//! in-process answer to any other task of the sample. A comparison that
//! could not tell answers to different tasks apart would call everything
//! alike and so proves nothing; this one cannot.

/// Per task, whether the served answer is nearer in meaning to its own
/// in-process answer than to the in-process answer of every other task.
///
/// `served[i]` and `in_process[i]` answer task `i`; every vector has unit
/// length, so the cosine is the dot product. A task with no other task to
/// be told apart from has no control and is not counted alike.
#[must_use]
pub fn says_the_same(served: &[Vec<f32>], in_process: &[Vec<f32>]) -> Vec<bool> {
    let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    served
        .iter()
        .enumerate()
        .map(|(task, answer)| {
            let own = in_process.get(task).map(|mine| cosine(answer, mine));
            let nearest_other = in_process
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != task)
                .map(|(_, theirs)| cosine(answer, theirs))
                .fold(None, |best: Option<f32>, c| {
                    Some(best.map_or(c, |b| b.max(c)))
                });
            matches!((own, nearest_other), (Some(own), Some(other)) if own > other)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: [f32; 3]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn a_reworded_answer_is_the_same_while_an_answer_to_another_task_is_not() {
        let in_process = vec![
            unit([1.0, 0.1, 0.0]),
            unit([0.0, 1.0, 0.1]),
            unit([0.1, 0.0, 1.0]),
        ];
        // Task 0 and 1 reworded a little; task 2 was answered as task 0 was.
        let served = vec![
            unit([0.9, 0.3, 0.1]),
            unit([0.2, 0.9, 0.2]),
            unit([1.0, 0.1, 0.0]),
        ];
        assert_eq!(says_the_same(&served, &in_process), [true, true, false]);
    }

    #[test]
    fn without_another_task_to_tell_apart_from_nothing_is_counted_alike() {
        let one = vec![unit([1.0, 0.0, 0.0])];
        assert_eq!(says_the_same(&one, &one), [false]);
        assert!(says_the_same(&[], &[]).is_empty());
    }
}
