// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Which failures a repair run takes: spread across tasks, in a seeded order.

use std::collections::BTreeMap;

use splinter_core::digest::Digest;

/// How [`spread`] selects.
pub struct Spread<'a> {
    /// The most of one task, counting `held`.
    pub per_task: Option<usize>,
    /// The most in all.
    pub max: usize,
    /// Orders the tasks and the items of each.
    pub seed: u64,
    /// Items of each task already taken in an earlier run.
    pub held: &'a BTreeMap<&'a str, usize>,
}

/// The rank of `name` under `seed`: an order no listing of the store moves.
pub(super) fn rank(seed: u64, name: &str) -> String {
    Digest::of(format!("{seed}:{name}").as_bytes())
        .hex()
        .to_string()
}

/// `items` selected so they spread across tasks: the tasks in the order of
/// their seeded rank, each task's items in theirs, taken one task at a time
/// in turn, at most `per_task` of a task (its held ones counted) and `max` in
/// all. Returns the selected items, then the rest.
pub fn spread<T>(
    items: Vec<T>,
    task_of: impl Fn(&T) -> &str,
    name_of: impl Fn(&T) -> &str,
    bounds: &Spread<'_>,
) -> (Vec<T>, Vec<T>) {
    let mut by_task: BTreeMap<String, Vec<(String, T)>> = BTreeMap::new();
    for item in items {
        let task = rank(bounds.seed, task_of(&item));
        let name = rank(bounds.seed, name_of(&item));
        by_task.entry(task).or_default().push((name, item));
    }
    let mut queues: Vec<(usize, std::collections::VecDeque<T>)> = Vec::new();
    for (_, mut group) in by_task {
        group.sort_by(|a, b| a.0.cmp(&b.0));
        let held = group
            .first()
            .map(|(_, item)| bounds.held.get(task_of(item)).copied().unwrap_or(0))
            .unwrap_or(0);
        queues.push((held, group.into_iter().map(|(_, item)| item).collect()));
    }
    let mut chosen = Vec::new();
    let mut rest = Vec::new();
    loop {
        let mut took = false;
        for (taken, queue) in &mut queues {
            if chosen.len() >= bounds.max || bounds.per_task.is_some_and(|cap| *taken >= cap) {
                continue;
            }
            if let Some(item) = queue.pop_front() {
                chosen.push(item);
                *taken += 1;
                took = true;
            }
        }
        if !took {
            break;
        }
    }
    for (_, queue) in queues {
        rest.extend(queue);
    }
    (chosen, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_selection_spreads_across_tasks_within_the_bounds_and_the_seed_fixes_it() {
        // Three tasks with 6, 2 and 1 failed attempts.
        let items: Vec<(String, String)> = [("a", 6), ("b", 2), ("c", 1)]
            .iter()
            .flat_map(|(task, n)| (0..*n).map(move |i| (task.to_string(), format!("{task}{i}"))))
            .collect();
        let none = BTreeMap::new();
        let seed = splinter_model::seed::draw().value();
        // Any seed: the same one must select the same items.
        let bounds = |per_task, max, held| Spread {
            per_task,
            max,
            seed,
            held,
        };
        let (chosen, rest) = spread(
            items.clone(),
            |i| &i.0,
            |i| &i.1,
            &bounds(Some(2), 4, &none),
        );
        let count = |task: &str| chosen.iter().filter(|i| i.0 == task).count();
        // Every task gets one before any gets a second, and none more than two.
        assert_eq!(chosen.len(), 4);
        for task in ["a", "b", "c"] {
            assert!((1..=2).contains(&count(task)), "{task}: {chosen:?}");
        }
        assert_eq!(chosen.len() + rest.len(), items.len());
        let again = spread(
            items.clone(),
            |i| &i.0,
            |i| &i.1,
            &bounds(Some(2), 4, &none),
        )
        .0;
        assert_eq!(chosen, again);
        // Repairs already graded count against a task's bound.
        let held = BTreeMap::from([("a", 2)]);
        let (chosen, _) = spread(items, |i| &i.0, |i| &i.1, &bounds(Some(2), 9, &held));
        assert!(chosen.iter().all(|i| i.0 != "a"), "{chosen:?}");
    }
}
