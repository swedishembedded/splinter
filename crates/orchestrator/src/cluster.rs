// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that find and use the
// machines a batch scheduler granted them, with no hand-written endpoint
// lists, for its clients. If your team needs expertise in running model
// serving and training across the nodes of a Slurm allocation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The cluster allocation a process runs inside, and which of its nodes the
//! process may use.
//!
//! A Slurm job describes its allocation in variables: its id, its nodes in
//! the scheduler's compressed host list (`n[424,442-445]`), the GPUs each node
//! was granted. [`ClusterAllocation::from_vars`] reads them through a lookup
//! the configuration hands in, so the environment is still read in one place
//! only; [`expand_hostlist`] is the host-list grammar. A value that does not
//! parse is refused by name, never read as "no allocation".
//!
//! [`ClusterSettings`] is what the command line makes of it: `--cluster off`
//! ignores the allocation, `--cluster-nodes` restricts it to the nodes this
//! process may use (another session may own the rest), and the step settings
//! say how a process is started on another node.

use std::path::PathBuf;

/// How the allocation is used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterMode {
    /// Use the allocation when one is detected.
    #[default]
    Auto,
    /// Ignore it: everything runs on this machine, as without one.
    Off,
}

impl ClusterMode {
    /// The mode a setting names: `auto` or `off`.
    ///
    /// # Errors
    /// Any other name, refused by name.
    pub fn parse(name: &str) -> Result<ClusterMode, ClusterError> {
        match name.trim() {
            "auto" => Ok(ClusterMode::Auto),
            "off" => Ok(ClusterMode::Off),
            other => Err(ClusterError::Malformed {
                what: "--cluster".into(),
                value: other.into(),
                reason: "the modes are `auto` and `off`".into(),
            }),
        }
    }

    /// The name the mode is configured and reported under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ClusterMode::Auto => "auto",
            ClusterMode::Off => "off",
        }
    }
}

/// A batch allocation: the job, its nodes, what each was granted, and the
/// node this process runs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterAllocation {
    /// The scheduler's job id.
    pub job_id: String,
    /// Every node of the allocation, in the scheduler's order.
    pub nodes: Vec<String>,
    /// The GPUs each node was granted; `0` when the allocation names none.
    pub gpus_per_node: u32,
    /// The node this process runs on, when the scheduler or the host says.
    pub this_node: Option<String>,
}

/// The variables an allocation is read from.
pub const SLURM_VARS: &[&str] = &[
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
    "SLURMD_NODENAME",
];

impl ClusterAllocation {
    /// The allocation `var` describes (`var` answers a variable's value, or
    /// `None` when it is unset or empty); `None` when no job id is set, which
    /// is a process outside any allocation. `host` names this node when the
    /// scheduler does not.
    ///
    /// The nodes come from `SLURM_JOB_NODELIST` (else `SLURM_NODELIST`); the
    /// GPUs per node from `SLURM_GPUS_ON_NODE`, else the count of
    /// `SLURM_JOB_GPUS`'s indices.
    ///
    /// # Errors
    /// A job id without a node list, a node list that does not parse, or a
    /// GPU count that is not a number.
    pub fn from_vars(
        var: impl Fn(&str) -> Option<String>,
        host: Option<String>,
    ) -> Result<Option<ClusterAllocation>, ClusterError> {
        let Some(job_id) = var("SLURM_JOB_ID") else {
            return Ok(None);
        };
        let list = var("SLURM_JOB_NODELIST")
            .or_else(|| var("SLURM_NODELIST"))
            .ok_or_else(|| ClusterError::Malformed {
                what: "SLURM_JOB_NODELIST".into(),
                value: String::new(),
                reason: format!("job {job_id} names no nodes"),
            })?;
        let nodes = expand_hostlist(&list).map_err(|e| e.named("SLURM_JOB_NODELIST"))?;
        let gpus_per_node = match (var("SLURM_GPUS_ON_NODE"), var("SLURM_JOB_GPUS")) {
            (Some(count), _) => {
                count
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| ClusterError::Malformed {
                        what: "SLURM_GPUS_ON_NODE".into(),
                        value: count.clone(),
                        reason: "not a count".into(),
                    })?
            }
            (None, Some(indices)) => gpu_indices(&indices)?,
            (None, None) => 0,
        };
        let this_node = var("SLURMD_NODENAME").or(host);
        Ok(Some(ClusterAllocation {
            job_id,
            nodes,
            gpus_per_node,
            this_node,
        }))
    }
}

/// How many GPU indices `SLURM_JOB_GPUS` lists (`0,1,2,3`).
fn gpu_indices(text: &str) -> Result<u32, ClusterError> {
    let mut count = 0u32;
    for index in text.split(',') {
        if index.trim().parse::<u32>().is_err() {
            return Err(ClusterError::Malformed {
                what: "SLURM_JOB_GPUS".into(),
                value: text.into(),
                reason: format!("{index:?} is not a GPU index"),
            });
        }
        count += 1;
    }
    Ok(count)
}

/// How this process uses an allocation, and how it starts work on its nodes.
///
/// The programs are values: a test points them at stand-ins. A step runs
/// `srun <step args> ... <exec prefix> <program>`: the exec prefix is what a
/// node needs before the program (a container runtime, an environment
/// script), since a step on another node starts on that node's host, not in
/// the container this process may run in.
#[derive(Clone, Debug)]
pub struct ClusterSettings {
    /// `--cluster`.
    pub mode: ClusterMode,
    /// What the environment says, as [`ClusterAllocation::from_vars`] read
    /// it; an error is kept and reported when the allocation is used, so
    /// `--cluster off` never fails on it.
    pub detected: Result<Option<ClusterAllocation>, ClusterError>,
    /// `--cluster-nodes`: the nodes this process may use, as a host list;
    /// every node of the allocation when `None`.
    pub nodes: Option<String>,
    /// `--cluster-exec-prefix`: the words a step's program is started under.
    pub exec_prefix: Vec<String>,
    /// The `srun` program a step is started with.
    pub srun: Option<PathBuf>,
    /// The `scancel` program a step is stopped with when `srun` does not end
    /// it.
    pub scancel: Option<PathBuf>,
    /// Arguments every step passes to `srun` (`--mem=64G`).
    pub step_args: Vec<String>,
    /// Arguments a step on one node passes besides: on a fabric that grants
    /// network access per multi-node step only, the option that grants it
    /// to a single-node step too.
    pub single_node_step_args: Vec<String>,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            mode: ClusterMode::Auto,
            detected: Ok(None),
            nodes: None,
            exec_prefix: Vec::new(),
            srun: None,
            scancel: None,
            step_args: Vec::new(),
            single_node_step_args: Vec::new(),
        }
    }
}

/// The part of an allocation this process uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsableCluster {
    /// The allocation as detected.
    pub allocation: ClusterAllocation,
    /// The nodes this process may use, in the allocation's order.
    pub nodes: Vec<String>,
    /// Whether `--cluster-nodes` chose them.
    pub chosen: bool,
}

impl UsableCluster {
    /// Whether work is spread over the allocation's nodes rather than kept on
    /// this machine: more than one usable node, or nodes chosen by name.
    /// One node detected and nothing chosen is today's single-machine run.
    #[must_use]
    pub fn spans_nodes(&self) -> bool {
        self.nodes.len() > 1 || self.chosen
    }

    /// How many GPUs the usable nodes hold together.
    #[must_use]
    pub fn cards(&self) -> u32 {
        let nodes = u32::try_from(self.nodes.len()).unwrap_or(u32::MAX);
        nodes.saturating_mul(self.allocation.gpus_per_node)
    }
}

impl ClusterSettings {
    /// The allocation this process uses: `None` with `--cluster off` or
    /// outside an allocation.
    ///
    /// # Errors
    /// The environment's allocation did not parse; `--cluster-nodes` does not
    /// parse, names a node outside the allocation, or is given outside one.
    pub fn usable(&self) -> Result<Option<UsableCluster>, ClusterError> {
        if self.mode == ClusterMode::Off {
            return Ok(None);
        }
        let detected = self.detected.clone()?;
        let Some(allocation) = detected else {
            return match &self.nodes {
                Some(list) => Err(ClusterError::Malformed {
                    what: "--cluster-nodes".into(),
                    value: list.clone(),
                    reason: "no Slurm allocation was detected (SLURM_JOB_ID is not set)".into(),
                }),
                None => Ok(None),
            };
        };
        let Some(list) = &self.nodes else {
            let nodes = allocation.nodes.clone();
            return Ok(Some(UsableCluster {
                allocation,
                nodes,
                chosen: false,
            }));
        };
        let chosen = expand_hostlist(list).map_err(|e| e.named("--cluster-nodes"))?;
        if let Some(stranger) = chosen.iter().find(|n| !allocation.nodes.contains(n)) {
            return Err(ClusterError::Malformed {
                what: "--cluster-nodes".into(),
                value: list.clone(),
                reason: format!(
                    "{stranger} is not a node of job {} ({})",
                    allocation.job_id,
                    allocation.nodes.join(",")
                ),
            });
        }
        // The allocation's order, each node once.
        let nodes = allocation
            .nodes
            .iter()
            .filter(|n| chosen.contains(n))
            .cloned()
            .collect();
        Ok(Some(UsableCluster {
            allocation,
            nodes,
            chosen: true,
        }))
    }
}

/// Why an allocation or a host list was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ClusterError {
    /// A value that does not say what it must.
    #[error("{what} {value:?}: {reason}")]
    Malformed {
        /// The variable or flag it came from.
        what: String,
        /// The value.
        value: String,
        /// What is wrong with it.
        reason: String,
    },
}

impl ClusterError {
    /// The same refusal, naming where the value came from.
    fn named(self, what: &str) -> ClusterError {
        match self {
            ClusterError::Malformed { value, reason, .. } => ClusterError::Malformed {
                what: what.into(),
                value,
                reason,
            },
        }
    }
}

/// The host names a Slurm host list names, in order: `n[424,442-445]` is
/// `n424, n442, n443, n444, n445`; `a1,b[01-02]` is `a1, b01, b02` (a range's
/// zero padding is kept); several bracket groups in one name are expanded as
/// their product (`r[1-2]n[1-2]`). Duplicates are kept once, at their first
/// place.
///
/// # Errors
/// An empty list or name, an unbalanced or nested bracket, an empty or
/// reversed range, or a range bound that is not a number.
pub fn expand_hostlist(list: &str) -> Result<Vec<String>, ClusterError> {
    let bad = |reason: String| ClusterError::Malformed {
        what: "host list".into(),
        value: list.into(),
        reason,
    };
    let list = list.trim();
    if list.is_empty() {
        return Err(bad("it names no host".into()));
    }
    let mut hosts: Vec<String> = Vec::new();
    for name in split_top_level(list).map_err(&bad)? {
        if name.is_empty() {
            return Err(bad("an empty host name".into()));
        }
        for host in expand_name(name).map_err(&bad)? {
            if !hosts.contains(&host) {
                hosts.push(host);
            }
        }
    }
    Ok(hosts)
}

/// `list` split at the commas outside brackets.
fn split_top_level(list: &str) -> Result<Vec<&str>, String> {
    let mut names = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (at, c) in list.char_indices() {
        match c {
            '[' if depth > 0 => return Err("a bracket inside a bracket".into()),
            '[' => depth = 1,
            ']' if depth == 0 => return Err("a closing bracket with no opening one".into()),
            ']' => depth = 0,
            ',' if depth == 0 => {
                names.push(list[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("an opening bracket with no closing one".into());
    }
    names.push(list[start..].trim());
    Ok(names)
}

/// Every host one name with bracket groups stands for.
fn expand_name(name: &str) -> Result<Vec<String>, String> {
    let Some(open) = name.find('[') else {
        if name.contains(']') || name.chars().any(char::is_whitespace) {
            return Err(format!("{name:?} is not a host name"));
        }
        return Ok(vec![name.to_string()]);
    };
    let close = name[open..]
        .find(']')
        .map(|at| open + at)
        .ok_or_else(|| "an opening bracket with no closing one".to_string())?;
    let prefix = &name[..open];
    let rest = expand_name(&name[close + 1..])?;
    let mut hosts = Vec::new();
    for item in expand_group(&name[open + 1..close])? {
        for tail in &rest {
            hosts.push(format!("{prefix}{item}{tail}"));
        }
    }
    Ok(hosts)
}

/// The numbers one bracket group lists: `1,3-5`, each range's width kept.
fn expand_group(group: &str) -> Result<Vec<String>, String> {
    if group.trim().is_empty() {
        return Err("an empty bracket".into());
    }
    let mut items = Vec::new();
    for part in group.split(',') {
        let part = part.trim();
        let (low, high) = part.split_once('-').unwrap_or((part, part));
        let number = |text: &str| -> Result<u64, String> {
            if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("{part:?} is not a number or a range of numbers"));
            }
            text.parse::<u64>()
                .map_err(|_| format!("{text:?} is too large"))
        };
        let (from, to) = (number(low)?, number(high)?);
        if from > to {
            return Err(format!("the range {part:?} runs backwards"));
        }
        if to - from >= 65_536 {
            return Err(format!(
                "the range {part:?} names more hosts than a cluster has"
            ));
        }
        let width = low.len();
        for n in from..=to {
            items.push(format!("{n:0width$}"));
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn hosts(list: &str) -> Vec<String> {
        expand_hostlist(list).unwrap_or_else(|e| panic!("{list}: {e}"))
    }

    #[test]
    fn a_host_list_expands_ranges_lists_padding_and_products() {
        assert_eq!(hosts("n424"), ["n424"]);
        assert_eq!(
            hosts("n[424,442-445]"),
            ["n424", "n442", "n443", "n444", "n445"]
        );
        assert_eq!(hosts("a1, b[01-03]"), ["a1", "b01", "b02", "b03"]);
        assert_eq!(hosts("gpu[9-11]"), ["gpu9", "gpu10", "gpu11"]);
        assert_eq!(hosts("r[1-2]n[1,3]"), ["r1n1", "r1n3", "r2n1", "r2n3"]);
        assert_eq!(hosts("n[1-2]-ib"), ["n1-ib", "n2-ib"]);
        assert_eq!(hosts("n1,n1,n[1-2]"), ["n1", "n2"], "each host once");
    }

    #[test]
    fn a_malformed_host_list_is_refused_with_the_reason() {
        for (list, reason) in [
            ("", "names no host"),
            ("n[1-2", "no closing"),
            ("n1-2]", "no opening"),
            ("n[[1]]", "inside a bracket"),
            ("n[]", "empty bracket"),
            ("n[5-3]", "backwards"),
            ("n[a-b]", "not a number"),
            ("n[1-]", "not a number"),
            ("n1,,n2", "empty host name"),
            ("n[0-99999999]", "more hosts"),
        ] {
            let error = expand_hostlist(list).expect_err(list).to_string();
            assert!(error.contains(reason), "{list}: {error}");
        }
    }

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn an_allocation_is_read_from_the_schedulers_variables() {
        let allocation = ClusterAllocation::from_vars(
            vars(&[
                ("SLURM_JOB_ID", "5600435"),
                ("SLURM_JOB_NODELIST", "n[424,442,445,467]"),
                ("SLURM_GPUS_ON_NODE", "4"),
                ("SLURMD_NODENAME", "n424"),
            ]),
            Some("host".into()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(allocation.job_id, "5600435");
        assert_eq!(allocation.nodes, ["n424", "n442", "n445", "n467"]);
        assert_eq!(allocation.gpus_per_node, 4);
        assert_eq!(allocation.this_node.as_deref(), Some("n424"));

        let by_indices = ClusterAllocation::from_vars(
            vars(&[
                ("SLURM_JOB_ID", "7"),
                ("SLURM_NODELIST", "a[1-2]"),
                ("SLURM_JOB_GPUS", "0,1"),
            ]),
            Some("a1".into()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(by_indices.gpus_per_node, 2);
        assert_eq!(by_indices.this_node.as_deref(), Some("a1"));
    }

    #[test]
    fn no_job_id_is_no_allocation_and_a_bad_value_is_refused_by_name() {
        assert_eq!(ClusterAllocation::from_vars(vars(&[]), None).unwrap(), None);
        for (pairs, named) in [
            (vec![("SLURM_JOB_ID", "1")], "SLURM_JOB_NODELIST"),
            (
                vec![("SLURM_JOB_ID", "1"), ("SLURM_JOB_NODELIST", "n[1-")],
                "SLURM_JOB_NODELIST",
            ),
            (
                vec![
                    ("SLURM_JOB_ID", "1"),
                    ("SLURM_JOB_NODELIST", "n1"),
                    ("SLURM_GPUS_ON_NODE", "four"),
                ],
                "SLURM_GPUS_ON_NODE",
            ),
            (
                vec![
                    ("SLURM_JOB_ID", "1"),
                    ("SLURM_JOB_NODELIST", "n1"),
                    ("SLURM_JOB_GPUS", "0,x"),
                ],
                "SLURM_JOB_GPUS",
            ),
        ] {
            let error = ClusterAllocation::from_vars(vars(&pairs), None)
                .expect_err(named)
                .to_string();
            assert!(error.contains(named), "{error}");
        }
    }

    fn detected() -> ClusterSettings {
        ClusterSettings {
            detected: Ok(Some(ClusterAllocation {
                job_id: "9".into(),
                nodes: vec!["n424".into(), "n442".into(), "n445".into()],
                gpus_per_node: 4,
                this_node: Some("n424".into()),
            })),
            ..ClusterSettings::default()
        }
    }

    #[test]
    fn the_usable_nodes_default_to_the_allocation_and_may_be_restricted_or_turned_off() {
        let all = detected().usable().unwrap().unwrap();
        assert_eq!(all.nodes, ["n424", "n442", "n445"]);
        assert!(all.spans_nodes() && !all.chosen);
        assert_eq!(all.cards(), 12);

        let one = ClusterSettings {
            nodes: Some("n442".into()),
            ..detected()
        };
        let one = one.usable().unwrap().unwrap();
        assert_eq!(one.nodes, ["n442"]);
        assert!(one.spans_nodes(), "a node chosen by name is used");

        let off = ClusterSettings {
            mode: ClusterMode::Off,
            detected: Err(ClusterError::Malformed {
                what: "x".into(),
                value: "y".into(),
                reason: "z".into(),
            }),
            ..ClusterSettings::default()
        };
        assert_eq!(
            off.usable().unwrap(),
            None,
            "off never reads the allocation"
        );
        assert_eq!(ClusterSettings::default().usable().unwrap(), None);
    }

    #[test]
    fn a_single_node_allocation_keeps_the_single_machine_run() {
        let single = ClusterSettings {
            detected: Ok(Some(ClusterAllocation {
                job_id: "1".into(),
                nodes: vec!["n1".into()],
                gpus_per_node: 2,
                this_node: Some("n1".into()),
            })),
            ..ClusterSettings::default()
        };
        assert!(!single.usable().unwrap().unwrap().spans_nodes());
    }

    #[test]
    fn chosen_nodes_outside_the_allocation_or_without_one_are_refused() {
        let stranger = ClusterSettings {
            nodes: Some("n[442,999]".into()),
            ..detected()
        };
        let error = stranger.usable().unwrap_err().to_string();
        assert!(
            error.contains("n999") && error.contains("--cluster-nodes"),
            "{error}"
        );

        let nowhere = ClusterSettings {
            nodes: Some("n1".into()),
            ..ClusterSettings::default()
        };
        let error = nowhere.usable().unwrap_err().to_string();
        assert!(error.contains("no Slurm allocation"), "{error}");
    }

    #[test]
    fn a_mode_is_parsed_by_name() {
        assert_eq!(ClusterMode::parse("auto").unwrap(), ClusterMode::Auto);
        assert_eq!(ClusterMode::parse(" off").unwrap(), ClusterMode::Off);
        assert!(ClusterMode::parse("on").is_err());
    }
}
