// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that start their servers
// and trainers on the nodes a batch scheduler granted, and leave nothing
// behind when they stop, for its clients. If your team needs expertise in
// running and tearing down work across the nodes of a Slurm allocation,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A program started on nodes of the Slurm allocation this process runs
//! inside, as one job step.
//!
//! [`NodeSteps`] builds the `srun` that starts it: `--overlap` (the step
//! shares the allocation with the steps already running), the nodes by name,
//! the tasks and GPUs per node stated (a step never inherits the job-wide GPU
//! count of the step that started it), then a script on the shared filesystem
//! that records the step's id, enters the directory it was given and starts
//! the program under the configured prefix. A step starts on the node's host,
//! not in a container this process may run in; the prefix is what enters it
//! again. In the script, `{rank}`, `{local_rank}` and `{world}` become the
//! scheduler's task variables, so one step carries every rank.
//!
//! [`StepProcess::stop`] ends it: `SIGTERM` to `srun`, which forwards it to
//! every task, then, when the step has not ended within the grace period,
//! `scancel` of the recorded step id and `SIGKILL`. A process that dies with
//! its caller sends `srun` `SIGTERM` too ([`die_with_caller`]).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// The file a step's script writes `<job id>.<step id>` into.
pub const STEP_ID_FILE: &str = "step.id";

/// The file a task of a step with one task per node ([`NodeSteps::per_node_script`])
/// writes its program's exit status into, in its node's directory: how a
/// caller tells which node's program ended while the step's other tasks run.
pub const TASK_EXIT_FILE: &str = "task.exit";

/// How long a step is given to end after `SIGTERM` before it is cancelled.
pub const STOP_GRACE: Duration = Duration::from_secs(30);

/// How work is started on the allocation's nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeSteps {
    /// The `srun` program.
    pub srun: PathBuf,
    /// The `scancel` program, for a step `srun` does not end; without one a
    /// step that ignores `SIGTERM` is left to `SIGKILL` of `srun`.
    pub scancel: Option<PathBuf>,
    /// Arguments every step passes to `srun`.
    pub args: Vec<String>,
    /// Arguments a step on a single node passes besides.
    pub single_node_args: Vec<String>,
    /// The words the program is started under on a node.
    pub exec_prefix: Vec<String>,
}

/// Where one step runs: its nodes, the tasks on each, and the GPUs each node
/// gives the step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The nodes, by name.
    pub nodes: Vec<String>,
    /// Tasks started on each node.
    pub tasks_per_node: u32,
    /// GPUs the step takes on each node; `0` for none.
    pub gpus_per_node: u32,
}

/// The task one node of a step runs: its own directory and its own command
/// (a server bound to that node's address, with that node's certificate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeTask {
    /// The node, by the name the scheduler gives it (`SLURMD_NODENAME`).
    pub node: String,
    /// The directory the task runs in and records its exit in.
    pub dir: PathBuf,
    /// The program and its arguments.
    pub command: Vec<String>,
}

impl NodeSteps {
    /// The script ONE step with a task on each node of `tasks` runs (one
    /// step, not one per node: on an HPE Slingshot fabric a network VNI is
    /// granted per step, so only the tasks of one step reach each other over
    /// it). Task 0 writes the step id into `dir/`[`STEP_ID_FILE`]; each task
    /// picks its node's entry by the scheduler's `SLURMD_NODENAME`, enters
    /// its directory, runs its command under the exec prefix and writes the
    /// exit status into its [`TASK_EXIT_FILE`]. A node with no entry fails
    /// with its name.
    #[must_use]
    pub fn per_node_script(&self, dir: &Path, tasks: &[NodeTask]) -> String {
        let mut script = format!(
            "#!/bin/sh\n\
             if [ \"${{SLURM_PROCID:-0}}\" = 0 ]; then printf '%s.%s\\n' \"$SLURM_JOB_ID\" \"$SLURM_STEP_ID\" > {}/{STEP_ID_FILE}; fi\n\
             case \"${{SLURMD_NODENAME:-}}\" in\n",
            quote(&dir.display().to_string())
        );
        for task in tasks {
            let words: Vec<String> = self
                .exec_prefix
                .iter()
                .chain(&task.command)
                .map(|word| shell_word(word))
                .collect();
            let task_dir = quote(&task.dir.display().to_string());
            script.push_str(&format!(
                "{node})\n    cd {task_dir} || exit 1\n    {command}\n    status=$?\n    \
                 echo \"$status\" > {task_dir}/{TASK_EXIT_FILE}\n    exit \"$status\" ;;\n",
                node = quote(&task.node),
                command = words.join(" ")
            ));
        }
        script.push_str(
            "*)\n    echo \"this step has no task for node ${SLURMD_NODENAME:-(unnamed)}\" >&2\n    exit 1 ;;\nesac\n",
        );
        script
    }

    /// The script a step runs in `dir`: it writes the step id into
    /// `dir/`[`STEP_ID_FILE`], enters `dir` and starts `command` under the
    /// exec prefix, with the placeholders (in the prefix too, so a wrapper
    /// can be told a rank's card) as the scheduler's task variables.
    #[must_use]
    pub fn script(&self, dir: &Path, command: &[String]) -> String {
        let dir = quote(&dir.display().to_string());
        let words: Vec<String> = self
            .exec_prefix
            .iter()
            .chain(command)
            .map(|word| shell_word(word))
            .collect();
        format!(
            "#!/bin/sh\n\
             if [ \"${{SLURM_PROCID:-0}}\" = 0 ]; then printf '%s.%s\\n' \"$SLURM_JOB_ID\" \"$SLURM_STEP_ID\" > {dir}/{STEP_ID_FILE}; fi\n\
             cd {dir} || exit 1\n\
             exec {}\n",
            words.join(" ")
        )
    }

    /// The `srun` that runs `script` (written by the caller where every node
    /// reads it) at `placement`.
    #[must_use]
    pub fn command(&self, placement: &Placement, script: &Path) -> Command {
        let mut command = Command::new(&self.srun);
        let nodes = u32::try_from(placement.nodes.len()).unwrap_or(u32::MAX);
        command
            .arg("--overlap")
            .arg(format!("--nodelist={}", placement.nodes.join(",")))
            .arg(format!("--nodes={nodes}"))
            .arg(format!("--ntasks-per-node={}", placement.tasks_per_node))
            .arg(format!(
                "--ntasks={}",
                nodes.saturating_mul(placement.tasks_per_node)
            ));
        if placement.gpus_per_node > 0 {
            command.arg(format!("--gpus-per-node={}", placement.gpus_per_node));
        } else {
            command.arg("--gpus=0");
        }
        if nodes == 1 {
            command.args(&self.single_node_args);
        }
        command.args(&self.args).arg("sh").arg(script);
        // The step states its own GPUs: the job-wide count the step that
        // started this process carries would be added to the request.
        command.env_remove("SLURM_GPUS");
        command
    }
}

/// How long checking a directory on the nodes may take: a step, and the
/// exec prefix entering the node's image.
pub const CHECK_DEADLINE: Duration = Duration::from_secs(180);

impl NodeSteps {
    /// Checks, in one step with a task on every one of `nodes` and under the
    /// exec prefix (where the programs that use it will run), that `dir`
    /// exists or can be made and can be written. `work` is a directory every
    /// node sees, for the step's script.
    ///
    /// # Errors
    /// The step could not run, or a node cannot write `dir`: the nodes'
    /// own words.
    pub fn check_writable(&self, nodes: &[String], dir: &Path, work: &Path) -> Result<(), String> {
        let script = work.join("check-writable.sh");
        let command = [
            "sh".to_string(),
            "-c".into(),
            "mkdir -p \"$1\" 2>/dev/null; if test -d \"$1\" && test -w \"$1\"; then exit 0; fi; \
             echo \"$(hostname): cannot write $1\"; exit 1"
                .into(),
            "sh".into(),
            dir.display().to_string(),
        ];
        fs::write(&script, self.script(work, &command))
            .map_err(|e| format!("cannot write {}: {e}", script.display()))?;
        let placement = Placement {
            nodes: nodes.to_vec(),
            tasks_per_node: 1,
            gpus_per_node: 0,
        };
        let mut child = self
            .command(&placement, &script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("{} could not start: {e}", self.srun.display()))?;
        let deadline = Instant::now() + CHECK_DEADLINE;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "checking {} on {} took longer than {}s",
                        dir.display(),
                        nodes.join(","),
                        CHECK_DEADLINE.as_secs()
                    ));
                }
            }
        }
        let output = child
            .wait_with_output()
            .map_err(|e| format!("reading the check of {}: {e}", dir.display()))?;
        let _ = fs::remove_file(&script);
        if output.status.success() {
            return Ok(());
        }
        let said = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = said
            .lines()
            .filter(|l| l.contains("cannot write"))
            .collect();
        Err(if lines.is_empty() {
            format!(
                "the check of {} on {} failed ({}): {}",
                dir.display(),
                nodes.join(","),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        } else {
            lines.join("; ")
        })
    }
}

/// A started step: its `srun`, and the directory its id is recorded in.
pub struct StepProcess {
    /// The `srun` process.
    pub child: Child,
    /// Where the step's script recorded its id.
    pub dir: PathBuf,
    /// The `scancel` that stops it when `srun` does not.
    pub scancel: Option<PathBuf>,
}

impl StepProcess {
    /// The scheduler's `<job>.<step>` id, once the step has started.
    #[must_use]
    pub fn step_id(&self) -> Option<String> {
        read_step_id(&self.dir)
    }

    /// Ends the step: `SIGTERM` to `srun` (which forwards it to every task),
    /// then after `grace` the step's cancellation and `SIGKILL`. Returns
    /// whether it had to be cancelled.
    pub fn stop(&mut self, grace: Duration) -> bool {
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return false;
        }
        // The pid is the child's, which this process has not reaped.
        if let Ok(pid) = i32::try_from(self.child.id()) {
            // SAFETY: signalling a child of this process.
            unsafe { libc::kill(pid, libc::SIGTERM) };
        }
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if let (Some(scancel), Some(step)) = (&self.scancel, self.step_id()) {
            // A step already gone is no error worth more than the stop.
            let _ = Command::new(scancel).arg(&step).status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        true
    }
}

impl Drop for StepProcess {
    fn drop(&mut self) {
        self.stop(STOP_GRACE);
    }
}

/// The `<job>.<step>` id the step started in `dir` recorded.
#[must_use]
pub fn read_step_id(dir: &Path) -> Option<String> {
    fs::read_to_string(dir.join(STEP_ID_FILE))
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|id| {
            let mut parts = id.split('.');
            matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(job), Some(step), None)
                    if !job.is_empty() && !step.is_empty()
                        && id.bytes().all(|b| b.is_ascii_digit() || b == b'.')
            )
        })
}

/// Makes the process `command` starts receive `signal` when this process
/// dies, so a killed caller does not leave it running. (Linux sends it when
/// the thread that spawned it ends; servers and steps are started from a
/// caller's main flow, not from a short-lived worker thread.)
pub fn die_with_caller(command: &mut Command, signal: i32) {
    use std::os::unix::process::CommandExt as _;
    let caller = std::process::id();
    // SAFETY: the closure makes only async-signal-safe calls (`prctl`,
    // `getppid`) and touches no memory shared with the parent.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, signal) != 0 {
                return Err(io::Error::last_os_error());
            }
            // The caller died between the fork and the prctl.
            if u32::try_from(libc::getppid()).ok() != Some(caller) {
                return Err(io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// Writes a fresh secret, 32 random bytes as 64 hex digits, to `path`,
/// readable by its owner only, and never over an existing file: a secret is
/// written once and shared by every process it admits.
///
/// # Errors
/// The file exists, or the system's randomness or the file cannot be used.
pub fn write_secret(path: &Path) -> io::Result<()> {
    use std::io::{Read, Write};
    use std::os::unix::fs::OpenOptionsExt;
    let mut bytes = [0u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(hex.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()
}

/// One command word for a step's shell.
///
/// The rank, local-rank and world placeholders become the scheduler's own
/// task variables, so one script gives every task its own values. Every
/// other character is quoted: a command is a program and its arguments,
/// never a shell line, and must not gain one here.
#[must_use]
pub fn shell_word(argument: &str) -> String {
    const EXPANSIONS: [(&str, &str); 3] = [
        ("{local_rank}", "${SLURM_LOCALID}"),
        ("{rank}", "${SLURM_PROCID}"),
        ("{world}", "${SLURM_NTASKS}"),
    ];
    let mut word = String::new();
    let mut rest = argument;
    loop {
        let next = EXPANSIONS
            .iter()
            .filter_map(|(placeholder, expansion)| {
                rest.find(placeholder)
                    .map(|at| (at, *placeholder, *expansion))
            })
            .min_by_key(|(at, _, _)| *at);
        let Some((at, placeholder, expansion)) = next else {
            word.push_str(&quote(rest));
            return word;
        };
        word.push_str(&quote(&rest[..at]));
        word.push_str(expansion);
        rest = &rest[at + placeholder.len()..];
    }
}

/// `text` as one single-quoted shell word, empty when there is nothing to
/// quote so placeholders compose with their surroundings.
fn quote(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps() -> NodeSteps {
        NodeSteps {
            srun: "/sl/srun".into(),
            scancel: Some("/sl/scancel".into()),
            args: vec!["--mem=64G".into()],
            single_node_args: vec!["--network=single_node_vni".into()],
            exec_prefix: vec!["apptainer".into(), "exec".into(), "/i.sif".into()],
        }
    }

    fn args(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn placeholders_become_the_schedulers_task_variables() {
        assert_eq!(shell_word("--rank={rank}"), "'--rank='${SLURM_PROCID}");
        assert_eq!(shell_word("--world={world}"), "'--world='${SLURM_NTASKS}");
        assert_eq!(shell_word("gpu{local_rank}"), "'gpu'${SLURM_LOCALID}");
        assert_eq!(shell_word("{rank}"), "${SLURM_PROCID}");
        assert_eq!(shell_word("plain"), "'plain'");
        assert_eq!(shell_word("two words"), "'two words'");
        assert_eq!(shell_word("it's"), "'it'\\''s'");
        assert_eq!(
            shell_word("{world}{rank}"),
            "${SLURM_NTASKS}${SLURM_PROCID}"
        );
    }

    #[test]
    fn a_step_names_its_nodes_tasks_and_gpus_and_never_the_inherited_gpu_count() {
        let one = Placement {
            nodes: vec!["n442".into()],
            tasks_per_node: 4,
            gpus_per_node: 4,
        };
        let command = steps().command(&one, Path::new("/shared/run.sh"));
        assert_eq!(
            args(&command),
            [
                "--overlap",
                "--nodelist=n442",
                "--nodes=1",
                "--ntasks-per-node=4",
                "--ntasks=4",
                "--gpus-per-node=4",
                "--network=single_node_vni",
                "--mem=64G",
                "sh",
                "/shared/run.sh"
            ]
        );
        assert!(command
            .get_envs()
            .any(|(name, value)| name == "SLURM_GPUS" && value.is_none()));

        let two = Placement {
            nodes: vec!["a".into(), "b".into()],
            tasks_per_node: 1,
            gpus_per_node: 0,
        };
        let spread = args(&steps().command(&two, Path::new("/s")));
        assert!(spread.contains(&"--gpus=0".to_string()), "{spread:?}");
        assert!(spread.contains(&"--ntasks=2".to_string()));
        assert!(
            !spread.contains(&"--network=single_node_vni".to_string()),
            "a step over several nodes has its fabric access"
        );
    }

    #[test]
    fn the_script_records_the_step_enters_its_directory_and_starts_under_the_prefix() {
        let script = steps().script(
            Path::new("/shared/job dir"),
            &["brain".into(), "--job-rank".into(), "{rank}".into()],
        );
        assert!(script.contains("cd '/shared/job dir'"), "{script}");
        assert!(script.contains(STEP_ID_FILE), "{script}");
        assert!(
            script
                .contains("exec 'apptainer' 'exec' '/i.sif' 'brain' '--job-rank' ${SLURM_PROCID}"),
            "{script}"
        );
    }

    #[test]
    fn a_step_of_one_task_per_node_runs_each_nodes_command_in_its_directory_and_records_its_exit() {
        let dir = tempfile::tempdir().unwrap();
        let task = |node: &str, command: &str| NodeTask {
            node: node.into(),
            dir: dir.path().join(node),
            command: vec!["sh".into(), "-c".into(), command.into()],
        };
        let tasks = [
            task("n105", "pwd > where; exit 0"),
            task("n460", "pwd > where; echo \"$VIA\" > via; exit 3"),
        ];
        for task in &tasks {
            fs::create_dir(&task.dir).unwrap();
        }
        let plain = NodeSteps {
            exec_prefix: vec!["env".into(), "VIA=prefix".into()],
            ..NodeSteps::default()
        };
        let script = dir.path().join("step.sh");
        fs::write(&script, plain.per_node_script(dir.path(), &tasks)).unwrap();
        let run = |node: &str, procid: &str| {
            Command::new("sh")
                .arg(&script)
                .env("SLURMD_NODENAME", node)
                .env("SLURM_PROCID", procid)
                .env("SLURM_JOB_ID", "77")
                .env("SLURM_STEP_ID", "4")
                .output()
                .unwrap()
        };
        let second = run("n460", "1");
        assert_eq!(
            second.status.code(),
            Some(3),
            "the task exits as its program did"
        );
        let n460 = dir.path().join("n460");
        assert_eq!(
            fs::read_to_string(n460.join("where")).unwrap().trim(),
            n460.display().to_string()
        );
        assert_eq!(
            fs::read_to_string(n460.join("via")).unwrap().trim(),
            "prefix"
        );
        assert_eq!(
            fs::read_to_string(n460.join(TASK_EXIT_FILE))
                .unwrap()
                .trim(),
            "3"
        );
        assert!(
            read_step_id(dir.path()).is_none(),
            "only task 0 records the step"
        );
        assert!(run("n105", "0").status.success());
        assert_eq!(read_step_id(dir.path()).as_deref(), Some("77.4"));
        assert_eq!(
            fs::read_to_string(dir.path().join("n105").join(TASK_EXIT_FILE))
                .unwrap()
                .trim(),
            "0"
        );
        let stray = run("n999", "2");
        assert!(!stray.status.success());
        assert!(
            String::from_utf8_lossy(&stray.stderr).contains("n999"),
            "a node the step has no task for is named"
        );
    }

    #[test]
    fn a_secret_is_owner_only_written_once_and_fresh() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        write_secret(&a).unwrap();
        write_secret(&b).unwrap();
        let text = fs::read_to_string(&a).unwrap();
        assert_eq!(text.trim().len(), 64);
        assert_ne!(text, fs::read_to_string(&b).unwrap());
        assert_eq!(
            fs::metadata(&a).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(write_secret(&a).is_err(), "never over an existing secret");
    }

    #[test]
    fn a_step_that_ignores_sigterm_is_cancelled_by_its_id_and_killed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let scancel = dir.path().join("scancel");
        fs::write(&scancel, "#!/bin/sh\necho \"$@\" > \"$0.args\"\n").unwrap();
        fs::set_permissions(&scancel, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(dir.path().join(STEP_ID_FILE), "5600435.7\n").unwrap();
        let child = Command::new("sh")
            .args(["-c", "trap '' TERM; sleep 30"])
            .spawn()
            .unwrap();
        let mut step = StepProcess {
            child,
            dir: dir.path().to_path_buf(),
            scancel: Some(scancel.clone()),
        };
        // Let the shell install its trap before it is asked to stop.
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            step.stop(Duration::from_millis(300)),
            "it had to be cancelled"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("scancel.args"))
                .unwrap()
                .trim(),
            "5600435.7"
        );
        assert!(
            matches!(step.child.try_wait(), Ok(Some(_))),
            "and it is gone"
        );

        let polite = Command::new("sleep").arg("30").spawn().unwrap();
        let mut step = StepProcess {
            child: polite,
            dir: dir.path().to_path_buf(),
            scancel: None,
        };
        assert!(!step.stop(Duration::from_secs(5)), "SIGTERM was enough");
    }

    /// A stand-in `srun` that runs the step's script once, here.
    fn local_srun(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let srun = dir.join("srun");
        fs::write(
            &srun,
            "#!/bin/sh\nwhile [ \"$1\" != sh ]; do shift; done\nshift\nSLURM_PROCID=0 exec sh \"$@\"\n",
        )
        .unwrap();
        fs::set_permissions(&srun, fs::Permissions::from_mode(0o755)).unwrap();
        srun
    }

    #[test]
    fn a_directory_is_checked_where_the_step_runs_and_a_refusal_names_the_node() {
        let dir = tempfile::tempdir().unwrap();
        let steps = NodeSteps {
            srun: local_srun(dir.path()),
            ..NodeSteps::default()
        };
        let nodes = ["here".to_string()];
        let scratch = dir.path().join("scratch");
        fs::create_dir(&scratch).unwrap();
        steps
            .check_writable(&nodes, &scratch.join("brain-models"), dir.path())
            .unwrap();
        assert!(
            scratch.join("brain-models").is_dir(),
            "made where it is missing"
        );

        // A directory nobody can make, root included (file modes do not stop
        // root): one below a file.
        let blocked = dir.path().join("blocked");
        fs::write(&blocked, "").unwrap();
        let refused = steps
            .check_writable(&nodes, &blocked.join("other"), dir.path())
            .unwrap_err();
        assert!(refused.contains("cannot write"), "{refused}");
        assert!(refused.contains("other"), "{refused}");
    }

    #[test]
    fn only_a_job_and_step_number_is_read_as_a_step_id() {
        let dir = tempfile::tempdir().unwrap();
        for (text, read) in [
            ("5600435.73\n", Some("5600435.73")),
            ("5600435\n", None),
            (".\n", None),
            ("1.2; rm -rf /\n", None),
        ] {
            fs::write(dir.path().join(STEP_ID_FILE), text).unwrap();
            assert_eq!(read_step_id(dir.path()).as_deref(), read, "{text}");
        }
    }
}
