// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training clients that never leave a run
// holding a shared accelerator after their operator stopped them, for its
// clients. If your team needs expertise in cancellation across a client and
// a model service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Ctrl-C and SIGTERM as a cancel of the recorded runs in progress.
//!
//! A brain training job is detached from the client that submitted it: the
//! client exiting does not stop it, and it keeps its card until it ends. So a
//! command interrupted while its runs are in progress cancels them instead of
//! dying: every recorded run ([`crate::runs::record`]) watches
//! [`interrupted`] as it watches a `runs cancel` request, fires its cancel
//! token, and the model adapter asks brain to cancel the job it is following
//! (`DELETE /v1/jobs/{id}`) and says whether brain confirmed it. A second
//! signal exits at once. With nothing in progress a signal exits at once, as
//! it would without the handler.
//!
//! [`Interrupts::Detach`] keeps the jobs running instead: the process exits on
//! the signal, and submitting the same command again continues each job from
//! its last saved state (its `resume_key`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The exit status of a process ended by an interrupt: 128 + SIGINT.
pub const INTERRUPTED_EXIT: i32 = 130;

/// Whether an interrupt asked the runs in progress to stop.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
/// How many recorded runs are in progress in this process.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// What an interrupt does to the brain jobs of the runs in progress.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interrupts {
    /// Cancel them, and exit once the runs have stopped.
    #[default]
    Cancel,
    /// Leave them running and exit at once; the same command resumes them.
    Detach,
}

/// What a signal did, for the caller to tell its operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interrupted {
    /// The runs in progress were asked to stop and cancel their jobs.
    Cancelling,
    /// The process exits and leaves the runs' jobs running.
    Detaching,
    /// The process exits at once: nothing was in progress, or this was a
    /// second signal.
    Exiting,
}

/// Whether an interrupt asked the runs in progress to stop.
#[must_use]
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// Marks a recorded run in progress until the guard drops.
pub(crate) struct InProgress(());

impl InProgress {
    pub(crate) fn enter() -> Self {
        ACTIVE.fetch_add(1, Ordering::SeqCst);
        Self(())
    }
}

impl Drop for InProgress {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What one more signal does under `interrupts`: the effect, and whether the
/// process exits now. The first signal while runs are in progress, under
/// [`Interrupts::Cancel`], asks them to stop.
pub fn interrupt(interrupts: Interrupts) -> Interrupted {
    if interrupts == Interrupts::Detach && ACTIVE.load(Ordering::SeqCst) > 0 {
        return Interrupted::Detaching;
    }
    if ACTIVE.load(Ordering::SeqCst) == 0 || INTERRUPTED.swap(true, Ordering::SeqCst) {
        return Interrupted::Exiting;
    }
    Interrupted::Cancelling
}

/// Handles SIGINT and SIGTERM for the rest of the process as [`interrupt`]
/// under `interrupts` says, telling `tell` what each did before the process
/// exits with [`INTERRUPTED_EXIT`] when it does.
///
/// # Errors
/// The signal handlers could not be installed.
pub fn handle_interrupts(
    interrupts: Interrupts,
    tell: impl Fn(Interrupted) + Send + 'static,
) -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()?;
    // The handlers are registered before this returns, so a signal sent
    // right after it is not missed.
    #[cfg(unix)]
    let (mut interrupt_signal, mut terminate) = {
        use tokio::signal::unix::{signal, SignalKind};
        let _entered = runtime.enter();
        (
            signal(SignalKind::interrupt())?,
            signal(SignalKind::terminate())?,
        )
    };
    std::thread::Builder::new()
        .name("interrupts".into())
        .spawn(move || {
            runtime.block_on(async move {
                loop {
                    #[cfg(unix)]
                    let received = tokio::select! {
                        r = interrupt_signal.recv() => r.is_some(),
                        r = terminate.recv() => r.is_some(),
                    };
                    #[cfg(not(unix))]
                    let received = tokio::signal::ctrl_c().await.is_ok();
                    if !received {
                        return;
                    }
                    let what = interrupt(interrupts);
                    tell(what);
                    if what != Interrupted::Cancelling {
                        std::process::exit(INTERRUPTED_EXIT);
                    }
                }
            });
        })?;
    Ok(())
}
