// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What `state` prints as text: the storage, maintenance, verification,
//! repair, archive and restore reports.

use std::fmt::Write as _;

use splinter_campaign::state::StateStorage;
use splinter_record::maintenance::{Maintained, Storage};
use splinter_record::recovery::{Archived, ArtifactFault, Repaired, Restored, StateVerify};

use crate::output::Report;

fn files(storage: &Storage) -> String {
    format!(
        "{} segment(s), {} blob pack(s), {} index run(s), {} pinned snapshot(s), {} manifest(s) walked on open; {} artifact file(s) tracked",
        storage.segments,
        storage.blob_packs,
        storage.index_runs,
        storage.pins,
        storage.history,
        storage.artifacts
    )
}

impl Report for StateStorage {
    fn human(&self) -> String {
        let mut out = format!(
            "state:   {}\ndatabase: {}\n",
            self.state.display(),
            files(&self.storage)
        );
        for loss in &self.losses {
            let _ = writeln!(
                out,
                "lost:    {} {} ({}); `state repair --from <copy>` cannot restore it unless a copy appears",
                loss.kind, loss.id, loss.detail
            );
        }
        out
    }
}

impl Report for Maintained {
    fn human(&self) -> String {
        let mut out = format!(
            "before: {}\nafter:  {}\nmerged {} group(s) of segments and {} of blob packs; retired {} finished writer(s)\n",
            files(&self.before),
            files(&self.after),
            self.segment_groups,
            self.blob_groups,
            self.writers_retired
        );
        if let Some(removed) = self.removed {
            out.push_str(&format!("collected {removed} file(s) nothing reaches\n"));
        }
        if let Some(orphans) = self.orphan_artifacts {
            out.push_str(&format!(
                "removed {orphans} artifact file(s) no commit made official\n"
            ));
        }
        out
    }
}

impl Report for StateVerify {
    fn human(&self) -> String {
        let db = &self.database;
        let mut out = format!(
            "database: {} manifest(s), {} segment(s), {} blob pack(s), {} index file(s) walked\n",
            db.manifests, db.segments, db.blob_packs, db.index_runs
        );
        for problem in &db.problems {
            let _ = writeln!(
                out,
                "problem: {:?} {} ({}): {:?}{}",
                problem.kind,
                problem.id,
                problem.path,
                problem.fault,
                if problem.records > 0 {
                    format!(", {} record(s) in it", problem.records)
                } else {
                    String::new()
                }
            );
        }
        for artifact in &self.artifacts {
            let what = match artifact.fault {
                ArtifactFault::Missing => "its file is missing",
                ArtifactFault::SizeChanged => "its file has another size than was recorded",
                ArtifactFault::Corrupt => "its bytes are not what its digest says",
            };
            let _ = writeln!(
                out,
                "problem: artifact {} ({}): {what}",
                artifact.digest, artifact.role
            );
        }
        if self.artifacts_unchecked {
            out.push_str("artifacts: not checked, the database that names them is damaged\n");
        }
        if self.is_sound() {
            out.push_str("sound: nothing is missing or damaged\n");
        } else {
            out.push_str("repair with: splinter state repair --from <archive or state root>\n");
        }
        out
    }
}

impl Report for Repaired {
    fn human(&self) -> String {
        let mut out = format!(
            "filled {} file(s) from copies, rebuilt {} index file(s), withdrew {} database file(s)\n",
            self.filled, self.rebuilt, self.quarantined
        );
        for loss in &self.lost {
            let _ = writeln!(
                out,
                "written off: {} {} ({}){}",
                loss.kind,
                loss.id,
                loss.detail,
                if loss.records > 0 {
                    format!(", {} record(s) lost", loss.records)
                } else {
                    String::new()
                }
            );
        }
        if self.unresolved > 0 {
            let _ = writeln!(
                out,
                "{} problem(s) remain: give another copy with --from, or --accept-loss to write them off",
                self.unresolved
            );
        }
        out
    }
}

impl Report for Archived {
    fn human(&self) -> String {
        format!(
            "archived snapshot {}: {} file(s) ({} artifact(s)), {} carried, {} bytes\n",
            self.snapshot, self.files, self.artifacts, self.carried, self.bytes
        )
    }
}

impl Report for Restored {
    fn human(&self) -> String {
        format!(
            "restored {} file(s) ({} artifact(s)); the state verified before it was put in place\n",
            self.files, self.artifacts
        )
    }
}
