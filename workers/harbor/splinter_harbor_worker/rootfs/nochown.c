// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

/*
 * Swedish Embedded AB implements solutions for reproducible container-image based
 * build and test environments on hosts without a container daemon for its clients.
 * If your team needs expertise in hermetic benchmark environments then you can
 * procure our services by sending an email to info@swedishembedded.com.
 *
 * Preloaded shim for programs that run as root in a user namespace mapping a single
 * uid and gid: while an image is built (LD_PRELOAD) and in a sandbox running a
 * task's own image (/etc/ld.so.preload of its private copy). Package managers and
 * archivers chown files to owners the namespace cannot represent (man, ssh, the
 * uid an archive was made by), and the kernel answers EINVAL. Every file there
 * belongs to the one mapped identity anyway, so the shim reports those requests
 * as done. Other errors still pass. Switching identity is the same story: sudo,
 * su and daemons drop to an account the namespace cannot map (setresuid answers
 * EINVAL, setgroups EPERM once groups are denied); there is only one identity to
 * be, so those report success too.
 *
 * The calls go straight to the kernel (fchownat, fchown), never through dlsym:
 * the shim is compiled against the host's C library and loaded by images with an
 * older one, so it may only use symbols every glibc has (syscall, errno).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

static int tolerate(long result)
{
	if (result != 0 && errno == EINVAL) {
		errno = 0;
		return 0;
	}
	return (int)result;
}

int fchownat(int dirfd, const char *path, uid_t owner, gid_t group, int flags)
{
	return tolerate(syscall(SYS_fchownat, dirfd, path, owner, group, flags));
}

int chown(const char *path, uid_t owner, gid_t group)
{
	return fchownat(AT_FDCWD, path, owner, group, 0);
}

int lchown(const char *path, uid_t owner, gid_t group)
{
	return fchownat(AT_FDCWD, path, owner, group, AT_SYMLINK_NOFOLLOW);
}

int fchown(int fd, uid_t owner, gid_t group)
{
	return tolerate(syscall(SYS_fchown, fd, owner, group));
}

static int tolerate_switch(long result)
{
	if (result != 0 && (errno == EINVAL || errno == EPERM)) {
		errno = 0;
		return 0;
	}
	return (int)result;
}

int setresuid(uid_t ruid, uid_t euid, uid_t suid)
{
	return tolerate_switch(syscall(SYS_setresuid, ruid, euid, suid));
}

int setresgid(gid_t rgid, gid_t egid, gid_t sgid)
{
	return tolerate_switch(syscall(SYS_setresgid, rgid, egid, sgid));
}

int setreuid(uid_t ruid, uid_t euid)
{
	return tolerate_switch(syscall(SYS_setreuid, ruid, euid));
}

int setregid(gid_t rgid, gid_t egid)
{
	return tolerate_switch(syscall(SYS_setregid, rgid, egid));
}

int setuid(uid_t uid)
{
	return tolerate_switch(syscall(SYS_setuid, uid));
}

int setgid(gid_t gid)
{
	return tolerate_switch(syscall(SYS_setgid, gid));
}

int seteuid(uid_t uid)
{
	return tolerate_switch(syscall(SYS_setresuid, -1, uid, -1));
}

int setegid(gid_t gid)
{
	return tolerate_switch(syscall(SYS_setresgid, -1, gid, -1));
}

int setgroups(size_t size, const gid_t *list)
{
	return tolerate_switch(syscall(SYS_setgroups, size, list));
}
