// Runs Vitest with an upper-case drive letter in the working directory.
// On Windows, a shell opened at "c:\..." makes Vitest load itself twice and
// every suite fails with "failed to find the current suite".
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const upperDrive = (p) => p.replace(/^[a-z]:/, (d) => d.toUpperCase());
const cwd = upperDrive(process.cwd());
const bin = upperDrive(fileURLToPath(new URL('../node_modules/vitest/vitest.mjs', import.meta.url)));
const run = spawnSync(process.execPath, [bin, ...process.argv.slice(2)], { cwd, stdio: 'inherit' });
process.exit(run.status ?? 1);
