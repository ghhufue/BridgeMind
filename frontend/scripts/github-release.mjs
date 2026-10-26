import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { githubManifest } from './github-release-manifest.mjs';
import { publishRelease } from './publish-release.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(root, '..');
const config = JSON.parse(readFileSync(join(root, 'release.config.json'), 'utf8'));
const version = JSON.parse(readFileSync(join(root, 'version.json'), 'utf8'));
const apk = join(repoRoot, '.build/frontend/BridgeMind-release.apk');
const metadata = JSON.parse(readFileSync(join(repoRoot, '.build/frontend/BridgeMind-release.build.json'), 'utf8'));
function command(executable, args, { allowFailure = false } = {}) {
  const result = spawnSync(executable, args, { cwd: repoRoot, encoding: 'utf8', env: { ...process.env, GIT_TERMINAL_PROMPT: '0' } });
  if ((result.error || result.status !== 0) && !allowFailure) throw new Error(`${executable} failed: ${result.stderr || result.error?.message}`);
  return result;
}
const git = args => command('git', args).stdout.trim();
if (git(['status', '--porcelain'])) throw new Error('Commit all source changes before publishing a release.');
const revision = git(['rev-parse', 'HEAD']);
if (metadata.variant !== 'release' || metadata.sourceDirty || metadata.sourceRevision !== revision ||
    metadata.versionCode !== version.versionCode || metadata.versionName !== version.versionName)
  throw new Error('Build the signed release APK from the current clean commit before publishing.');
const remote = git(['remote', 'get-url', 'origin']);
if (![ `https://github.com/${config.repository}.git`, `https://github.com/${config.repository}`, `git@github.com:${config.repository}.git` ].includes(remote))
  throw new Error('origin does not match release.config.json.');
if (!git(['ls-remote', 'origin', 'refs/heads/main']).startsWith(revision + '\t'))
  throw new Error('Push this commit to origin/main before publishing.');
const repo = JSON.parse(command('gh', ['repo', 'view', config.repository, '--json', 'nameWithOwner,isPrivate']).stdout);
if (repo.isPrivate || repo.nameWithOwner.toLowerCase() !== config.repository.toLowerCase()) throw new Error('Expected a public release repository.');
const notesPath = process.argv.find(arg => arg.startsWith('--notes-file='))?.slice('--notes-file='.length);
const notes = notesPath ? readFileSync(resolve(root, notesPath), 'utf8') : readFileSync(join(root, 'release-notes.md'), 'utf8');
const bytes = readFileSync(apk);
if (createHash('sha256').update(bytes).digest('hex') !== metadata.sha256) throw new Error('APK changed after the verified build.');
const manifest = githubManifest({ bytes, version, repository: config.repository, notes });
const tag = `v${version.versionName}`, asset = `BridgeMind-${tag}.apk`;
if (command('gh', ['release', 'view', tag, '--repo', config.repository], { allowFailure: true }).status === 0)
  throw new Error('This release already exists. Bump version.json instead of replacing published assets.');
const directory = join(repoRoot, '.build/github-release', tag);
mkdirSync(directory, { recursive: true });
copyFileSync(apk, join(directory, asset));
writeFileSync(join(directory, 'latest.json'), JSON.stringify({ ...manifest, sourceRevision: revision }, null, 2) + '\n');
writeFileSync(join(directory, 'SHA256SUMS'), `${manifest.sha256}  ${asset}\n`);
writeFileSync(join(directory, 'release-notes.md'), notes);
// A draft prevents clients seeing a release until every required asset is uploaded.
command('gh', ['release', 'create', tag, '--repo', config.repository, '--target', revision, '--title', `BridgeMind ${tag}`,
  '--notes-file', join(directory, 'release-notes.md'), '--draft']);
command('gh', ['release', 'upload', tag, join(directory, asset), join(directory, 'latest.json'), join(directory, 'SHA256SUMS'), '--repo', config.repository]);
const checkDirectory = join(directory, `verified-${Date.now()}`);
mkdirSync(checkDirectory);
command('gh', ['release', 'download', tag, '--repo', config.repository, '--dir', checkDirectory, '--pattern', asset, '--pattern', 'latest.json']);
if (createHash('sha256').update(readFileSync(join(checkDirectory, asset))).digest('hex') !== manifest.sha256 ||
    JSON.parse(readFileSync(join(checkDirectory, 'latest.json'), 'utf8')).sha256 !== manifest.sha256)
  throw new Error('Uploaded assets failed verification; the GitHub release remains a draft.');
command('gh', ['release', 'edit', tag, '--repo', config.repository, '--draft=false', '--latest']);
// Existing LAN-only clients can bootstrap into the GitHub-aware release build.
publishRelease({ apk, directory: join(repoRoot, '.build/releases/android'), version, notes });
process.stdout.write(`Published https://github.com/${config.repository}/releases/tag/${tag}\n`);
