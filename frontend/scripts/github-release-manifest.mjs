import { createHash } from 'node:crypto';
import { validateVersion } from './publish-release.mjs';

export function githubManifest({ bytes, version, repository, notes = '' }) {
  validateVersion(version);
  if (!/^[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+$/.test(repository)) throw new Error('Invalid GitHub repository.');
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(version.versionName)) throw new Error('Version name is not safe for a release tag.');
  if (!bytes.length || bytes.length > 128 * 1024 * 1024) throw new Error('APK must be 1..128 MiB.');
  if (typeof notes !== 'string' || notes.length > 4000) throw new Error('Release notes must be at most 4000 characters.');
  const tag = `v${version.versionName}`, asset = `BridgeMind-${tag}.apk`;
  return {
    protocol: 'bridgemind.update.v1', applicationId: 'com.bridgemind.app',
    versionCode: version.versionCode, versionName: version.versionName, size: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    apkPath: `https://github.com/${repository}/releases/download/${tag}/${asset}`, notes,
  };
}
