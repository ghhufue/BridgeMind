import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export function validateVersion(version) {
  if (!Number.isSafeInteger(version.versionCode) || version.versionCode < 1 || version.versionCode > 2100000000 ||
      typeof version.versionName !== 'string' || !version.versionName.trim() || version.versionName.length > 80)
    throw new Error('version.json requires versionCode 1..2100000000 and a nonempty versionName (max 80 characters).');
}

export function publishRelease({ apk, directory, version, notes = '' }) {
  validateVersion(version);
  if (typeof notes !== 'string' || notes.length > 4000) throw new Error('Release notes must be at most 4000 characters.');
  const bytes = readFileSync(apk);
  if (!bytes.length || bytes.length > 128 * 1024 * 1024) throw new Error('APK must be 1..128 MiB.');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const manifest = {
    protocol: 'bridgemind.update.v1', applicationId: 'com.bridgemind.app',
    ...version, size: bytes.length, sha256, apkPath: `/updates/android/${sha256}.apk`, notes,
  };
  const latest = join(directory, 'latest.json');
  if (existsSync(latest)) {
    const previous = JSON.parse(readFileSync(latest, 'utf8'));
    if (version.versionCode < previous.versionCode ||
        (version.versionCode === previous.versionCode && sha256 !== previous.sha256))
      throw new Error('Increase versionCode in frontend/version.json before publishing a changed APK.');
  }
  mkdirSync(directory, { recursive: true });
  const packageFile = join(directory, `${sha256}.apk`);
  if (!existsSync(packageFile)) {
    copyFileSync(apk, `${packageFile}.tmp`);
    renameSync(`${packageFile}.tmp`, packageFile);
  } else if (createHash('sha256').update(readFileSync(packageFile)).digest('hex') !== sha256) {
    throw new Error('Published package checksum mismatch.');
  }
  // Readers see either the previous complete release or the next complete release.
  writeFileSync(`${latest}.tmp`, JSON.stringify(manifest, null, 2) + '\n');
  renameSync(`${latest}.tmp`, latest);
  return manifest;
}
