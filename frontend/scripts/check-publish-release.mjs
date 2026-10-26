import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { publishRelease, validateVersion } from './publish-release.mjs';
import { githubManifest } from './github-release-manifest.mjs';

test('publication names immutable packages by checksum and advances manifest atomically', () => {
  const temp = mkdtempSync(join(tmpdir(), 'bridgemind-release-'));
  try {
    const apk = join(temp, 'build.apk'), directory = join(temp, 'releases');
    const bytes = Buffer.from('first test APK');
    writeFileSync(apk, bytes);
    const first = publishRelease({ apk, directory, version: { versionCode: 2, versionName: '1.1' }, notes: '新增更新功能' });
    assert.equal(first.sha256, createHash('sha256').update(bytes).digest('hex'));
    assert.deepEqual(readFileSync(join(directory, first.sha256 + '.apk')), bytes);
    assert.deepEqual(JSON.parse(readFileSync(join(directory, 'latest.json'))), first);
    writeFileSync(apk, 'second test APK');
    assert.throws(() => publishRelease({ apk, directory, version: { versionCode: 2, versionName: '1.1' } }), /Increase versionCode/);
    assert.deepEqual(JSON.parse(readFileSync(join(directory, 'latest.json'))), first);
    publishRelease({ apk, directory, version: { versionCode: 3, versionName: '1.2' } });
    assert.deepEqual(readFileSync(join(directory, first.sha256 + '.apk')), bytes);
    assert.throws(() => publishRelease({ apk, directory, version: { versionCode: 2, versionName: '1.1' } }), /Increase versionCode/);
  } finally { rmSync(temp, { recursive: true, force: true }); }
});

test('invalid versions cannot be published', () => {
  for (const versionCode of [0, -1, 2.5, '2', 2100000001])
    assert.throws(() => validateVersion({ versionCode, versionName: '1.1' }));
  assert.throws(() => validateVersion({ versionCode: 2, versionName: '' }));
});

test('GitHub manifest points to a versioned asset and hashes the exact APK bytes', () => {
  const bytes = Buffer.from('signed release fixture');
  const version = { versionCode: 7, versionName: '1.2.2' };
  const manifest = githubManifest({ bytes, version, repository: 'ghhufue/BridgeMind', notes: 'Release' });
  assert.equal(manifest.apkPath, 'https://github.com/ghhufue/BridgeMind/releases/download/v1.2.2/BridgeMind-v1.2.2.apk');
  assert.equal(manifest.sha256, createHash('sha256').update(bytes).digest('hex'));
  assert.equal(manifest.size, bytes.length);
  assert.throws(() => githubManifest({ bytes, version, repository: '../other' }), /repository/);
  assert.throws(() => githubManifest({ bytes, version: { ...version, versionName: '../escape' }, repository: 'ghhufue/BridgeMind' }), /tag/);
});
