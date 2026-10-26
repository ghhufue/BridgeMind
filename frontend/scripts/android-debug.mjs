import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { createHash } from 'node:crypto';
import { publishRelease, validateVersion } from './publish-release.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const android = join(root, 'android');
const windows = process.platform === 'win32';
const releaseBuild = process.argv.includes('--release');
const version = JSON.parse(readFileSync(join(root, 'version.json'), 'utf8'));
validateVersion(version);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}`);
}
function capture(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', ...options });
  if (result.error || result.status !== 0) throw new Error(`Cannot run ${command}. Install Android SDK / JDK 21 or configure the environment.`);
  return `${result.stdout || ''}\n${result.stderr || ''}`;
}

const sourceRevision = capture('git', ['rev-parse', 'HEAD'], { cwd: root }).trim();
const sourceDirty = !!capture('git', ['status', '--porcelain'], { cwd: root }).trim();
const adbPath = capture(windows ? 'where.exe' : 'which', ['adb']).trim().split(/\r?\n/)[0];
const sdk = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT || dirname(dirname(adbPath));
const javaHome = process.env.JAVA_HOME || capture('java', ['-XshowSettings:properties', '-version']).match(/java.home\s*=\s*(.+)/)?.[1].trim();
if (!existsSync(join(sdk, 'platforms', 'android-36')) || !javaHome) throw new Error('Android SDK 36 and JDK 21 are required.');

const env = { ...process.env, ANDROID_HOME: sdk, JAVA_HOME: javaHome, BRIDGEMIND_ANDROID_DEBUG: '1' };
if (releaseBuild) {
  const config = JSON.parse(readFileSync(join(root, 'release.config.json'), 'utf8'));
  env.VITE_UPDATE_MANIFEST_URL = config.manifestUrl;
  // Existing installations keep their saved LAN address; new installs configure their own server.
  env.VITE_ANALYSIS_SERVER_URL = process.env.VITE_ANALYSIS_SERVER_URL || '';
  if (process.argv.includes('--compat-signing')) {
    env.BRIDGEMIND_KEYSTORE_PATH = join(homedir(), '.android', 'debug.keystore');
    env.BRIDGEMIND_KEYSTORE_PASSWORD = 'android';
    env.BRIDGEMIND_KEY_ALIAS = 'androiddebugkey';
    env.BRIDGEMIND_KEY_PASSWORD = 'android';
  }
  for (const key of ['BRIDGEMIND_KEYSTORE_PATH', 'BRIDGEMIND_KEYSTORE_PASSWORD', 'BRIDGEMIND_KEY_ALIAS', 'BRIDGEMIND_KEY_PASSWORD']) {
    if (!env[key]) throw new Error(`Configure ${key} or use --compat-signing to retain the existing local app certificate.`);
  }
  if (!existsSync(env.BRIDGEMIND_KEYSTORE_PATH)) throw new Error('Signing keystore not found. Keep it outside the repository.');
}
writeFileSync(join(android, 'local.properties'), `sdk.dir=${sdk.replace(/\\/g, '/').replace(/:/g, '\\:')}\n`);
const npm = process.env.npm_execpath;
if (!npm) throw new Error('Run this script through npm run android:build or npm run android:install.');
run(process.execPath, [npm, 'run', 'build'], { env });
run(process.execPath, [join(root, 'node_modules/@capacitor/cli/bin/capacitor'), 'sync', 'android'], { env });
const variant = releaseBuild ? 'release' : 'debug';
const task = releaseBuild ? 'assembleRelease' : 'assembleDebug';
if (windows) run('cmd.exe', ['/d', '/s', '/c', `gradlew.bat ${task} --console=plain`], { cwd: android, env });
else run(join(android, 'gradlew'), [task, '--console=plain'], { cwd: android, env });

const apk = resolve(root, `../.build/frontend/BridgeMind-${variant}.apk`);
mkdirSync(dirname(apk), { recursive: true });
copyFileSync(join(android, `app/build/outputs/apk/${variant}/app-${variant}.apk`), apk);
const buildMetadata = {
  variant, ...version,
  sourceRevision,
  sourceDirty: sourceDirty || !!capture('git', ['status', '--porcelain'], { cwd: root }).trim(),
  sha256: createHash('sha256').update(readFileSync(apk)).digest('hex'),
};
process.stdout.write(`\n${releaseBuild ? 'Release' : 'Debug'} APK: ${apk}\n`);

if (releaseBuild || process.argv.includes('--publish')) {
  const toolVersion = readdirSync(join(sdk, 'build-tools')).filter(name => /^\d+\.\d+\.\d+$/.test(name))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).at(-1);
  if (!toolVersion) throw new Error('Android SDK build-tools are required to publish APKs.');
  const buildTools = join(sdk, 'build-tools', toolVersion);
  const badging = capture(join(buildTools, windows ? 'aapt.exe' : 'aapt'), ['dump', 'badging', apk]);
  const match = badging.match(/package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'/);
  if (!match || match[1] !== 'com.bridgemind.app' || Number(match[2]) !== version.versionCode || match[3] !== version.versionName)
    throw new Error('Built APK does not match version.json / applicationId.');
  if (releaseBuild && badging.includes('application-debuggable')) throw new Error('Release APK must not be debuggable.');
  run(join(javaHome, 'bin', windows ? 'java.exe' : 'java'), ['-jar', join(buildTools, 'lib/apksigner.jar'), 'verify', apk], { env });
}
if (sourceRevision !== capture('git', ['rev-parse', 'HEAD'], { cwd: root }).trim()) throw new Error('HEAD changed during the build; rebuild before publishing.');
writeFileSync(resolve(root, `../.build/frontend/BridgeMind-${variant}.build.json`), JSON.stringify(buildMetadata, null, 2));
if (process.argv.includes('--publish')) {
  const notesFile = process.argv.find(arg => arg.startsWith('--notes-file='))?.slice('--notes-file='.length);
  const notes = notesFile ? readFileSync(resolve(root, notesFile), 'utf8') : '';
  const directory = resolve(root, '../.build/releases/android');
  const release = publishRelease({ apk, directory, version, notes });
  process.stdout.write(`Published Android ${release.versionName} (${release.versionCode}): ${directory}\n`);
}

if (process.argv.includes('--install')) {
  const devices = capture(adbPath, ['devices']).split(/\r?\n/).map(line => line.match(/^(\S+)\s+device$/)?.[1]).filter(Boolean);
  const serial = process.argv.find(arg => arg.startsWith('--serial='))?.slice('--serial='.length) || (devices.length === 1 ? devices[0] : null);
  if (!serial || !devices.includes(serial)) throw new Error('Connect and authorize one Android phone, or pass --serial=DEVICE.');
  run(adbPath, ['-s', serial, 'install', '-r', apk]);
  run(adbPath, ['-s', serial, 'shell', 'am', 'start', '-W', '-n', 'com.bridgemind.app/.MainActivity']);
}
