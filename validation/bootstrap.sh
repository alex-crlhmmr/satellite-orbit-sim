#!/usr/bin/env bash
set -euo pipefail

validation_tools_dir="${1:?usage: validation/bootstrap.sh /absolute/tools/directory}"
mkdir -p "$validation_tools_dir"

machine_arch="$(uname -m)"
case "$machine_arch" in
  aarch64|arm64) adoptium_arch="aarch64" ;;
  x86_64|amd64) adoptium_arch="x64" ;;
  *) echo "unsupported architecture: $machine_arch" >&2; exit 2 ;;
esac

jdk_dir="$validation_tools_dir/jdk-21.0.12+8"
if [ ! -x "$jdk_dir/bin/java" ]; then
  curl -fL "https://api.adoptium.net/v3/binary/version/jdk-21.0.12%2B8/linux/${adoptium_arch}/jdk/hotspot/normal/eclipse" -o "$validation_tools_dir/jdk.tar.gz"
  tar -xzf "$validation_tools_dir/jdk.tar.gz" -C "$validation_tools_dir"
fi

maven_dir="$validation_tools_dir/apache-maven-3.9.11"
if [ ! -x "$maven_dir/bin/mvn" ]; then
  curl -fL "https://archive.apache.org/dist/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz" -o "$validation_tools_dir/maven.tar.gz"
  tar -xzf "$validation_tools_dir/maven.tar.gz" -C "$validation_tools_dir"
fi

orekit_revision="baf158744d38ec76cf94e2d396280d545b9f0ba2"
orekit_data_dir="$validation_tools_dir/orekit-data-$orekit_revision"
if [ ! -d "$orekit_data_dir" ]; then
  curl -fL "https://gitlab.orekit.org/orekit/orekit-data/-/archive/${orekit_revision}/orekit-data-${orekit_revision}.zip" -o "$validation_tools_dir/orekit-data.zip"
  unzip -q "$validation_tools_dir/orekit-data.zip" -d "$validation_tools_dir"
fi

gravity_file="$validation_tools_dir/EGM2008.gfc"
if [ ! -f "$gravity_file" ]; then
  curl -fL "https://icgem.gfz-potsdam.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc" -o "$gravity_file"
fi
expected_gravity_sha="ab5b524da073e63b5bdceb7ca47a0de07a26dd44a1c5798f39fc98dc80af70fd"
actual_gravity_sha="$(sha256sum "$gravity_file" | cut -d' ' -f1)"
if [ "$actual_gravity_sha" != "$expected_gravity_sha" ]; then
  echo "EGM2008 checksum mismatch: $actual_gravity_sha" >&2
  exit 3
fi

export JAVA_HOME="$jdk_dir"
export PATH="$JAVA_HOME/bin:$maven_dir/bin:$PATH"
(cd validation/orekit && mvn -q package)

echo "Validation tools ready. Run:"
echo ".venv/bin/python validation/run.py --java '$jdk_dir/bin/java' --orekit-jar validation/orekit/target/orekit-reference-1.0.0.jar --orekit-data '$orekit_data_dir' --gravity-file '$gravity_file'"

