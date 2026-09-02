#!/usr/bin/env bash
# One-command local setup — automates README step 2, everything after
# `cdk deploy`:
#
#   1. python deps (pipeline/requirements.txt)
#   2. pinned DreamZero clone (the prep stage shells into its converter)
#   3. training image + config: waits for the CodeBuild image build the deploy
#      kicked off (progress shown; ~1h on a first deploy), then writes
#      pipeline/pipeline_config.json from the stack outputs with the
#      digest-pinned image URI from SSM
#   4. base weights in S3 (check, and stage with --stage-assets)
#
#   ./setup.sh                  do 1-3, then report S3/weights status; if the
#                               image build is still running, report and exit
#                               instead of waiting
#   ./setup.sh --stage-assets   also WAIT for the image build, and stage the
#                               base weights if missing (~128GB download;
#                               needs ~250GB free disk)
#
# Any other argument is passed straight to cdk/generate_pipeline_config.py, so
# --region/--stack/--profile pick which deployment to configure:
#
#   ./setup.sh --region us-east-2 --stage-assets
#
# That matters because boto3 ignores AWS_REGION (see cdk/README.md): without
# --region, a shell set up for `cdk deploy` can configure a DIFFERENT region's
# stack, successfully and silently. The generated file is
# $DREAMZERO_PIPELINE_CONFIG, else pipeline/pipeline_config.json — set that env
# var to keep one config per region.
#
# Idempotent: re-running is always safe. Requires step 1 (cdk deploy) to have
# run first for parts 3-4.
set -uo pipefail
cd "$(dirname "$0")"

# --help must come first and must not do anything. Unrecognised arguments are
# forwarded to generate_pipeline_config.py by design (see above), so without this
# `./setup.sh --help` silently ran the entire four-stage setup — installing
# packages, cloning, and calling AWS — for a reader who only wanted the usage.
case "${1:-}" in
    -h|--help)
        sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
esac

DREAMZERO_URL=https://github.com/dreamzero0/dreamzero.git
# The upstream commit the golden-tested conversion used. prep_dataset.py
# shells into this clone's converter and its output was validated
# byte-for-byte against a reference conversion — silent upstream drift would
# invalidate that, so the clone is pinned. Override: DREAMZERO_PIN=<sha>
DREAMZERO_PIN=${DREAMZERO_PIN:-ab790c198fbce33503358efbbd4187ce9a89adf3}

# --stage-assets is ours; everything else is forwarded to the config generator.
# Kept as a plain string rather than an array so this works on bash 3.2 (macOS);
# the forwarded values are region/stack/profile names, which never contain
# whitespace. --out is rejected because parts 3-4 below read the config back,
# and DREAMZERO_PIPELINE_CONFIG is the setting both halves agree on.
STAGE_ASSETS=0
GENARGS=
for arg in "$@"; do
    case "$arg" in
        --stage-assets) STAGE_ASSETS=1 ;;
        --out|--out=*)
            echo "setup.sh: use DREAMZERO_PIPELINE_CONFIG=<path> instead of --out" >&2
            exit 2 ;;
        *) GENARGS="$GENARGS $arg" ;;
    esac
done

echo "==> [1/4] python dependencies (pipeline/requirements.txt)"
python3 -m pip install --quiet -r pipeline/requirements.txt || exit 1
echo "    ok"

echo "==> [2/4] DreamZero repo (the prep stage shells into its converter)"
if [ -e dreamzero/.git ]; then
    HAVE=$(git -C dreamzero rev-parse HEAD)
    if [ "$HAVE" = "$DREAMZERO_PIN" ]; then
        echo "    ok — already cloned at the pinned commit"
    else
        # Never reset someone's working clone; just say why it differs.
        echo "    WARNING: existing clone is at ${HAVE:0:12}, validated pin is"
        echo "    ${DREAMZERO_PIN:0:12} — leaving it untouched. If prep output"
        echo "    looks wrong, re-clone or: git -C dreamzero checkout $DREAMZERO_PIN"
    fi
else
    git clone --quiet "$DREAMZERO_URL" dreamzero || exit 1
    git -C dreamzero checkout --quiet "$DREAMZERO_PIN" || exit 1
    echo "    ok — cloned at pinned commit ${DREAMZERO_PIN:0:12}"
fi
# prep_dataset.py defaults DREAMZERO_REPO to ./dreamzero, so no export is
# needed. Set DREAMZERO_REPO only if you keep the clone elsewhere.

echo "==> [3/4] training image + pipeline config"
# Same default the generator and pipeline/pipeline_config.py use, so all three
# agree on which file is "the" config even when it is not the standard path.
CFG=${DREAMZERO_PIPELINE_CONFIG:-pipeline/pipeline_config.json}
if [ -f "$CFG" ] && grep -q '"image_uri".*@sha256:' "$CFG"; then
    # a digest-pinned URI only exists after a successful build published it
    echo "    ok — $CFG already carries a digest-pinned image URI"
else
    # Materialize the CDK stack outputs. Works mid-build too (falls back to
    # the configured tag); regenerated below once the build reaches SSM.
    # pipefail is set, so a generate failure still fails the pipe to sed
    if ! python3 cdk/generate_pipeline_config.py --quiet $GENARGS 2>&1 | sed 's/^/    /'; then
        echo "    FAILED to read the CDK stack outputs — deploy the"
        echo "    infrastructure first (README step 1), then re-run ./setup.sh"
        exit 1
    fi
    PROJ=$(python3 -c "import json; print(json.load(open('$CFG')).get('project') or 'dreamzero')")
    # region/profile names never contain whitespace, so a plain string is safe
    AWSARGS=$(python3 -c "
import json; c = json.load(open('$CFG')); a = []
if c.get('region'):  a += ['--region',  c['region']]
if c.get('profile'): a += ['--profile', c['profile']]
print(' '.join(a))")
    BUILD_PROJECT="${PROJ}-image-build"
    WAIT_START=$SECONDS
    while :; do
        BUILD_ID=$(aws codebuild list-builds-for-project --project-name "$BUILD_PROJECT" \
                       --sort-order DESCENDING --query 'ids[0]' --output text \
                       $AWSARGS 2>/dev/null)
        if [ -z "$BUILD_ID" ] || [ "$BUILD_ID" = "None" ]; then
            echo "    no image build found for $BUILD_PROJECT — the deploy normally starts"
            echo "    one automatically; start it by hand and re-run ./setup.sh:"
            echo "      aws codebuild start-build --project-name $BUILD_PROJECT $AWSARGS"
            exit 1
        fi
        set -- $(aws codebuild batch-get-builds --ids "$BUILD_ID" \
                     --query 'builds[0].[buildStatus,currentPhase]' --output text \
                     $AWSARGS 2>/dev/null)
        STATUS=${1:-UNKNOWN}; PHASE=${2:--}
        case "$STATUS" in
            SUCCEEDED)
                echo "    image build succeeded"
                python3 cdk/generate_pipeline_config.py --quiet $GENARGS 2>&1 | sed 's/^/    /' || exit 1
                break ;;
            IN_PROGRESS)
                if [ "$STAGE_ASSETS" = 1 ]; then
                    echo "    image build in progress: $PHASE  (waited $(( (SECONDS - WAIT_START) / 60 ))m; ~1h on a first build; next check in 60s)"
                    sleep 60
                else
                    echo "    image build in progress: $PHASE  (~1h on a first build)"
                    echo "    re-run ./setup.sh when it finishes, or ./setup.sh --stage-assets to wait here"
                    exit 1
                fi ;;
            UNKNOWN)
                echo "    could not read the build status (expired credentials?)"
                if [ "$STAGE_ASSETS" = 1 ]; then sleep 60; else exit 1; fi ;;
            *)
                echo "    image build ended $STATUS — inspect build $BUILD_ID of"
                echo "    $BUILD_PROJECT in the CodeBuild console, fix, and restart it:"
                echo "      aws codebuild start-build --project-name $BUILD_PROJECT $AWSARGS"
                exit 1 ;;
        esac
    done
fi

echo "==> [4/4] base weights in S3"
if python3 pipeline/stage_base_assets.py --check; then
    echo "    ok — all base weights staged"
elif [ "$STAGE_ASSETS" = 1 ]; then
    python3 pipeline/stage_base_assets.py || exit 1
else
    echo "    to stage them (~128GB download, one-time):  ./setup.sh --stage-assets"
    exit 1
fi

echo
echo "setup complete. Next:  python3 pipeline/run_pipeline.py --name <run-name> ..."
