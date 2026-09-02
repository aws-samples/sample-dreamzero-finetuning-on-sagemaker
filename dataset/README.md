# dataset/ — your local datasets live here

1. Put your LeRobot dataset in this folder, one subfolder per dataset:

   ```text
   dataset/
     my_robot/          <- your LeRobot v2.1 dataset
       data/
       videos/
       meta/
   ```

   **LeRobot v3 dataset?** Convert it to v2.1 first (local, no AWS needed —
   the shipped converter handles the OpenArms v3 layout; adapt it for
   others):

   ```bash
   python3 pipeline/convert_lerobot_v3_to_v21.py dataset/my_robot_v3 dataset/my_robot
   ```

2. Create an embodiment config for your robot (cameras, state/action layout —
   see *Bring your own dataset* in the root README) by copying a shipped
   example in `pipeline/configs/` and editing it:

   ```bash
   cp pipeline/configs/aloha_bimanual_14dim.yaml pipeline/configs/my_robot.yaml
   ```

3. Point `project_config.json` (repo root) at both:

   ```json
   "dataset": {
     "source": "local",
     "local_path": "dataset/my_robot"
   },
   "embodiment_config": "configs/my_robot.yaml"
   ```

   (`dataset/…` resolves against the repo root; `configs/…` against
   `pipeline/`.)

4. Run the pipeline:

   ```bash
   python3 pipeline/run_pipeline.py --name my-robot-v1
   ```

`--name` is a label for the run — it names the S3 prefixes
(`datasets/<name>/`, `models/<name>-merged/`) and the SageMaker jobs. It does
not have to match your dataset folder.

Note: with the default `source: "huggingface"`, the fetch stage downloads the
HuggingFace dataset into `dataset/<name>/` — pick `--name` values that don't
collide with folders you created yourself.

Everything here except this README is **gitignored**: datasets are data, not
code, and are never committed or published with the sample.
