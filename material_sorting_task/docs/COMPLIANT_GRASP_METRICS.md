# Compliant-grasp metrics and plots

These tools are passive: they subscribe to feedback and logs, write files, and
never publish a robot command.  The recorded `effort` is MuJoCo joint-actuator
generalized effort from `/joint_states`; it is not fingertip force in newtons.

## Record one three-task run

Start the Server and Client normally.  In a third host terminal, start the
recorder before launching `scripts/run_client.sh`:

```bash
docker exec -it material_sorting_client_v4 bash -lc '
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=101
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd /workspace/baseline
python3 scripts/record_grasp_metrics.py --output /tmp/grasp_metrics.csv
'
```

Leave it running during all three tasks.  After the Client reports
`controller=finished`, stop the recorder with `Ctrl+C`.

## Generate plots

Inside the Client container:

```bash
docker exec -it material_sorting_client_v4 bash -lc '
cd /workspace/baseline
python3 scripts/plot_grasp_metrics.py \
  /tmp/grasp_metrics.csv \
  --output-dir /tmp/grasp_metrics_plots
'
```

Copy the CSV, PNG plots and summaries to the 4090 host:

```bash
docker cp material_sorting_client_v4:/tmp/grasp_metrics.csv \
  /home/abc123/polaris/workspace/grasp_metrics.csv

docker cp material_sorting_client_v4:/tmp/grasp_metrics_plots \
  /home/abc123/polaris/workspace/grasp_metrics_plots
```

The plot directory contains one `taskN_grasp_curves.png` per recorded task,
plus `grasp_summary.csv` and `grasp_summary.json`.  Contact/alignment event
markers come from the Client's throttled progress logs, so their timestamps
have log-period resolution.  Joint angle, velocity, raw effort and independently
filtered effort-delta curves retain the `/joint_states` sample rate.
