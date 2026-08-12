# Latest official Docker interface audit

Verified against the local `material_sorting:offline-server` and
`material_sorting:offline-client` images on 2026-08-12.

The client consumes these official feedback topics:

- `/material/instruction` (`std_msgs/String`)
- `/referee/gameinfo` (`std_msgs/String`)
- `/referee/taskinfo` (`std_msgs/String`)
- `/referee/score` (`std_msgs/Int32`)
- `/joint_states` (`sensor_msgs/JointState`)

`/joint_states` contains 17 named joints. Its `effort` field is populated from
the simulator force sensors and includes `slide_joint`, six left-arm joints,
and six right-arm joints used by compliant placement.

The latest Server does **not** publish `/material/place_confirmed`,
`/material/grasp_confirmed`, or `/material/unsafe_collision`. Placement success
is authoritative only when the current task's parsed `/referee/gameinfo`
reports `step=place`. The referee sets that step after the object is no longer
gripped, its speed is below the settle threshold, and its position satisfies
the instructed placement region.

The implemented placement sequence is therefore:

1. hold still for 0.40 s and estimate the 13-axis placement effort baseline;
2. lower the slide in 2 mm settled increments;
3. stop on stable bilateral effort contact, or use the bounded geometric
   target if effort is absent;
4. hold support contact for 0.40 s and open each arm by 0.040 m from the held
   width;
5. require the official `step=place` result before completing placement;
6. emit structured `[TIMING]` records with count, mean and nearest-rank P95.
