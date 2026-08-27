# Client Architecture

The formal Client is organized around one long-lived ROS 2 process and one
perception process.

```text
/material/instruction ─┐
/referee/*             ├─> client_task.py ─> task state machine
/odom, /joint_states   ┘         │
                                 ├─> navigation/
/material/detections ────────────┼─> grasp and placement planning
                                 └─> ROS 2 command topics
```

`client_task.py` owns process lifecycle and task transitions. Pure logic stays
in importable modules so it can be tested without ROS 2. The Client must remain
alive for the complete Server session and must stop the base on every error.

The `reference/` directory is not imported by the formal Client.
