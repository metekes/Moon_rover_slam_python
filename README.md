# Moon Rover Cooperative Visual SLAM ## Commands
## Commands:
Launch the sim
```bash
ros2 launch ezrassor_sim_gazebo gazebo_launch.py world:=moon.world gui:=0
```
Run vel controller
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __node:=teleop_node -r cmd_vel:=/cmd_vel
```
Run SLAM
```bash
colcon build
ros2 run rover_slam_python rover_slam
```
