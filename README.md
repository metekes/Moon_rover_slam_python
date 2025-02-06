# Moon Rover Cooperative Visual SLAM ## Commands
## Commands:
Launch the sim
```bash
ros2 launch rover_gazebo low_moon.launch.py launch_gui:=0
```
Run vel controller
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Run the dataset (example path: /home/mete/dataset_rosbags/tum_rgbd/fr1_xyz/fr1_xyz.db3)
```bash
ros2 bag play {path}
```

Run SLAM
```bash
colcon build
ros2 run rover_slam_python rover_slam
```

Plot the trajectories (example txt file: positions.txt)
```bash
python3 plot_trajectories.py {txt_file}
```
