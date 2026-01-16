---
title: "Wall-following robot"
excerpt: "Wall following robot that can navigate through a maze<br/><img src='/images/BingdaRobot.png'>"
date: "2025-11-15"
collection: projects
---

# Maze navigation ground robot

This project is done to satisfy the Hands-on Activity of RB1101 at National University of Singapore.

[View on Github](https://github.com/patuan07/rb1101)

## Introduction

This project extends the existing Bingda holonomic robot by enabling it to autonomously navigate a maze using wall-following techniques. Built on ROS 2 Jazzy and tested in Gazebo, the system relies on LiDAR data to perceive its environment. A RANSAC-based line-fitting algorithm is used to detect walls from the LiDAR scans, allowing the robot to estimate its position and adjust its motion accordingly. With these enhancements, the Bingda Robot can follow maze boundaries and move through complex environments with improved accuracy and stability.

## Technical Overview

The system uses LiDAR to enable the Bingda holonomic robot to navigate a maze using wall-following algorithms. LiDAR scan data is first collected and processed in real time through ROS 2 Jazzy, where each frame of range measurements is converted into a 2D point cloud. To identify structural features in the environment, the system applies a RANSAC line-fitting algorithm, which isolates dominant wall segments by iteratively selecting candidate points and determining the best-fit linear model. This allows the robot to detect left, right, and frontal walls even in the presence of noise, reflections, and irregular obstacles.

The detected wall segments are then used to compute the robot’s relative position to the maze's walls. By comparing the perpendicular distance and alignment to the nearest wall, the navigation module determines corrective velocity commands for the holonomic drivetrain. These commands are generated to maintain a stable offset from the wall while ensuring forward progress through the maze.

Simulation and testing are conducted in Gazebo, where a virtual model of the Bingda Robot interacts with configurable maze environments. ROS 2 nodes handle LiDAR simulation, RANSAC processing, command generation, and real-time visualization. Using different environment variables for simulation and real environments, we minimized the changes needed to be done when using the physical robot.

This system demonstrates how geometric perception and lightweight control algorithms can enable reliable autonomous navigation in constrained spaces.

## Instructions ##
1. Extract compressed rb1101 folder to desired directory. The following steps assume you extracted to ~/Documents/ (where ~/ is a shortcut for home/user/)
2. Change directory to extracted folder then colcon build. In terminal, enter the following commands: \
`cd ~/Documents/rb1101` \
`colcon build --symlink-install`

3. Assign permission to run shell scripts in terminal. You only need to do this once on each computer/account. \
`chmod +x *.sh`

4. Start the gazebo sim in a terminal. The following command changes directory to the location of your workspace, then runs the Gazebo shell script. \
`cd ~/Documents/rb1101 && ./gz_heading.sh`
   
5. In **another** terminal, run your wall-following script. \
`cd ~/Documents/rb1101 && ./wall_following.sh`
* The default behaviour is left wall-following. Add your modifications to the timer_callback() function of WallFollowingNode() class, in the file `src/rb1101/wall_following.py`.

6. When running wall-following in real life, remember to set the parameter `is_simulation` to `False` in `wall_following.py`.

## Objectives ##
1. Implement wall-following with heading such that the robot will be able to turn right if there's a corner on the back-right, and allow the robot to strive towards facing heading = 0 (initial heading). Run Gazebo with the following shell script: \
`cd ~/Documents/rb1101 && ./gz_heading.sh`

* If your wall-following with heading works, you should be able to escape the "G" and proceed forward after the G.

2. Implement wall-following with pledge such that the robot turns left or right as needed, and moves forward if the cumulative heading is equal to the initial heading. Run Gazebo with the following shell script: \
`cd ~/Documents/rb1101 && ./gz_pledge.sh` 


## References ##
RANSAC algorithm from wall_following.py uses an adapted version from https://github.com/creminem94/Advanced-Wall-Following/tree/main. All credits for the RANSAC implementation to them.
