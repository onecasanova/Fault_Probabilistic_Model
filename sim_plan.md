# AUV Fault Diagnosis Simulation Plan (ROS + UUV Simulator + DBN)

## Overview

This document outlines a plan to integrate a probabilistic fault diagnosis model (DBN / Bayesian Network) with a simulated AUV using ROS and UUV Simulator (LAUV model).

The goal is to perform passive fault diagnosis:
- Run a normal AUV simulation
- Subscribe to simulated sensor data
- Compute probabilities of component faults (battery, thrusters, etc.)
- Validate the model using fault injection scenarios

## Architecture
```
UUV Simulator (LAUV)
    ├── publishes sensor topics (/imu, /odom, /thrusters, etc.)
    ↓
Fault Diagnosis Node (Python, ROS)
    ├── subscribes to sensor topics
    ├── computes features / residuals
    ├── runs DBN inference
    └── publishes fault probabilities
```
## Tools and Frameworks

Core Stack:
- ROS (ROS1 Noetic recommended)
- Gazebo
- UUV Simulator
- LAUV model
- Python
- pgmpy

## Simulation Platform

UUV Simulator:
- Hydrodynamics
- Thruster models
- Sensor simulation

LAUV Model:
- Prebuilt AUV model
- Includes URDF + launch files

## Implementation Plan

1. Install ROS + Gazebo + UUV Simulator + LAUV
2. Run baseline simulation
3. Inspect topics using:
   rostopic list
   rostopic echo /imu
4. Build Python ROS node:
   - subscribe to sensors
   - compute features
   - run DBN
   - publish fault probabilities
5. Run passive diagnosis

## Test Case

Test 1: Baseline
- No faults
- Expect low probabilities

Test 2: Thruster Fault
- Reduce thrust or disable thruster
- Expect increase in fault probability

Test 3: Battery Fault (optional)
- Simulate power anomaly
- Expect increase in battery failure probability

## Output Metrics

- Probability vs time plots
- Compare baseline vs fault

## Notes for Codex

- Use rospy
- Keep modular structure
- Start simple (IMU + thrust)
- Add logging

## Goal

Demonstrate probabilistic fault detection using simulated AUV sensor data.
