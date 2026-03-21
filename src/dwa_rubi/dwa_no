<?xml version="1.0"?>
<launch>
  <!-- DWA Controller parameters -->
  <node name="dwa_inspection" pkg="my_room_world" type="dwa_inspection.py" output="screen">
    <!-- Timing -->
    <param name="start_delay" value="5.0"/>
    <param name="control_rate" value="10.0"/>
   
    <!-- Goal tolerances -->
    <param name="goal_tol_xy" value="0.5"/>
    <param name="goal_tol_z" value="0.25"/>
   
    <!-- Motion limits -->
    <param name="v_max" value="0.8"/>
    <param name="v_min" value="0.0"/>
    <param name="w_max" value="1.5"/>
    <param name="vz_max" value="0.3"/>
    <param name="vz_min" value="-0.3"/>
   
    <!-- Acceleration limits -->
    <param name="a_max" value="0.5"/>
    <param name="alpha_max" value="2.0"/>
    <param name="az_max" value="0.2"/>
   
    <!-- DWA sampling -->
    <param name="pred_time" value="2.0"/>
    <param name="sim_dt" value="0.1"/>
    <param name="Nv" value="11"/>
    <param name="Nw" value="21"/>
    <param name="Nz" value="7"/>
   
    <!-- Obstacle parameters -->
    <param name="obstacle_radius" value="0.45"/>
    <param name="safety_margin" value="0.8"/>
  </node>
</launch>
