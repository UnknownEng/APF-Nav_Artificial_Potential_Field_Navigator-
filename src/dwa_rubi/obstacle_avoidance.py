#!/usr/bin/env python3
import math
from threading import Lock

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool


class DWALawnmowerFollower3D(object):
def __init__(self):
rospy.init_node('dwa_lawnmower_follower')

self.cfg = {
# Goal tolerance
'goal_tol_xy': rospy.get_param("~goal_tol_xy", 0.5),
'goal_tol_z': rospy.get_param("~goal_tol_z", 0.25),

# Control timing
'control_rate': rospy.get_param("~control_rate", 10.0),
'dt': 1.0 / rospy.get_param("~control_rate", 10.0),

# Motion limits
'inflate': rospy.get_param("~inflate", 0.8),
'v_min': rospy.get_param("~v_min", 0.0),
'v_max': rospy.get_param("~v_max", 0.6),
'w_max': rospy.get_param("~w_max", 1.8),
'vz_min': rospy.get_param("~vz_min", -0.35),
'vz_max': rospy.get_param("~vz_max", 0.35),

'a_max': rospy.get_param("~a_max", 1.0),
'alpha_max': rospy.get_param("~alpha_max", 3.0),
'az_max': rospy.get_param("~az_max", 0.8),

# Vertical hold gain
'kz': rospy.get_param("~kz", 0.8),

# DWA horizon
'sim_dt': rospy.get_param("~sim_dt", 0.08),
'pred_time_open': rospy.get_param("~pred_time_open", 1.8),
'pred_time_tight': rospy.get_param("~pred_time_tight", 1.2),

# Sampling
'Nv_base': rospy.get_param("~Nv_base", 9),
'Nw_base': rospy.get_param("~Nw_base", 25),
'Nz_base': rospy.get_param("~Nz_base", 7),

# Carrot lookahead
'L_open': rospy.get_param("~L_open", 1.2),
'L_tight': rospy.get_param("~L_tight", 0.6),

# Obstacle processing
'obs_z_window': rospy.get_param("~obs_z_window", 3.0),
'obs_range_max': rospy.get_param("~obs_range_max", 10.0),
'cluster_cell_xy': rospy.get_param("~cluster_cell_xy", 0.35),
'cluster_cell_z': rospy.get_param("~cluster_cell_z", 0.35),
'default_obs_r': rospy.get_param("~default_obs_r", 0.45),
'max_points': rospy.get_param("~max_points", 3000),

# Timeouts
'data_timeout': rospy.get_param("~data_timeout", 2.0),
'pc_timeout': rospy.get_param("~pc_timeout", 1.0),
'waypoint_handoff_time': rospy.get_param("~waypoint_handoff_time", 0.2),

# Safety
'critical_clearance': rospy.get_param("~critical_clearance", 0.8),
'stop_clearance': rospy.get_param("~stop_clearance", 0.4),

# Vertical scoring
'z_score_weight': rospy.get_param("~z_score_weight", 2.0),
'z_progress_weight': rospy.get_param("~z_progress_weight", 1.5),
'z_speed_weight': rospy.get_param("~z_speed_weight", 0.2),
'max_goal_z_error_for_level_flight': rospy.get_param(
"~max_goal_z_error_for_level_flight", 0.15
),
}

self.state = {
'p': np.array([0.0, 0.0, 0.0], dtype=float),
'yaw': 0.0,
'v_xy': 0.0,
'vz': 0.0,
'wz': 0.0,
'stamp': rospy.Time.now()
}

self.goal = np.array([0.0, 0.0, 0.0], dtype=float)
self.goal_stamp = rospy.Time.now()

# obstacles: [x, y, z, r]
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time(0)

self.survey_done = False
self.have_odom = False
self.have_goal = False

self.state_lock = Lock()
self.goal_lock = Lock()
self.obs_lock = Lock()

self.waypoint_pause_until = rospy.Time.now()
self.last_waypoint = None

odom_topic = rospy.get_param("~odom_topic", "/ground_truth/state")
waypoint_topic = rospy.get_param("~waypoint_topic", "/dwa/goal")
survey_topic = rospy.get_param("~survey_topic", "/survey_done")
pc_topic = rospy.get_param("~pointcloud_topic", "/rtabmap/cloud_map")
cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
rospy.Subscriber(waypoint_topic, PoseStamped, self.waypoint_callback, queue_size=1)
rospy.Subscriber(survey_topic, Bool, self.survey_callback, queue_size=1)
rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback, queue_size=1)

self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
rospy.on_shutdown(self.safe_stop)

rospy.loginfo("Waiting for first odometry on %s...", odom_topic)
rospy.wait_for_message(odom_topic, Odometry, timeout=10.0)

rospy.loginfo("Waiting for first waypoint on %s...", waypoint_topic)
rospy.wait_for_message(waypoint_topic, PoseStamped, timeout=10.0)

rospy.loginfo("DWA lawnmower follower ready. Publishing velocity to %s", cmd_topic)
self.rate = rospy.Rate(self.cfg['control_rate'])

def odom_callback(self, msg):
with self.state_lock:
px = msg.pose.pose.position.x
py = msg.pose.pose.position.y
pz = msg.pose.pose.position.z

qx = msg.pose.pose.orientation.x
qy = msg.pose.pose.orientation.y
qz = msg.pose.pose.orientation.z
qw = msg.pose.pose.orientation.w

yaw = math.atan2(
2.0 * (qw * qz + qx * qy),
1.0 - 2.0 * (qy * qy + qz * qz)
)

vx = msg.twist.twist.linear.x
vy = msg.twist.twist.linear.y
vz = msg.twist.twist.linear.z
wz = msg.twist.twist.angular.z

self.state['p'] = np.array([px, py, pz], dtype=float)
self.state['yaw'] = yaw
self.state['v_xy'] = math.hypot(vx, vy)
self.state['vz'] = vz
self.state['wz'] = wz
self.state['stamp'] = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
self.have_odom = True

def waypoint_callback(self, msg):
with self.goal_lock:
new_goal = np.array([
msg.pose.position.x,
msg.pose.position.y,
msg.pose.position.z
], dtype=float)

if self.last_waypoint is None or not np.allclose(new_goal, self.last_waypoint, atol=1e-3):
self.waypoint_pause_until = rospy.Time.now() + rospy.Duration(self.cfg['waypoint_handoff_time'])
self.last_waypoint = new_goal.copy()

self.goal = new_goal
self.goal_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
self.have_goal = True

def survey_callback(self, msg):
self.survey_done = msg.data

def pc_callback(self, msg):
points = []
try:
for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
points.append([p[0], p[1], p[2]])
except Exception as e:
rospy.logwarn_throttle(5.0, "Failed to read point cloud: %s", str(e))
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

if not points:
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

xyz = np.array(points, dtype=float)

with self.state_lock:
current_pos = self.state['p'].copy()
current_yaw = self.state['yaw']

# Height filter
z_diff = np.abs(xyz[:, 2] - current_pos[2])
xyz = xyz[z_diff <= self.cfg['obs_z_window'] / 2.0]
if len(xyz) == 0:
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

# Range filter
dx = xyz[:, 0] - current_pos[0]
dy = xyz[:, 1] - current_pos[1]
dist = np.sqrt(dx ** 2 + dy ** 2)
xyz = xyz[dist <= self.cfg['obs_range_max']]
if len(xyz) == 0:
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

# Prefer front points, but keep nearby side points
dx = xyz[:, 0] - current_pos[0]
dy = xyz[:, 1] - current_pos[1]
dist = np.sqrt(dx ** 2 + dy ** 2)
forward = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
side_keep = dist < 2.0
xyz = xyz[(forward > -0.5) | side_keep]
if len(xyz) == 0:
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

if len(xyz) > self.cfg['max_points']:
idx = np.random.choice(len(xyz), self.cfg['max_points'], replace=False)
xyz = xyz[idx]

if len(xyz) < 3:
with self.obs_lock:
self.obstacles = np.zeros((0, 4), dtype=float)
self.obstacles_stamp = rospy.Time.now()
return

# 3D voxel clustering
gx = np.round(xyz[:, 0] / self.cfg['cluster_cell_xy']) * self.cfg['cluster_cell_xy']
gy = np.round(xyz[:, 1] / self.cfg['cluster_cell_xy']) * self.cfg['cluster_cell_xy']
gz = np.round(xyz[:, 2] / self.cfg['cluster_cell_z']) * self.cfg['cluster_cell_z']
cells = np.unique(np.column_stack([gx, gy, gz]), axis=0)

with self.obs_lock:
self.obstacles = np.column_stack([
cells,
self.cfg['default_obs_r'] * np.ones(len(cells))
])
self.obstacles_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

def is_data_fresh(self):
now = rospy.Time.now()
timeout = rospy.Duration(self.cfg['data_timeout'])
pc_timeout = rospy.Duration(self.cfg['pc_timeout'])

with self.state_lock:
odom_fresh = (now - self.state['stamp']) < timeout

with self.goal_lock:
goal_fresh = (now - self.goal_stamp) < timeout

with self.obs_lock:
# obstacle data may be empty and still valid
pc_fresh = (len(self.obstacles) == 0) or ((now - self.obstacles_stamp) < pc_timeout)

return odom_fresh and goal_fresh and pc_fresh

def min_clearance_3d(self, p3, obstacles=None):
if obstacles is None:
with self.obs_lock:
obstacles = self.obstacles.copy()

if len(obstacles) == 0:
return float('inf')

d = float('inf')
for obs in obstacles:
center = obs[:3]
r = obs[3]
d = min(d, np.linalg.norm(p3 - center) - (r + self.cfg['inflate']))
return d

def hold_altitude(self, z, z_goal):
err = z_goal - z
return max(min(self.cfg['kz'] * err, self.cfg['vz_max']), self.cfg['vz_min'])

def path_progress(self, p, path, path_dists):
best_d2 = float('inf')
best_s = 0.0
for i in range(len(path) - 1):
a = path[i]
b = path[i + 1]
ab = b - a
L2 = np.dot(ab, ab)
if L2 < 1e-12:
continue
t = np.dot(p - a, ab) / L2
t = max(0.0, min(1.0, t))
proj = a + t * ab
d2 = np.sum((p - proj) ** 2)
if d2 < best_d2:
best_d2 = d2
best_s = path_dists[i] + t * np.sqrt(L2)
return best_s

def point_at_arc(self, path, path_dists, s_query):
s_query = max(0.0, min(s_query, path_dists[-1]))
for i in range(len(path_dists) - 1):
if path_dists[i] <= s_query <= path_dists[i + 1]:
seg = (s_query - path_dists[i]) / (path_dists[i + 1] - path_dists[i] + 1e-12)
return path[i] + seg * (path[i + 1] - path[i])
return path[-1]

def rollout_3d(self, p0_xy, z0, psi0, v, vz, w, T, dt, obstacles):
N = max(1, int(round(T / dt)))
p_xy = p0_xy.copy()
z = float(z0)
psi = psi0
minclr = float('inf')

for _ in range(N):
psi = self.wrap_to_pi(psi + w * dt)
p_xy = p_xy + np.array([v * np.cos(psi), v * np.sin(psi)]) * dt
z = z + vz * dt

p3 = np.array([p_xy[0], p_xy[1], z], dtype=float)
if len(obstacles) > 0:
for obs in obstacles:
d = np.linalg.norm(p3 - obs[:3]) - (obs[3] + self.cfg['inflate'])
minclr = min(minclr, d)

if minclr <= 0:
break

if minclr == float('inf'):
minclr = 5.0

return minclr, p_xy, z, psi

def compute_cmd_dwa_3d(self, pose, global_path, path_s, goal3):
p_xy = pose['p']
z = pose['z']
psi = pose['psi']
v = pose['v']
vz = pose['vz']
w = pose['w']

with self.obs_lock:
obstacles = self.obstacles.copy()

clear_now = self.min_clearance_3d(np.array([p_xy[0], p_xy[1], z]), obstacles)

if clear_now < self.cfg['stop_clearance']:
rospy.logwarn_throttle(1.0, "Emergency stop. 3D clearance: %.2f", clear_now)
return 0.0, 0.0, 0.0

if clear_now < self.cfg['critical_clearance']:
L = self.cfg['L_tight']
predT = self.cfg['pred_time_tight']
Nv = max(self.cfg['Nv_base'], 9)
Nw = max(self.cfg['Nw_base'], 31)
Nz = max(self.cfg['Nz_base'], 9)
speed_limit = 0.25
elif clear_now < 1.0:
L = self.cfg['L_tight']
predT = self.cfg['pred_time_tight']
Nv = max(self.cfg['Nv_base'], 11)
Nw = max(self.cfg['Nw_base'], 25)
Nz = max(self.cfg['Nz_base'], 7)
speed_limit = self.cfg['v_max']
else:
L = self.cfg['L_open']
predT = self.cfg['pred_time_open']
Nv = self.cfg['Nv_base']
Nw = self.cfg['Nw_base']
Nz = self.cfg['Nz_base']
speed_limit = self.cfg['v_max']

s_now = self.path_progress(p_xy, global_path, path_s)
carrot = self.point_at_arc(global_path, path_s, s_now + L)

# Dynamic window
v_lo = max(self.cfg['v_min'], v - self.cfg['a_max'] * self.cfg['dt'])
v_hi = min(self.cfg['v_max'], v + self.cfg['a_max'] * self.cfg['dt'])
v_hi = min(v_hi, speed_limit)

vz_lo = max(self.cfg['vz_min'], vz - self.cfg['az_max'] * self.cfg['dt'])
vz_hi = min(self.cfg['vz_max'], vz + self.cfg['az_max'] * self.cfg['dt'])

w_lo = max(-self.cfg['w_max'], w - self.cfg['alpha_max'] * self.cfg['dt'])
w_hi = min(self.cfg['w_max'], w + self.cfg['alpha_max'] * self.cfg['dt'])

z_goal_err = goal3[2] - z
if abs(z_goal_err) < self.cfg['max_goal_z_error_for_level_flight']:
vz_lo = min(vz_lo, 0.0)
vz_hi = max(vz_hi, 0.0)

v_hi = max(v_hi, 0.20)
v_lo = min(v_lo, 0.05)

v_s = np.linspace(v_lo, v_hi, Nv)
vz_s = np.linspace(vz_lo, vz_hi, Nz)
w_s = np.linspace(w_lo, w_hi, Nw)

best_score = -float('inf')
v_best = 0.0
vz_best = 0.0
w_best = 0.0

for vi in v_s:
for vzi in vz_s:
for wi in w_s:
minclr, p_end_xy, z_end, psi_end = self.rollout_3d(
p_xy, z, psi, vi, vzi, wi, predT, self.cfg['sim_dt'], obstacles
)

if minclr <= 0:
continue

s_clear = min(minclr / 2.0, 1.0)

angC = math.atan2(carrot[1] - p_end_xy[1], carrot[0] - p_end_xy[0])
e_head = abs(self.wrap_to_pi(angC - psi_end))
s_head = 1.0 - e_head / math.pi

dC = np.linalg.norm(carrot - p_end_xy)
s_dist = 1.0 / (1.0 + dC)
s_speed = (vi - self.cfg['v_min']) / (self.cfg['v_max'] - self.cfg['v_min'] + 1e-6)
s_goal_xy = 1.0 / (1.0 + np.linalg.norm(goal3[:2] - p_end_xy))

z_err_end = abs(goal3[2] - z_end)
s_goal_z = 1.0 / (1.0 + z_err_end)
s_zspeed = 1.0 - min(abs(vzi) / max(abs(self.cfg['vz_max']), 1e-6), 1.0)

s_end = self.path_progress(p_end_xy, global_path, path_s)
adv = s_end - s_now
s_prog = max(-1.0, min(1.0, adv / 0.30))

z_prog = abs(z_goal_err) - z_err_end
s_zprog = max(-1.0, min(1.0, z_prog / 0.30))

rot_pen = abs(wi) / (self.cfg['w_max'] + 1e-6)

if clear_now < self.cfg['critical_clearance']:
w_clear = 7.0
w_speed = 0.05
elif clear_now < 1.0:
w_clear = 5.0
w_speed = 0.15
else:
w_clear = 2.5
w_speed = 0.8

score = (
2.0 * s_head +
1.6 * s_dist +
w_clear * s_clear +
w_speed * s_speed +
1.2 * s_goal_xy +
4.0 * s_prog +
self.cfg['z_score_weight'] * s_goal_z +
self.cfg['z_progress_weight'] * s_zprog +
self.cfg['z_speed_weight'] * s_zspeed -
0.25 * rot_pen
)

if vi < 0.08:
score -= 1.5

if score > best_score:
best_score = score
v_best = vi
vz_best = vzi
w_best = wi

if not math.isfinite(best_score):
return self.fallback_turn_creep_3d(p_xy, z, psi, goal3, obstacles)

return v_best, vz_best, w_best

def fallback_turn_creep_3d(self, p_xy, z, psi, goal3, obstacles=None):
angC = math.atan2(goal3[1] - p_xy[1], goal3[0] - p_xy[0])
e = self.wrap_to_pi(angC - psi)

w_cmd = max(min(2.0 * e, self.cfg['w_max']), -self.cfg['w_max'])
v_cmd = 0.08
vz_cmd = self.hold_altitude(z, goal3[2])

if self.min_clearance_3d(np.array([p_xy[0], p_xy[1], z]), obstacles) < self.cfg['stop_clearance']:
v_cmd = 0.0
w_cmd = 0.8 * (1.0 if e >= 0.0 else -1.0)

z_err = goal3[2] - z
vz_cmd = max(min(0.25 * np.sign(z_err + 1e-6), self.cfg['vz_max']), self.cfg['vz_min'])

return v_cmd, vz_cmd, w_cmd

@staticmethod
def wrap_to_pi(angle):
return (angle + math.pi) % (2.0 * math.pi) - math.pi

def safe_stop(self):
try:
cmd = Twist()
cmd.linear.x = 0.0
cmd.linear.y = 0.0
cmd.linear.z = 0.0
cmd.angular.z = 0.0
self.cmd_pub.publish(cmd)
except Exception:
pass

def run(self):
while not rospy.is_shutdown():
now = rospy.Time.now()

if not self.have_odom or not self.have_goal:
self.rate.sleep()
continue

if not self.is_data_fresh():
rospy.logwarn_throttle(5.0, "Data stale - stopping")
self.safe_stop()
self.rate.sleep()
continue

if self.survey_done:
rospy.loginfo("Survey complete")
self.safe_stop()
break

with self.state_lock:
current_state = {
'p': self.state['p'].copy(),
'yaw': self.state['yaw'],
'v_xy': self.state['v_xy'],
'vz': self.state['vz'],
'wz': self.state['wz']
}

with self.goal_lock:
current_goal = self.goal.copy()

dist_xy = np.linalg.norm(current_state['p'][:2] - current_goal[:2])
dist_z = abs(current_state['p'][2] - current_goal[2])

if now < self.waypoint_pause_until:
cmd = Twist()
cmd.linear.x = 0.0
cmd.linear.y = 0.0
cmd.linear.z = self.hold_altitude(current_state['p'][2], current_goal[2])
cmd.angular.z = 0.0
self.cmd_pub.publish(cmd)
self.rate.sleep()
continue

if dist_xy < self.cfg['goal_tol_xy'] and dist_z < self.cfg['goal_tol_z']:
cmd = Twist()
cmd.linear.x = 0.0
cmd.linear.y = 0.0
cmd.linear.z = self.hold_altitude(current_state['p'][2], current_goal[2])
cmd.angular.z = 0.0
self.cmd_pub.publish(cmd)
self.rate.sleep()
continue

global_path = np.array([current_state['p'][:2], current_goal[:2]], dtype=float)
path_s = np.array([0.0, max(dist_xy, 1e-6)], dtype=float)

pose = {
'p': current_state['p'][:2],
'z': current_state['p'][2],
'psi': current_state['yaw'],
'v': current_state['v_xy'],
'vz': current_state['vz'],
'w': current_state['wz']
}

v_cmd, vz_cmd, w_cmd = self.compute_cmd_dwa_3d(
pose, global_path, path_s, current_goal
)

cmd = Twist()
cmd.linear.x = v_cmd
cmd.linear.y = 0.0
cmd.linear.z = vz_cmd
cmd.angular.z = w_cmd
self.cmd_pub.publish(cmd)

rospy.loginfo_throttle(
1.0,
"cmd: vx=%.2f vz=%.2f wz=%.2f | dxy=%.2f dz=%.2f",
v_cmd, vz_cmd, w_cmd, dist_xy, dist_z
)

self.rate.sleep()

self.safe_stop()


if __name__ == '__main__':
try:
node = DWALawnmowerFollower3D()
node.run()
except rospy.ROSInterruptException:
pass


