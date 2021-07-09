import numpy as np
import matplotlib.pyplot as plt
import math
import shelve
import statistics

def disparity_to_point(x,y,disp):
    
    vec_tmp = np.array([x,y,disp,1.0])
    vec_tmp = np.reshape(vec_tmp, (4,1))

    globQ = np.array([
                        np.array([1, 0, 0, -525.1669464111328]),
                        np.array([0, 1, 0, -516.3065414428711]),
                        np.array([0, 0, 0, 2026.464289992846]),
                        np.array([0, 0, 0.003669942135328761, 0.8054465728326005])
                    ])

    vec_tmp = np.matmul(globQ, vec_tmp)
    vec_tmp = np.reshape(vec_tmp, (1,4))[0]
    x = vec_tmp[0]
    y = vec_tmp[1]
    z = vec_tmp[2]
    w = vec_tmp[3]

    point = [x,y,z]/w

    return point # this is the 3D point

def track_disp_to_3D(tracks):
    """
        Extracts disparity from the track and converts to 3D point for x,y.
        track: x,y,xmin,ymin,xmax,ymax,score,inner_xmin,inner_ymin,inner_xmax,inner_ymax,avg,var
        x,y: center of bbox
        avg: the avg disparity in the inner area
    """
    trajectories = []
    for idx, track in enumerate(tracks):
        trajectory = []
        bboxes = track["bboxes"]
        #print("bboxes", bboxes)
        for bbox in bboxes:
            x,y = bbox[0], bbox[1]
            disp = bbox[11]
            if disp is not None:
                point = disparity_to_point(x,y,disp)
                trajectory.append(point)
            else:
                trajectory.append(None)

        traj_stripped = [t for t in trajectory if t is not None]
        x = [p[0] for p in traj_stripped]
        y = [p[1] for p in traj_stripped]
        z = [p[2] for p in traj_stripped]
        if x and y and z:
            trajectories.append(trajectory)
    x = np.array([p[0]/1000 for p in max_traj])
    y = np.array([p[1]/1000 for p in max_traj])
    z = np.array([p[2]/1000 for p in max_traj])
    if len(x) and len(y) and len(z):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, color=np.random.rand(3))
        ax.scatter(x[0], y[0], z[0], marker='o', color='g')
        ax.scatter(x[-1], y[-1], z[-1], marker='o', color='r')
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')

        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.invert_zaxis()
        plt.show()
    return trajectories

def calculate_velocity_from_3D(trajectory, fps):
    """
        Use 3D trajectory and frame rate to calculate swimming velocity.
        First, find the speed per frame. Then, convert to speed per second.
        trajectory: contains one point for each frame. Unknown points = None. NB: given in mm.
        fps: frames per second of video.
    """
    velocities = []
    start_idx = 0
    prevPoint = trajectory[start_idx]
    # handle None entries in beginning of array
    while prevPoint is None:
        start_idx += 1
        prevPoint = trajectory[start_idx]
    frame_count = 1
    for point in trajectory[start_idx+1:]:
        # handle None entries in array
        if point is None: 
            frame_count += 1
            continue
        # velocity vector
        dx = point[0]-prevPoint[0]
        dy = point[1]-prevPoint[1]
        dz = point[2]-prevPoint[2]
        # absolute velocity
        d = math.sqrt(dx**2+dy**2+dz**2)
        velocities.append(d/frame_count)
        frame_count = 1
        prevPoint = point
    if len(velocities) == 0:
        return None
    velocity_per_frame = sum(velocities)/len(velocities)
    
    # convert to m/s
    vel = velocity_per_frame*fps/1000
    return vel

def calc_avg_and_var(vels):
    v = [vel for vel in vels if vel is not None]
    avg = sum(v)/len(v)
    std = statistics.stdev(v)
    maxv = max(v)
    minv = min(v)
    maxv_length = 0
    minv_length = 0
    print("avg, std, max v, min v")
    print(avg, std, maxv, minv)
    return 0

if __name__ == "__main__":
    shelf = shelve.open("tracks.slhf") # open track file
    tracks = shelf["tracks"]
    shelf.close()

    trajs = track_disp_to_3D(tracks)
    vels = []
    for t in trajs:
        vel = calculate_velocity_from_3D(t, 24)
        vels.append(vel)

    calc_avg_and_var(vels)
    
