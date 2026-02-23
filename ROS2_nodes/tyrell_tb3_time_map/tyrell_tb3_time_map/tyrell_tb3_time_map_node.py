'''

TYRELL PROJECT

turtlebot3 Burger Time Map node

This node works along with a Coppelia simulation scene
and the ROS2 time_map_pkg.

The Coppelia scene handles the simulation of the robot,
including the simulation of what each particle in 
the PF particle cloud "sees" using a simulated lidar, 
as well as the communication with the RL_spin_decoupler.

The time_map_pkg launches the PF and a coordinator that
handles the interactions among nodes and topics.

This node just reads the real lidar topic,
gets the minimum distance for each sector, 
and sends it to the proper topic so the rest of elements
of the experiment read that info.

Remember to source install/local_setup.bash
after each colcon build!!

Ana Cruz-Martín
November 2025

'''


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32

from enum import Enum
import numpy as np
import math

import time


class TyrellTb3TimeMap(Node):

    def __init__(self):
        super().__init__('tyrell_tb3_time_map')
              
        # turtlebot3 linear and angular initial speed values
        self.v = 0.0
        self.w = 0.0
        
        # turtlebot3 linear and angular speed max values
        self.v_max = 0.15
        self.w_max = 1.5
              
        # min distance to goal (i.e., the robot has reached the target)
        self.mindistance2goal = 0.16
       
        # The sectors into we have split the lidar scan
        self.lidar_sectors = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
             
        # publishers
        # important: if the teleop_key node is launched, it takes precedence
        # on the cmd_vel and the robot does not move
      
        # RELATED TO: z_pub publisher in ros_interface script in Coppelia
        self.z_publisher = self.create_publisher(Float32MultiArray, 'z', 10)
        
        #subscribers
        
        # QoS Profile required by the lidar subscriber    
        lidarQoS = QoSProfile(
           reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.VOLATILE,
           history=HistoryPolicy.KEEP_LAST,
           depth=10)
        ask_z_QoS = QoSProfile(
           reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.VOLATILE,
           history=HistoryPolicy.KEEP_LAST,
           depth=10)
        print_cmdvel_QoS = QoSProfile(
           reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.VOLATILE,
           history=HistoryPolicy.KEEP_LAST,
           depth=10)
                         
        self.lidar_sub = self.create_subscription( 
            LaserScan,
            '/scan',
            self.lidar_callback,
            lidarQoS)
        # RELATED TO: ask_z_callback in ros_interface script in Coppelia    
        self.ask_z_sub = self.create_subscription( 
            Int32,
            '/ask_z',
            self.ask_z_callback,
            ask_z_QoS)
            
        self.print_cmdvel_sub = self.create_subscription( 
            Twist,
            '/cmd_vel',
            self.print_cmdvel_callback,
            print_cmdvel_QoS)
            
        
        print('[INIT] tyrell_tb3_time_map init OK!!', flush=True)
        
                
    def lidar_callback(self, msg): 
        # This callback should get the minimum of the 4 scan sectors we take from the lidar
        # We divide the 360º in four 60º sectors with a 120º blind spot on the rear of the robot
        # Such processing should be faster than the /scan publishing rate
        lidar_ranges = msg.ranges
        range_angle = msg.angle_min
        angle_increment = msg.angle_increment
        pithird = np.pi/3
        #print('[LIDAR CALLBACK] Number of measurements:',len(lidar_ranges))
        #print('[LIDAR CALLBACK] Min angle:',msg.angle_min)
        #print('[LIDAR CALLBACK] Max angle:',msg.angle_max)
        #print('[LIDAR CALLBACK] Angle increment:',msg.angle_increment)
        # the lidar sectors array is initialized to the max distance the lidar can measure
        # in this way, if a sector is empty because there are no measures
        # (improbable, but not impossible), the array stores a computable value
        # that the RL Agent can deal with
        min_lidar_sectors = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)
        for ranges_index in range(len(lidar_ranges)):
            sector_index = int(range_angle // pithird)
            #print('[LIDAR CALLBACK] Angle, sector, range: %f %f %f' % (range_angle, sector_index, lidar_ranges[ranges_index]), flush=True)
            if (lidar_ranges[ranges_index] < min_lidar_sectors[sector_index]):
               min_lidar_sectors[sector_index]=lidar_ranges[ranges_index]
            range_angle = range_angle+angle_increment
        #print('[LIDAR CALLBACK] Processed scan:',min_lidar_sectors, flush=True)    
        # this is necessary because the RL sectors and the turtlebot sectors do not match (should we make it match?)   
        self.lidar_sectors[0] = min_lidar_sectors[1]
        self.lidar_sectors[1] = min_lidar_sectors[0]
        self.lidar_sectors[2] = min_lidar_sectors[5]
        self.lidar_sectors[3] = min_lidar_sectors[4]
        # final checks: no nans and no values under max or min range values
        self.lidar_sectors[self.lidar_sectors == 'nan'] = 8.0
        self.lidar_sectors[self.lidar_sectors < msg.range_min] = msg.range_min
        self.lidar_sectors[self.lidar_sectors > msg.range_max] = msg.range_max
        #print('[LIDAR CALLBACK] Translated scan:', self.lidar_sectors, flush=True)
        
    def ask_z_callback(self, msg): 
        # This callback gets an observation petition from the coordinator
        # and then publishes the observation (i.e., the minimum of the sectors the lidar has been divided into)
        msg_z = Float32MultiArray()
        #print('[ASK Z CALLBACK] Lidar sectors: %f %f %f %f', (self.lidar_sectors[0],self.lidar_sectors[1],self.lidar_sectors[2],self.lidar_sectors[3]), flush=True)
        msg_z.data = [float(self.lidar_sectors[0]),float(self.lidar_sectors[1]),float(self.lidar_sectors[2]),float(self.lidar_sectors[3])]
        self.z_publisher.publish(msg_z)
        
    def print_cmdvel_callback(self, msg): 
        # This callback gets the linear and angular speed values in topic cmd_vel
        print('[PRINT CMDVEL] v: %f w: %f ', (msg.linear.x,msg.angular.z), flush=True)

        

def main():
    rclpy.init()
    
    tyrell_tb3_time_map = TyrellTb3TimeMap()
    print('[MAIN] Turtlebot3 Time Map node created!!', flush=True)
    
    rclpy.spin(tyrell_tb3_time_map)
    print('[MAIN] Everybody is spinning!!', flush=True)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tyrell_tb3_time_map.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
