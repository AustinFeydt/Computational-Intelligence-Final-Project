counter=1

while [ $counter -le 200 ]
do
# generate random x,y, and yaw (rotation abt z axis)
ox=$((1 + RANDOM % 20))
oy=$((-20 + RANDOM % 40))
oyaw=$((RANDOM % 15))
x=$(echo "scale=2;($ox/100)" | bc)
y=$(echo "scale=2;($oy/100)" | bc)
yaw=$(echo "scale=2;($oyaw/10)" | bc)


echo $counter 
echo $x
echo $y
echo $yaw

roslaunch image_recognition add_spinner.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture spinner_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model spinner


roslaunch image_recognition add_bulb.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture bulb_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model bulb


roslaunch image_recognition add_can.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture can_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model can


roslaunch image_recognition add_helmet.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture helmet_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model helmet


roslaunch image_recognition add_jug.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture jug_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model jug


roslaunch image_recognition add_knife.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture knife_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model knife


roslaunch image_recognition add_mouse.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture mouse_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model mouse


roslaunch image_recognition add_sofa.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1 
rosrun image_recognition take_picture sofa_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model sofa


roslaunch image_recognition add_wheel.launch x:=$x y:=$y z:=1.5 yaw:=$yaw
sleep 1
rosrun image_recognition take_picture wheel_${counter}_${xx}_${yy}_${zz}.pcd
rosservice call gazebo/delete_model wheel

((counter++))

done

echo All done
