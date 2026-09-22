# A frame-local 3D detection. id is not a tracking ID.
uint32 id
int32 class_id
string class_name
float32 score
geometry_msgs/Pose pose
# dimensions.x/y/z are box length along local x/y/z (model w/l/h).
geometry_msgs/Vector3 dimensions
geometry_msgs/Twist velocity
# Four bottom-face corners, counter-clockwise in the output frame.
geometry_msgs/Point[4] corners

