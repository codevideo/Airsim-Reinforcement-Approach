# Please take the links below as reference
# https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/21#issuecomment-518615107

# Before we make the environment configure(such as gate pose, scale, and etc.), we should do the following things
# First, define the client
#      Ex. self.client = airsim.MultirotorClient()
# Second, load the level
#      Ex. self.client.simLoadLevel('Soccer_Field_Easy')
# Then you can adjust your environment now( such as modified the gate)

# In addition, you may check the officail api document to get the name of API
# Ex. https://microsoft.github.io/AirSim-NeurIPS2019-Drone-Racing/api.html
# or this technical paper
# https://arxiv.org/pdf/2003.05654.pdf


def configEnvironment(self):
    # list all the object in the scene and delete all the gates 
    for old_gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
        # simDestoryObject is a sirsim build-in function
        self.client.simDestroyObject(old_gate_object)
        # time.sleep(0.05) # you can sleep a while here to avoid environment crash, but the environment should be find idealy

    # The gate name you can check at previous step by print the objects name list
    # take this link as reference
    # https://github.com/microsoft/AirSim/blob/master/PythonClient/airsim/client.py#L386
    # And idealy you can print all the object name by calling simListSceneObjects
    
    # Create a gate and set the gate at position(world frame)  = 0, 0 ,20. And we set scale of this gate 0.85 
    # Just call simSpawnObject to create a gate
    # Take the link as reference
    # https://github.com/microsoft/AirSim/blob/master/PythonClient/airsim/client.py#L400
    #      Ex. print the object list
    # object_list = self.client.simListSceneObjects()
    # print(object_list)
    self.target_gate = self.client.simSpawnObject("target_red_gate", "Gate01", Pose(position_val=Vector3r(0,0,20)), 0.85)
    
    # self.target_gate = self.client.simSpawnObject("red_gate", "Gate11_23", Pose(position_val=Vector3r(0,0,20)), 0.85)
    

    # The pose of the gate contains orientation and position
    # object pose's format like(use nan here for default value):

    """
    <Pose> {   'orientation': <Quaternionr> {   'w_val': nan,
        'x_val': nan,
        'y_val': nan,
        'z_val': nan},
        'position': <Vector3r> {   'x_val': nan,
        'y_val': nan,
        'z_val': nan}}
    """
    
    # So you can set the gate pose by call simSetObjectPose
    # the ref link list below:
    # https://github.com/microsoft/AirSim/blob/master/PythonClient/airsim/client.py#L343




    # open the csv file path for recording the data 
    if os.path.exists(self.csv_path):
        # If datapath is exist, just append new data name into this list 
        self.file = open(self.csv_path, "a")
    else:
        # If datapath is not exist, we create one new csv file to record the data
        self.file = open(self.csv_path, "w")