import hou
import os
import time


class Visualizing_MSW_Model(object):

    def __init__(self):

        """setup batch visualizing network"""

    # initialize the related nodes
        self.obj = hou.node("/obj")
        crushing = self.obj.createNode("geo")
        self.parentnode = hou.node("obj/geo1")
        self.file_node = self.parentnode.createNode("file")

    # Enable simulation
        hou.setSimulationEnabled(True)

    # move the created nodes to proper position
        self.file_node.moveToGoodPosition()

    def batch_visualizing(self, wastenet_category_path):
        '''
            batch MSW model visualizing function
        
        '''
        input_path = wastenet_category_path
        model_id = os.listdir(input_path)
        i = 0

        for id in model_id:

            model_path = input_path + "/" + id + "/" + "models" + "/" + "model_normalized.obj"
            self.file_node.parm("file").set(model_path)
            print("can_" + id + " " + "visualizing process start!")

            # setup desplay_flag
            self.file_node.setDisplayFlag(True)
            self.file_node.setRenderFlag(True) 

            # handle time asynchronism between script and houdini simulator
            time.sleep(5)
            i = i+1 
            print(i)

    def run(self):
        '''
            main function
        '''

        wastenet_category_path = "/media/walker2/ZHUOLI/DPS/dataset/WasteNet/02946921"
        self.batch_visualizing(wastenet_category_path)


if __name__ == "__main__":
    visualizing_operation = Visualizing_MSW_Model()
    visualizing_operation.run()
