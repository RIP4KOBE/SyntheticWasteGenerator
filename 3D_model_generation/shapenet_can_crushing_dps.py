import hou
import os
import time

#global variable
current_frame = 0

class Crushing_DPS(object):

    """Crushing DPS (CDPS) """

    def __init__(self):

        """setup CDPS network"""

    #initialize the related nodes
        self.obj = hou.node("/obj")
        crushing = self.obj.createNode("geo")
        self.parentnode = hou.node("obj/geo1")
    
        self.file_node = self.parentnode.createNode("file")
        self.transform_node = self.parentnode.createNode("xform")
        self.remesh_node = self.parentnode.createNode("remesh::2.0")
        self.vellum_constraints_node = self.parentnode.createNode("vellumconstraints")
        self.vellum_solver_node = self.parentnode.createNode("vellumsolver")
        self.rop_geometry_node = self.parentnode.createNode("rop_geometry")
        self.box_node = self.parentnode.createNode("box")

    #set the input of each node
        self.transform_node.setNextInput(self.file_node)
        self.remesh_node.setNextInput(self.transform_node)

        self.vellum_constraints_node.setNextInput(self.remesh_node)
        self.vellum_constraints_node.setInput(2, self.box_node, 0)
        self.vellum_solver_node.setInput(0, self.vellum_constraints_node, 0)
        self.vellum_solver_node.setInput(1, self.vellum_constraints_node, 1)
        self.vellum_solver_node.setInput(2, self.vellum_constraints_node, 2)

        self.rop_geometry_node.setInput(0, self.vellum_solver_node, 0)

    #set the parameters of each node

        #transform_node
        self.transform_node.parm("scale").set(1)
        self.transform_node.parm("ty").set(0.55)
        
        #remesh_node
        self.remesh_node.parm("targetsize").set(0.05)
        
        #vellum_sonstraints_node
        self.vellum_constraints_node.setParms({"constrainttype": 3})#indicate "Cloth" type (Note that 3 is the index of "Cloth" type)
        self.vellum_constraints_node.parm("domass").set(3)#indicate "Calculate Varing" type
        self.vellum_constraints_node.parm("dothickness").set(2)#indicate "Calculate uniform" type
        self.vellum_constraints_node.parm("thicknessscale").set(1)
        self.vellum_constraints_node.parm("dragnormal").set(2500)
        self.vellum_constraints_node.parm("dragtangent").set(2500)
        self.vellum_constraints_node.parm("stretchdampingratio").set(0.08)
        self.vellum_constraints_node.parm("bendstiffness").set(0.5)
        self.vellum_constraints_node.parm("benddampingratio").set(1)
        self.vellum_constraints_node.parm("bendplasticity").set(1)
        self.vellum_constraints_node.parm("bendplasticthreshold").set(1)
        self.vellum_constraints_node.parm("bendplasticrate").set(1)
        self.vellum_constraints_node.parm("bendplastichardening").set(1000)
        self.vellum_constraints_node.parm("stretchgrp").set("stretch")
        self.vellum_constraints_node.parm("bendgrp").set("bend")
        
        #vellum_solver_node
        self.vellum_solver_node.parm("substeps").set(4)
        self.vellum_solver_node.parm("niter").set(16)
        self.vellum_solver_node.parm("smoothiter").set(10)
        # vellum_solver_node.setParms({"groundposx": 0.1, "groundposy": 0.1, "groundposz": 0.1})
        self.vellum_solver_node.parm("useground").set(True)
        
        #box_node
        self.box_node.setParms({"sizex":1, "sizey":1, "sizez":1})
        key = hou.Keyframe()
        parm = self.box_node.parm("ty")
        key.setFrame(0)
        key.setValue(1.5)
        parm.setKeyframe(key)
        key.setFrame(50)
        key.setValue(0.8)
        parm.setKeyframe(key)

    #Enable simulation
        hou.setSimulationEnabled(True)


    #move the created nodes to proper position
        self.file_node.moveToGoodPosition()
        self.transform_node.moveToGoodPosition()
        self.remesh_node.moveToGoodPosition()
        self.vellum_constraints_node.moveToGoodPosition()
        self.vellum_solver_node.moveToGoodPosition()
        self.box_node.moveToGoodPosition()
        self.rop_geometry_node.moveToGoodPosition()

    def outputPlaybarEvent(self, event_type, frame):
        '''
            callback function
        
        '''
        global current_frame

        if event_type == hou.playbarEvent.FrameChanged:
            current_frame = frame
            # print("current_frame:", current_frame)

    def batch_crushing_dps(self, shapenet_category_path, dps_saved_path):
        '''
            batch CDPS function
        
        '''
        input_path = shapenet_category_path
        model_id = os.listdir(input_path)
        i = 0

        #record the playbar's current frame
        hou.playbar.addEventCallback(self.outputPlaybarEvent)

        for id in model_id:

            #CDPS handling
            model_path = input_path + "/" + id + "/" + "models" + "/" + "model_normalized.obj"
            self.file_node.parm("file").set(model_path)

            #adopt ROP Geometry node to save .obj geometry  
            output_path = dps_saved_path + "/" + id + "/"  + "models" + "/" + "model_normalized.obj"
            self.rop_geometry_node.parm("sopoutput").set(output_path)


            # handle time asynchronism between script and houdini simulator
            time.sleep(2)

            hou.playbar.play()
            print("can_" + id + " " + "crushing process start!")

            while True:

                #setup desplay_flag
                self.vellum_solver_node.setDisplayFlag(True)
                self.vellum_solver_node.setRenderFlag(True) 

                self.rop_geometry_node.parm("execute").pressButton()

                if current_frame >= 48:
                    i = i +1
                    hou.playbar.stop()
                    self.vellum_solver_node.setDisplayFlag(False)
                    self.vellum_solver_node.setRenderFlag(False) 
                    # self.rop_geometry_node.parm("execute").pressButton()
                    # self.vellum_solver_node.geometry().saveToFile("msw_model_normalized.obj") 
                    # hou.hscript('opsave ‘ + operator + ’ ' + filename)   
                    # msw_data.saveToFile(output_path + "/" + id + "/"  + "models" + "/" + "msw_model_normalized.obj") 
                    print("can_" + id + " " + "dps finished!")
                    print("msw_model_number:", i, '\n')            
                    break

            # hou.playbar.stop()
            hou.setFrame(1)
            # handle time asynchronism between script and houdini simulator
            time.sleep(2)
            

    def run(self):

        '''
            main function
        
        '''
        shapenet_category_path = "/media/walker2/ZHUOLI/DPS/dataset/ShapeNetCoreV2/ShapeNetCoreV2/ShapeNetCore/data/v2/ShapeNetCorev2/02946921"
        dps_saved_path = "/media/walker2/ZHUOLI/DPS/dataset/WasteNet/02946921"
        self.batch_crushing_dps(shapenet_category_path, dps_saved_path)
        
if __name__ == "__main__":
    crushing_operation = Crushing_DPS()
    crushing_operation.run()