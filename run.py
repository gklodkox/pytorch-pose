import pose.models as models
import pose.datasets as datasets
import torch
import openvino as ov


njoints = datasets.__dict__["mpii"].njoints
model = models.__dict__["hg"](num_stacks=2,
                                       num_blocks=1,
                                       num_classes=njoints,
                                       resnet_layers=50)
                                       
ov.save_model(ov.convert_model(model,example_input=torch.zeros((1,3,256,256))), "hg-s2-b1-mpii.xml")                                      
