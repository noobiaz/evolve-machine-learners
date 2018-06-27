# Load numpy
import numpy as np
import pickle

# Set random seed
np.random.seed(0)

newdata=[[3.6216,8.6661,-2.8073,-0.44699],
         [4.5459,8.1674,-2.4586,-1.4621],
         [-3.5637,-8.3827,12.393,-1.2823],
         [-2.5419,-0.65804,2.6842,1.1952]
]
objectfile = open("randomforest_model.mdl",'rb')
clf=pickle.load(objectfile)

# Create actual english names for the plants for each predicted plant class
pred = clf.predict(newdata)

print (pred)
