import numpy as np
import pickle
import streamlit
filename= 'trained_model.sav'
loaded_model = pickle.load(open(filename))


input_data=(4583,0.0,133.0,360.0,0.0)
#changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction1=loaded_model.predict(input_data_reshaped)
print(prediction1)

if(prediction1[0]==0):
    print("the person is not eligible for loan")
else:
    print("the person can get loan")