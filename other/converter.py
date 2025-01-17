from hailo_sdk_client import ClientRunner
import numpy as np

######### Parsing #########

model_name = 'ufld_v2_tusimple'
onnx_path = './{}.onnx'.format(model_name) 

# The slices and reshapes at the end of the model are currently not supported, so they will
# be performed on the host.
end_node = 'Gemm_51' 

runner = ClientRunner(hw_arch='hailo8l')
hn, npz = runner.translate_onnx_model(onnx_path, model_name, end_node_names=[end_node])

hailo_model_har_name = '{}_hailo_model.har'.format(model_name)
runner.save_har(hailo_model_har_name)

######### Optimizing #########

alls_content = [
    'norm_layer1 = normalization([123.675,116.28,103.53],[58.395,57.12,57.365])',
]
alls_path = './{}.alls'.format(model_name) 
open(alls_path,'w').writelines(alls_content)

calib_dataset = np.load('calibset_64_320_800.npy')

runner.load_model_script(alls_path)
runner.optimize(calib_dataset)

quantized_har_path = './{}_quantized.har'.format(model_name)
runner.save_har(quantized_har_path)

######### Compiling #########

hef = runner.compile()

file_name = model_name + '.hef'
with open(file_name, 'wb') as f:
    f.write(hef)