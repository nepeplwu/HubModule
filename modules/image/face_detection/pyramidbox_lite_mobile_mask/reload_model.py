# coding=utf-8
import paddle
import paddle.fluid as fluid

place = fluid.CPUPlace()
exe = fluid.Executor(place)
test_prog, feed_name, fetch_list = fluid.io.load_inference_model(
    dirname="face_detector_mask",
    executor=exe,
    model_filename='model',
    params_filename='weights')

print(feed_name)
print(fetch_list)

fluid.io.save_inference_model(
    dirname="pyramidbox_lite_mobile_mask_detectoion",
    feeded_var_names=feed_name,
    target_vars=fetch_list,
    main_program=test_prog,
    executor=exe)
