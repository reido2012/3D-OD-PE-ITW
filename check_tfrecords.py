
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pascal3d
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
OVERFIT_TEST_TFRECORDS = "/notebooks/selerio/overfit_check.tfrecords"
EVAL_TEST_TFRECORDS =  "/home/omarreid/selerio/datasets/real_domain_tfrecords/imagenet_val.tfrecords"

def line_boxes(axis, virtual_control_points_pred, current_color):
    # Connect Points to Visualize Cube
    axis.plot(virtual_control_points_pred[:2,0], virtual_control_points_pred[:2,1],current_color)
    axis.plot(virtual_control_points_pred[2:4,0], virtual_control_points_pred[2:4,1],current_color)
    axis.plot(virtual_control_points_pred[4:6,0], virtual_control_points_pred[4:6,1],current_color)
    axis.plot(virtual_control_points_pred[6:8,0], virtual_control_points_pred[6:8,1],current_color)
    #ALONG
    axis.plot([virtual_control_points_pred[1,0],virtual_control_points_pred[5,0]],
             [virtual_control_points_pred[1,1],virtual_control_points_pred[5,1]], current_color)
    axis.plot([virtual_control_points_pred[2,0],virtual_control_points_pred[6,0]],
             [virtual_control_points_pred[2,1],virtual_control_points_pred[6,1]], 'y')
    axis.plot([virtual_control_points_pred[3,0],virtual_control_points_pred[7,0]],
             [virtual_control_points_pred[3,1],virtual_control_points_pred[7,1]], 'y')
    axis.plot([virtual_control_points_pred[0,0],virtual_control_points_pred[4,0]],
             [virtual_control_points_pred[0,1],virtual_control_points_pred[4,1]],current_color)

    #VERTICAL
    axis.plot([virtual_control_points_pred[2,0],virtual_control_points_pred[0,0]],
             [virtual_control_points_pred[2,1],virtual_control_points_pred[0,1]],current_color)
    axis.plot([virtual_control_points_pred[3,0],virtual_control_points_pred[1,0]],
             [virtual_control_points_pred[3,1],virtual_control_points_pred[1,1]],current_color)
    axis.plot([virtual_control_points_pred[5,0],virtual_control_points_pred[7,0]],
             [virtual_control_points_pred[5,1],virtual_control_points_pred[7,1]],current_color)
    axis.plot([virtual_control_points_pred[4,0],virtual_control_points_pred[6,0]],
             [virtual_control_points_pred[4,1],virtual_control_points_pred[6,1]], current_color)

def main():
    reconstructed_records = []
    record_iterator = tf.python_io.tf_record_iterator(path=EVAL_TEST_TFRECORDS)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)    

        output_vector = example.features.feature['output_vector'].float_list.value

        img_string = (example.features.feature['object_image']
                                      .bytes_list
                                      .value[0])


        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((224, 224, -1))

        reconstructed_records.append((reconstructed_img, output_vector))
        break

    print(len(reconstructed_records))

    overfit_test_model_output_vectors = [np.array([0.7187886 , 0.65310484, 0.703458  , 0.23609623, 0.05358947,
           0.6517882 , 0.04232024, 0.25617638, 0.95103073, 0.6369277 ,
           0.9445842 , 0.23127775, 0.2952587 , 0.6450021 , 0.27361175,
           0.25556254, 0.19366504, 0.44451308, 0.12243512], dtype=np.float32), np.array([0.7187886 , 0.65310484, 0.703458  , 0.23609623, 0.05358947,
           0.6517882 , 0.04232024, 0.25617638, 0.95103073, 0.6369277 ,
           0.9445842 , 0.23127775, 0.2952587 , 0.6450021 , 0.27361175,
           0.25556254, 0.19366504, 0.44451308, 0.12243512], dtype=np.float32), np.array([-0.0239566 ,  0.59844226, -0.01136585,  0.25561845,  0.7250648 ,
            0.62555623,  0.73980594,  0.3037456 ,  0.20461608,  0.6317109 ,
            0.2250872 ,  0.27127463,  0.9808837 ,  0.64997816,  0.9878197 ,
            0.31468123,  0.18865545,  0.44844636,  0.13619864], dtype=np.float32), np.array([0.48931915, 0.9077469 , 0.4577347 , 0.3478176 , 0.13867906,
           0.8663925 , 0.10426442, 0.33999634, 1.1568272 , 0.7281852 ,
           1.1512758 , 0.19059807, 0.74646163, 0.6940164 , 0.720421  ,
           0.18382867, 0.34786937, 0.3188813 , 0.19324563], dtype=np.float32), np.array([0.13749509, 0.87916064, 0.12559421, 0.0735916 , 0.6073713 ,
           0.7327508 , 0.5996402 , 0.05301648, 0.57760656, 0.85866594,
           0.5757314 , 0.04514929, 0.96842957, 0.71376985, 0.9496281 ,
           0.03016482, 0.08137416, 0.47213647, 0.14396743], dtype=np.float32), np.array([0.3507287 , 0.9290097 , 0.2954786 , 0.36910856, 0.57081085,
           0.7597713 , 0.5164338 , 0.23582195, 0.58257765, 0.92684346,
           0.54922915, 0.35211468, 0.73906183, 0.75350106, 0.6880743 ,
           0.22287148, 0.12550703, 0.4254062 , 0.16461858], dtype=np.float32), np.array([-9.3685836e-04,  9.0947896e-01, -8.1444353e-02, -8.6855069e-02,
           -2.5583297e-02,  8.7846267e-01, -7.4943013e-02, -5.2352674e-02,
            3.3531007e-01,  9.3864477e-01,  3.0339205e-01, -3.4601826e-02,
            3.1604254e-01,  9.1162968e-01,  2.5313842e-01, -2.0378776e-02,
            1.0279549e-01,  5.3679474e-02,  4.9268785e-01], dtype=np.float32), np.array([ 0.3600914 ,  1.1668217 ,  0.27296248,  0.24224426, -0.03178392,
            1.0927744 , -0.10671698,  0.21048747,  1.0965536 ,  1.0417019 ,
            1.0442414 ,  0.15362379,  0.69008386,  0.97552776,  0.6122069 ,
            0.12528446,  0.21094927,  0.26436332,  0.33510113], dtype=np.float32), np.array([ 1.1895597 ,  0.73511404,  1.165459  ,  0.16514616,  0.38494942,
            0.9939261 ,  0.349514  ,  0.3646413 ,  0.67333513,  0.59088457,
            0.64974   ,  0.04696846, -0.15605962,  0.84155315, -0.19837648,
            0.23930195,  0.2358393 ,  0.3840887 ,  0.19205976], dtype=np.float32), np.array([-0.0211289 ,  1.1663948 , -0.14549269, -0.0900111 , -0.24286798,
            1.0392663 , -0.36384207, -0.12958942,  0.792834  ,  1.2024897 ,
            0.69214946, -0.04743359,  0.50510854,  1.0735892 ,  0.38948506,
           -0.09140408,  0.17267635,  0.30698714,  0.30407467], dtype=np.float32), np.array([ 0.85536563,  0.6650021 ,  0.87286407, -0.10621923,  0.6266766 ,
            0.8599119 ,  0.64578974,  0.05275044,  0.3045078 ,  0.6379949 ,
            0.3281358 , -0.11878517,  0.04216306,  0.83490133,  0.04525986,
            0.03200029,  0.2259851 ,  0.3783791 ,  0.2520491 ], dtype=np.float32), np.array([0.7187886 , 0.65310484, 0.703458  , 0.23609623, 0.05358947,
           0.6517882 , 0.04232024, 0.25617638, 0.95103073, 0.6369277 ,
           0.9445842 , 0.23127775, 0.2952587 , 0.6450021 , 0.27361175,
           0.25556254, 0.19366504, 0.44451308, 0.12243512], dtype=np.float32), np.array([ 0.23873308,  0.9568063 ,  0.25011802,  0.24409385, -0.09171816,
            0.801963  , -0.07489874,  0.17755629,  0.8865029 ,  0.92781985,
            0.9409559 ,  0.25617057,  0.49545276,  0.78530514,  0.5174366 ,
            0.18233177,  0.12476974,  0.29796264,  0.31696737], dtype=np.float32), np.array([ 0.20299011,  0.95766884,  0.15097986,  0.23002708,  0.00137485,
            0.6272439 , -0.03776877, -0.08551416,  0.89424384,  1.1542064 ,
            0.89394337,  0.42064416,  0.67520356,  0.81939566,  0.65901136,
            0.0928608 ,  0.39029327,  0.17701136,  0.38576582], dtype=np.float32), np.array([ 0.13931467,  1.1165925 ,  0.17736724, -0.01614794,  0.29367843,
            0.9895835 ,  0.34618652, -0.10889689,  0.5803245 ,  1.1547688 ,
            0.6538753 ,  0.0294215 ,  0.7301514 ,  1.0145257 ,  0.7802182 ,
           -0.09163707,  0.21456364,  0.25012356,  0.43997082], dtype=np.float32), np.array([ 1.2325265 ,  0.917777  ,  1.2413272 , -0.28104872,  0.5469198 ,
            1.1996784 ,  0.5645141 , -0.06252349,  0.47244072,  0.9797687 ,
            0.49156284, -0.204148  , -0.2230312 ,  1.2518817 , -0.22800833,
            0.00771637,  0.22431026,  0.29433092,  0.34215054], dtype=np.float32), np.array([-0.07501819,  0.8357577 , -0.10170341,  0.36880124,  0.221697  ,
            0.8360987 ,  0.20196372,  0.3994243 ,  0.7477002 ,  0.8251189 ,
            0.7451714 ,  0.36566365,  1.0148724 ,  0.8259771 ,  1.0021017 ,
            0.38905555,  0.39697444,  0.263388  ,  0.12326317], dtype=np.float32),np.array([0.09327006, 0.7342062 , 0.08737992, 0.1888405 , 0.2258617 ,
           0.48478633, 0.2203486 , 0.05507139, 0.88444364, 0.7620656 ,
           0.90294534, 0.21271007, 0.91246724, 0.5079117 , 0.9078499 ,
           0.07711841, 0.15183888, 0.45202062, 0.15137576], dtype=np.float32), np.array([ 1.3206236 ,  0.6747888 ,  1.3245734 ,  0.0928131 ,  0.11028839,
            0.762558  ,  0.10132063,  0.14706884,  0.93469244,  0.61334264,
            0.9480634 ,  0.05801451, -0.23878369,  0.693088  , -0.24717739,
            0.10881647,  0.20205285,  0.42895728,  0.20380062], dtype=np.float32), np.array([-0.09895074,  1.0387137 , -0.20963342,  0.33453482,  0.00709361,
            0.74591875, -0.09108543,  0.1145011 ,  0.95782125,  0.963497  ,
            0.8902376 ,  0.27061325,  0.9795817 ,  0.67266285,  0.890505  ,
            0.04045931,  0.3067491 ,  0.35819018,  0.1722561 ], dtype=np.float32), np.array([-0.00572798,  0.64968896,  0.00678879,  0.2374812 ,  0.60992926,
            0.6057931 ,  0.6236416 ,  0.2523066 ,  0.36417508,  0.6874731 ,
            0.38434154,  0.25383586,  0.9622903 ,  0.6285633 ,  0.9683409 ,
            0.2632054 ,  0.13588859,  0.471036  ,  0.10864083], dtype=np.float32)]

    for counter , (image, output_vector) in enumerate(reconstructed_records):
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1,2,2)
        ax.imshow(image)
        ax2.imshow(image)
        current_color = 'b'
        
        #output_vector_predicted = overfit_test_model_output_vectors[counter]
        print(np.array(output_vector))
        virtual_control_points = np.array(output_vector[:16]).reshape(8,2) * 224
        print("Real Labels:")
        print(virtual_control_points)
        #virtual_control_points_pred = np.array(output_vector_predicted[:16]).reshape(8,2) * 224
        #print("Predicted Labels:")
        #print(virtual_control_points_pred)
        #ax.scatter(virtual_control_points[:, 0], virtual_control_points[:, 1], c='r')
        #ax2.scatter(virtual_control_points_pred[:, 0], virtual_control_points_pred[:, 1])
        #line_boxes(ax2, virtual_control_points_pred, 'b')
        #line_boxes(ax, virtual_control_points, 'r')

        #plt.savefig("./{}_check.jpg".format(counter))
        print("Done")

main()
