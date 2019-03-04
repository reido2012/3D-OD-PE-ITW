import pascal3d

data_type = 'all'
dataset = pascal3d.dataset.Pascal3DDataset(data_type, dataset_path="/home/omarreid/selerio/datasets/PASCAL3D+_release1.1", generate=True)
#print("Dataset Length: {}".format(len(dataset)))
dataset.create_tfrecords("/home/omarreid/selerio/final_year_project/real_domain_tfrecords/")
