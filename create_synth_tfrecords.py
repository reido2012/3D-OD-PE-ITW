import pascal3d

data_type = 'all'
dataset = pascal3d.dataset.Pascal3DDataset(data_type, dataset_path="/home/omarreid/selerio/datasets/PASCAL3D+_release1.1")
dataset.create_tfrecords_synth_domain("/home/omarreid/selerio/datasets/synth_domain_tfrecords_new/", True)

