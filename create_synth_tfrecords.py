import pascal3d

data_type = 'all'
dataset = pascal3d.dataset.Pascal3DDataset(data_type, dataset_path="/Users/omarreid/Documents/2D23D/PASCAL3D+_release1.1")
dataset.create_tfrecords_synth_domain("/Users/omarreid/Documents/2D23D/synth_domain_records/", True)

