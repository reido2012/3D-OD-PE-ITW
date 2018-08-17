import pascal3d

data_type = 'all'
dataset = pascal3d.dataset.Pascal3DDataset(data_type, generate=False)
#print("Dataset Length: {}".format(len(dataset)))
dataset._create_tfrecords_from_data_ids('airplanes.tfrecords', ['2008_003140', '2010_001174'])
