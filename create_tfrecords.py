import pascal3d

data_type = 'all'
dataset = pascal3d.dataset.Pascal3DDataset(data_type, generate=False)
#print("Dataset Length: {}".format(len(dataset)))
dataset.create_tfrecords(data_type)
