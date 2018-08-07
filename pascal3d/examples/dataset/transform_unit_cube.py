#!/usr/bin/env python

import pascal3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(2):
        print('[{dtype}:{id}] showing cad overlay'
              .format(dtype=data_type, id=i))
        transformed_cube = dataset.transform_unit_cube(i)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
      
        counter = 0
        r = [-1, 1]
        for s, e in combinations(cube, 2):
            colors = ["r", "g", "b"]
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color=colors[counter])
                counter+=1
                counter = counter %3
        
        #savefig("./fig"+str(i)+".jpg")
        plt.show()


if __name__ == '__main__':
    main()

