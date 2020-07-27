import numpy as np
import pandas as pd
from skopt import BayesSearchCV

class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):

        ious=np.max(self.iou(boxes, clusters), axis=1)


        matched=ious>0.5

        howmany=np.sum(matched)/ious.shape[0]*100
        #print(np.sum(matched)/ious.shape[0]*100,'% iou >0.5')


        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy,howmany
        np.dot
    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):

        marking = pd.read_csv(self.filename)

        klasses = list(set(marking['source']))
        klasses.sort()
        print(klasses)
        bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

        f = open(self.filename, 'r')
        dataSet = []
        for box in bboxs:

            dataSet.append([box[2], box[3]])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self,):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)

        print(result.tolist())
        print("K anchors:\n {}".format(result))


        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result)[0] * 100))

        print("how many: {:.2f}%".format(
            self.avg_iou(all_boxes, result)[1] * 100))

        max_ioumatch=0
        # for base in range(12,33,2):
        #
        #     for h in range(0,20,1):
        #         for w in range(0,20,1):
        if 1:
            if 1:
                if 1:
                    base=4*8
                    h=14
                    w=7

                    base=base

                    base_anchor=base

                    ratio=[h/10,w/10]

                    effdet_anchor=[]

                    for i in range(5):
                        for scale in range(3):
                            base_anchor_cur=base_anchor*2**i*(2**(scale/3))
                            #print(base_anchor_cur)
                            effdet_anchor.append([base_anchor_cur,base_anchor_cur])
                            effdet_anchor.append([ratio[0]*base_anchor_cur, ratio[1]*base_anchor_cur])
                            effdet_anchor.append([ratio[1]*base_anchor_cur, ratio[0]*base_anchor_cur])


                    result2=np.array(effdet_anchor)

                    a,howmany=self.avg_iou(all_boxes, result2)
                    print(a)
                    if howmany>max_ioumatch:
                        max_ioumatch=howmany
                        print(max_ioumatch,'anchor matched with',base,' ',w/10,' ',h/10)
        # print(result2)


        print("newanchor Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result2) * 100))


if __name__ == "__main__":
    cluster_number = 45
    filename = "../global-wheat-detection/train.csv"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()

    {'inrae_1', 'arvalis_1', 'arvalis_3', 'rres_1', 'arvalis_2', 'usask_1', 'ethz_1'}