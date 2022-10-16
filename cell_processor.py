import csv
import numpy as np
import pandas as pd
import skimage.measure
from scipy.optimize import linear_sum_assignment
from nd2reader import ND2Reader


# Node를 저장할 객체
class Node():
    def __init__(self, node_id, track_id=-1, mask=None):
        self.node_id = node_id
        self.track_id = track_id
        self.mask = mask
        self.volume = np.sum(self.mask)
        self.coordinate = np.where(self.mask)
        self.bbox = [(np.min(x_range), np.max(x_range)) for x_range in self.coordinate]


# Frame을 저장할 객체
class Frame():
    def __init__(self, image, label):
        self.image = image
        self.label = label

        self.node_list = []
        for i in range(1, np.max(label)+1):
            node = Node(node_id=i, mask=(label==i))
            self.node_list.append(node)

    def initialize_track_id(self):
        for node in self.node_list:
            node.track_id = node.node_id    


class CellProcessor():
    def __init__(self):
        pass

    # 현미경으로 촬영한 이미지 파일(.nd2)를 Load하여 Numpy Array로 제작
    def nd2_to_ndarray(self, file_path):
        data = []
        with ND2Reader(file_path) as images:
            shape = (images.sizes['c'], images.sizes['t'], images.sizes['z'], images.sizes['x'], images.sizes['y'])
            for c in range(images.sizes['c']):
                for t in range(images.sizes['t']):
                    for z in range(images.sizes['z']):
                        i = images.get_frame_2D(c=c, t=t, z=z)
                        data.append(i)
        data = np.array(data).reshape(shape)
        return data

    # IOU 영역의 크기를 계산
    def calc_iou(self, node1, node2):
        iou = np.sum(node1.mask & node2.mask) / np.sum(node1.mask | node2.mask)
        return iou

    # 입력받은 (노드1 길이, 노드2 길이) 만큼의 Numpy Array를 만들고 IOU 값을 삽입
    def calc_iou_matrix(self, node_list1, node_list2):
        num1 = len(node_list1)
        num2 = len(node_list2)
        iou_matrix = np.zeros((num1, num2))
        for i in range(num1):
            for j in range(num2):
                node1 = node_list1[i]
                node2 = node_list2[j]
                iou_matrix[i, j] = self.calc_iou(node1, node2)
        return iou_matrix

    # 너무 작은 노드 정보 삭제
    def delete_small_node(self, label, min_voxel):
        num = np.max(label)
        for i in range(num):
            if (np.sum(label==i) < min_voxel):
                label[np.where(label==i)] = 0
        label = skimage.measure.label(label)
        return label

    # Ndarray로 된 데이터를 DataFrame객체로 변환
    def stats_to_dataframe(self, stats):
        dfStats = pd.DataFrame(stats, columns=['frame_id', 'node_id', 'track_id', 'volume', 'bbox', 'intensity', 'dummy1', 'dummy2', 'dummy3', 'dummy4','brightness'])
        return dfStats

    # DataFrame객체를 CSV로 저장
    def save_to_csv(self, dfData, path='data.csv'):
        dfData.to_csv(path, na_rep='NaN', index=False)

    # 메인 프로세싱 함수
    def processing_cell_data(self, data):
        # (c,t,z,x,y) : 1번세트의 세포 이미지들을 100x100 크기로 Cropping
        images = data[1, :, :, 250:350, 200:300] 

        (t_total, z_total, x_total, y_total) = images.shape # (11, 41, 100, 100)

        arr = images[0, 20].flatten() # 다차원 배열을 평탄화
        arr = arr[arr>0] # 0보다 큰 값으로 솎아냄

        # 임계값 분할 결과의 연결 도메인 찾기
        theshold = 1000
        # label = (images[1, 10] >= theshold)
        labels = []
        for t in range(t_total):
            image = images[t]
            label = skimage.measure.label(image >= theshold)
            label = self.delete_small_node(label=label, min_voxel=20)
            labels.append(label)
        labels = np.array(labels)

        frames = []
        for t in range(t_total): # 총 11개
            # 첫 번째 프레임인 경우 track_id를 node_id로 초기화
            if (t == 0):
                frame = Frame(image=images[t], label=labels[t])
                frame.initialize_track_id()
                frames.append(frame)
            
            # 두 번째 프레임부터 추적
            if (t > 0):
                frame = Frame(image=images[t], label=labels[t])
                frames.append(frame)
                # 이미지와 이전 프레임 사이의 iou를 계산.
                iou_matrix = self.calc_iou_matrix(frames[t-1].node_list, frames[t].node_list)
                # 최대 iou와 일치하는 이분 그래프(헝가리안  알고리즘)
                row_index, col_index = linear_sum_assignment(cost_matrix=iou_matrix, maximize=True)
                for i in range(len(col_index)):
                    frames[t].node_list[col_index[i]].track_id = frames[t-1].node_list[row_index[i]].track_id

        stats = []
        for t in range(t_total):
            for node in frames[t].node_list:
                frame_id = t
                node_id = node.node_id
                track_id = node.track_id
                volume = node.volume                
                (x1, x2), (y1, y2), (z1, z2) = node.bbox
                #ROI Bounding box 내 픽셀값 평균
                # brightness = frames[t].image[x1+1:x2, y1+1:y2, z1+1:z2].mean()  바운딩 박스 경계선 제외
                brightness = frames[t].image[x1:x2+1, y1:y2+1, z1:z2+1].mean()  #바운딩 박스 경계선 포함
                row = [int(frame_id), node_id, track_id, volume, x1, x2, y1, y2, z1, z2,brightness]
                stats.append(row)
        stats = np.array(stats)
        return stats
