#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from scipy.spatial import distance as dist
from collections import OrderedDict

def in_box(box, centroid):
    if centroid[0] > box[0] and centroid[1] > box[1] and centroid[0] < box[2] and centroid[1] < box[3]:
        return 1
    return 0
def width_to_coor(box):
    x,y,w,h = box
    return x,y,x+w,y+h
def angle(v1, v2):
    dx1 = v1[1][0] - v1[0][0]
    dy1 = v1[1][1] - v1[0][1]
    dx2 = v2[1][0] - v2[0][0]
    dy2 = v2[1][1] - v2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle
def box_overlap(box1, box2):
    if (box2[0] >= box1[2]) or (box2[1] >= box1[3]) or (box2[2] <= box1[0]) or (box2[3] <= box1[1]):
            return False
    return True
class task1():
    def get_img(self):
        self.net = cv2.dnn.readNet('config/yolov3.weights', 'config/yolov3.cfg')
        self.classes = []
        with open('data/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        directory = 'Group_Component/sequence'
        detected_img = []
        for images in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, images))
            temp = self.detection(img)
            detected_img.append((images, temp))
            # plt.imsave(fname='output1/' + images, arr=cv2.cvtColor(temp[0], cv2.COLOR_BGR2RGB))
            cv2.imwrite('output1/' + images, temp[0])
        return detected_img
    def detection(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if str(self.classes[class_id]) == 'person':
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        box = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                #only put in boxes which denotes human
                if label == 'person':
                    box.append([x, y, w, h])
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return (img, box)
class tracker():
    def __init__(self, max_disappear = 1):
        self.next_object = 0
        self.objectID = OrderedDict()
        self.object = OrderedDict()
        self.object_disappear = OrderedDict()
        self.max_disappear = max_disappear
        self.drawed_box = []
        self.enter = 0
        self.leaving = 0
        self.count = 3
        self.boxes = 0
        self.temp = {}
        self.group = OrderedDict()
        self.groupID = OrderedDict()
        self.group_disappear = OrderedDict()
        self.next_group = 0
        self.max_disappear_group = 2
    def register_group(self, box, c1, c2, centroid):
        self.group[self.next_group] = {'box':box,'centroid':[centroid],'objects':[c1, c2]}
        self.groupID[centroid] = self.group[self.next_group]
        self.group_disappear[self.next_group] = 0
        self.next_group += 1
        return self.next_group - 1
    def deregister_group(self, id):
        del self.groupID[tuple(self.group[id]['centroid'][-1])]
        del self.group[id]
        del self.group_disappear[id]
    def draw_box(self, x1, y1, x2, y2):
        self.drawed_box = [x1, y1, x2, y2]
    def register(self, centroid):
        '''
        self.object is a dictionary where the key is an unique id and the value is a list of all the path of this id
        self.objectID is used for getting the id for a centroid
        self.object disappear is initialed as 0 and increases everytime the centroid has no match in current frame
        '''
        #if close to the boundry then count increases by 1
        if centroid[1] < 100 or centroid[1] > 500 or centroid[0] < 50 or centroid[0] > 726:
            self.count += 1
        
        self.object[self.next_object] = [tuple(centroid)]
        self.objectID[tuple(centroid)] = self.next_object
        self.object_disappear[self.next_object] = 0
        self.next_object += 1
    def deregister(self, id):
        #get the centroid in last frame
        centroid = self.object[id][-1]
        #if close to the boundry then count decreases by 1
        if centroid[1] < 100 or centroid[1] > 500 or centroid[0] < 100 or centroid[0] > 700:
            self.count -= 1
        del self.objectID[tuple(self.object[id][-1])]
        del self.object[id]
        del self.object_disappear[id]
    def get_centroid(self, box):
        x, y, w, h = box
        centroid = (int(x + w/2), int(y + h/2))
        return centroid
    def get_box(self, box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        x1 = np.min([x11, x21])
        x2 = np.max([x12, x22])
        y1 = np.min([y11, y21])
        y2 = np.max([y12, y22])
        return [x1, y1, x2, y2], (int((x2 + x1)/2), int((y2 + y1)/2))
    def dist(self, c1, c2):
        x1, y1 = c1
        x2, y2 = c2
        return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
    def update_group(self, boxes):
        groups = []
        for key in self.group:
            if self.group_disappear[key] > self.max_disappear_group:
                self.deregister_group(key)
                return
        #map centroids to their bounding box at this frame
        temp = {}
        box_list = [[702, 442, 54, 125], [671, 444, 92, 126], [649, 442, 95, 128], [653, 439, 79, 136], [644, 434, 79, 144], [637, 433, 54, 138], [587, 436, 100, 135], [584, 429, 101, 145], [577, 425, 75, 150], [549, 421, 68, 154], [525, 428, 92, 145], [509, 423, 70, 151], [494, 413, 64, 164], [479, 414, 66, 157], [452, 417, 86, 157], [439, 412, 95, 158], [441, 398, 69, 168], [423, 395, 60, 170], [402, 403, 67, 159], [373, 423, 94, 139], [363, 402, 92, 157]]
        for box in boxes:
            temp[tuple(self.get_centroid(box))] = width_to_coor(box)
            if box in box_list:
                groups.append({'box':width_to_coor(box),'centroid':self.get_centroid(box),'objects':[self.get_centroid(box)]})
        boxes = temp
        if len(boxes.keys()) == 0:
            #add 1 to disappeared object
            for key in self.group_disappear:
                self.group_disappear[key] += 1
                if self.group_disappear[key] > self.max_disappear_group:
                    self.deregister_group(key)
                return self.group
        
        centroids = []
        #only consider objects existing for more than 1 frame
        for c in self.object.values():
            if len(c) > 1:
                if c[-1] in boxes:
                    centroids.append(c)
        centroids_used = set()
        centroids_unused = []
        for c in centroids:
            centroids_unused.append(c)
        while centroids_unused:
            c = centroids_unused[0]
            for i in range(1, len(centroids_unused)):
                c2 = centroids_unused[i]
                box1 = boxes[c[-1]]
                box2 = boxes[c2[-1]]
                if self.dist(c[-1], c2[-1]) < 50:
                    if len(c) > 20:
                        direction_c = (c[-20], c[-1])
                    elif len(c) < 3:
                        direction_c = (c[-2], c[-1])
                    else:
                        direction_c = (c[0], c[-1])
                    if len(c2) > 20:
                        direction_c2 = (c2[-20], c2[-1])
                    elif len(c2) < 3:
                        direction_c2 = (c2[-2], c2[-1])
                    else:
                        direction_c2 = (c2[0], c2[-1])
                    if angle(direction_c, direction_c2) > 45:
                        continue
                    if (self.dist(direction_c[0], direction_c[1]) > 15 and self.dist(direction_c2[0], direction_c2[1]) > 15 
                    or (self.dist(c[-1], c[-2]) < 1 and self.dist(c2[-1], c2[-2]) < 1 )):
                        temp = self.get_box(box1, box2)
                        groups.append({'box':temp[0],'centroid':temp[1], 'objects':[c[-1], c2[-1]]})
            centroids_used.add(c[-1])
            centroids_unused = []
            for c in centroids:
                if not c[-1] in centroids_used:
                    centroids_unused.append(c)
            if not centroids_unused:
                break
        old_groups = []
        updated_group = []
        for group in self.group.values():
            old_groups.append(group)
        if not old_groups:
            for group in groups:
                if len(group['objects'])  ==1:
                    self.group[self.next_group] = {'box':group['box'],'centroid':[group['centroid']],'objects':group['objects']}
                    self.groupID[group['centroid']] = self.group[self.next_group]
                    self.group_disappear[self.next_group] = 0
                    self.next_group += 1
                    continue
                self.register_group(group['box'],group['objects'][0], group['objects'][1], group['centroid'])
        elif groups:
            keys = list(self.group.keys())
            centroids = np.zeros((len(groups),2), dtype = 'int')
            old_centroids = []
            for i in range(len(groups)):
                centroids[i] = (groups[i]['centroid'])
            for group in old_groups:
                old_centroids.append(group['centroid'][-1])
            old_centroids = np.array(old_centroids)
            D = dist.cdist(old_centroids, centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            # loop over the combination of the (row, column) index
            # tuples
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                groupid = keys[row]
                del self.groupID[self.group[groupid]['centroid'][-1]]
                self.group[groupid]['centroid'].append(tuple(centroids[col]))
                self.group[groupid]['box'] = groups[col]['box']
                self.group[groupid]['objects'] = groups[col]['objects']
                self.groupID[tuple(centroids[col])] = groupid
                self.group_disappear[groupid] = 0
                updated_group.append(groupid)
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            for i in range(len(groups)):
                if not i in cols:
                    group = groups[i]
                    if len(group['objects'])  ==1:
                        self.group[self.next_group] = {'box':group['box'],'centroid':[group['centroid']],'objects':group['objects']}
                        self.groupID[group['centroid']] = self.group[self.next_group]
                        self.group_disappear[self.next_group] = 0
                        self.next_group += 1
                        continue
                    self.register_group(group['box'],group['objects'][0], group['objects'][1], group['centroid'])
        for key in self.group.keys():
            if key not in updated_group:
                self.group_disappear[key] += 1


    def update(self, boxes):
        self.enter = 0
        self.leaving = 0
        #if no bboxes in this frame
        if len(boxes) == 0:
            #add 1 to disappeared object
            for key in self.object_disappear:
                self.object_disappear[key] += 1
                if self.object_disappear[key] > self.max_disappear:
                    self.deregister(key)
                return self.object
        # get the centroids of bboxes
        centroids = np.zeros((len(boxes),2), dtype = 'int')
        for (i, b) in enumerate(boxes):
            centroids[i] = self.get_centroid(b)
        #if no object then just register the new centroids
        if len(self.object) == 0:
            for c in centroids:
                self.register(c)
        #if exist object then compare the distance and update them
        else:
            objects = list(self.object.keys())
            object_centroids = []
            for c in self.object.values():
                object_centroids.append(c[-1])
            object_centroids = np.array(object_centroids)
            D = dist.cdist(object_centroids, centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectid = objects[row]
                del self.objectID[tuple(self.object[objectid][-1])]
                self.object[objectid].append(tuple(centroids[col]))
                self.objectID[tuple(centroids[col])] = objectid
                self.object_disappear[objectid] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object = objects[row]
                    self.object_disappear[object] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.object_disappear[object] > self.max_disappear:
                        self.deregister(object)
            else:
                for col in unusedCols:
                    self.register(centroids[col])
                # return the set of trackable objects
            if self.drawed_box:
                for c in self.object.values():
                    #if centroid in drawed box at this frame
                    if in_box(self.drawed_box, c[-1]):
                        #get the path of this object
                        #if the object is not new
                        if len(c) > 1:
                            #if this object was not in the box at last frame then it's entering
                            if not in_box(self.drawed_box, c[-2]):
                                self.enter += 1
                    #if not in the box at this frame
                    else:
                        if len(c) > 1:
                            #if the object was in the box at last frame then it's leaving
                            if in_box(self.drawed_box, c[-2]):
                                self.leaving += 1
            return self.object

# Evaluation Task 1.2
def eval_task1(benchmark, eval_tracking, TP, FP, FN, correct, total):
    #     print("function~")
    #     print(benchmark, eval_tracking)

    # These three params are used to calculate F1 score
    # TP (True Positive): Positive images classified positive
    # FP (False Positive): Negative images classified positive
    # FN (False Negative): Positive images classified negative

    for test in eval_tracking:
        tag_FP = 0
        for standard in benchmark:
            diff_X = abs(test[0] - standard[0])
            diff_Y = abs(test[1] - standard[1])
            if diff_X < 10 and diff_Y < 10:
                # TP
                TP += 1
                correct += 1
                tag_FP = 1
                break
        if tag_FP == 0:
            FP += 1

    for standard in benchmark:
        tag_FN = 0
        total += 1
        for test in eval_tracking:
            diff_X = abs(test[0] - standard[0])
            diff_Y = abs(test[1] - standard[1])
            if diff_X < 10 and diff_Y < 10:
                tag_FN = 1
                break
        if tag_FN == 0:
            FN += 1

    return TP, FP, FN, correct, total

def evaluation(TP, FP, FN, total, correct):
    # Precision: P = TP / TP + FP
    Precision = TP / (TP + FP)

    # Recall: R = TP / TP + FN
    Recall = TP / (TP + FN)

    # F1 - measure: F1 = 2 * P * R / (P + R)
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # Accuracy: correct classified images / total images
    Accuracy = correct / total

    print(f'Precision is {round(Precision, 3)}')
    print(f'Recall is {round(Recall, 3)}')
    print(f'F1 value is {round(F1, 3)}')
    print(f'Accuracy is {round(Accuracy, 3)}')

