# *_* coding:utf-8 *_*
# UNSW COMP9517 20T1
# Individual Project
# Author: WENXUN PENG
# zID: z5195349

# Classification of single images with and without the presence of pedestrians


import os
import cv2
import numpy as np


def load_positive_sample(file_path1, file_path2):
    '''
        Load the positive data and return lists of samples and labels
        :param file_path1: ['train_positive_A', 'train_positive_B', 'train_positive_C']
        :param file_path2: ['00000000', '00000001', '00000002', '00000003', '00000004', '00000005', '00000006', '00000007'...]
        :return p_samples, p_labels: List of positive samples and labels
    '''
    print("Loading the positive samples...")
    pwd = os.getcwd()

    path1 = 'Individual_Component'
    path2 = 'train'
    p_labels_len = 0

    p_samples = []
    p_labels = []

    for path3 in file_path1:
        for path4 in file_path2:
            # Extract positive data samples
            pos_dir = os.path.join(pwd, path1, path2, path3, path4)
            if os.path.exists(pos_dir):
                print(f'Positive data path is {pos_dir}')
                pos_files = os.listdir(pos_dir)
                print(f'Positive data path is {len(pos_files)}')

                for f in pos_files:
                    file_path = os.path.join(pos_dir, f)
                    print(f'Loading positive data set: path is {file_path}')
                    if os.path.exists(file_path):
                        p_samples.append(file_path)
                        p_labels.append(1.)

                p_labels_len += len(pos_files)
    print(f'**************************** positive len is {len(p_samples)}***************************************')
    return p_samples, p_labels, p_labels_len


def load_negative_sample(negative_file_path, p_samples, p_labels, p_labels_len):
    '''
    Load the negative data and append the list of samples and labels from positive data,
    And then return list of samples and labels including positive and negative data
    :param negative_file_path: ['00000000', '00000001', '00000002', '00000003', '00000004', '00000005', '00000006', '00000007',
                     '00000008', '00000009', '00000010', '00000011', '00000012', '00000013', '00000014', '00000015',
                     '00000016', '00000017', '00000018', '00000019', '00000020', '00000021']
    :param p_samples: Positive samples
    :param p_labels: Corresponding positive labels
    :param p_labels_len: The length of positive labels
    :return sample, labels: All of the samples and labels
    '''
    print("Loading the negative samples...")
    pwd = os.getcwd()

    path1 = 'Individual_Component'
    path2 = 'train'
    path3 = 'train_negative_A'
    labels_len = p_labels_len

    for path4 in negative_file_path:
        # Extract negative data samples
        neg_dir = os.path.join(pwd, path1, path2, path3, path4)
        if os.path.exists(neg_dir):
            print(f'Negative data path is {neg_dir}')
            neg_files = os.listdir(neg_dir)
            print(f'Negative data path is {len(neg_files)}')

            for f in neg_files:
                file_path = os.path.join(neg_dir, f)
                print(f'Loading negative data set: path is {file_path}')
                if os.path.exists(file_path):
                    p_samples.append(file_path)
                    p_labels.append(-1.)
            labels_len += len(neg_files)

    samples = p_samples
    # labels is transformed to array numpy，the type is np.int32
    labels = np.int32(p_labels)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels


def extract_hog(samples):
    '''
    Extract HOG Feature from training data set and return the feature
    :param samples: All samples from load_positive_sample and load_negative_sample
    :return train: The HOG feature from training data set
    '''
    train = []
    print('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        print(f'Processing {f} {num/total*100:2.1f}%')

        # cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
        #                             histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        # except the winSize, other values are default
        hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        # if not use default params, get poor performance
        # hog = cv2.HOGDescriptor((64, 128), (4, 4), (2, 2), (2, 2), 9)

        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64, 128))
        descriptors = hog.compute(img)
        print(f'hog feature descriptor size: {descriptors.shape}')          # (3780, 1)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train


def get_svm_detector(svm):
    '''
    Derive a list of SVM detectors that can be used in cv2.HOGDescriptor(),
    which is essentially a list of trained SVM support vectors and rho parameters
    :param svm: Trained SVM classifier
    :return: A list of SVM support vectors and rho parameters, which can be used as the SVM detector for cv2.HOGDescriptor()
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def train_svm(train, labels):
    '''
    Train SVM classifier
    :param train: Training data set
    :param labels: Corresponding labels for training set
    :return: SVM detector
    (Note: the SVM in HOGDescriptor of opencv cannot directly use the SVM model of opencv,
    but must export an array of the corresponding format. Thus, we need get_svm_detector to transform the format)
    '''
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)

    # improve the performance
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setType(cv2.ml.SVM_EPS_SVR)

    print('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    print('Training done.')

    # Saving the trained classifier in svm.xml
    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    print(f'Trained SVM classifier is saved as: {model_path}')

    return get_svm_detector(svm)


def test_hog_detect(svm_detector):
    '''
    :param svm_detector: SVM detector for HOGDescriptor
    :return: TP, FP, FN, total_num, correct_num: All of these are used to calculate the Accuracy and F1 score
    '''
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(svm_detector)
    # Opencv comes with a trained pedestrian classifier -> bad performance
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    sum_pos_num = 0
    sum_num = 0
    flag = 0
    test_set_path = ['test_positive', 'test_negative']
    test_set = ['00000000', '00000001', '00000002', '00000003', '00000004', '00000005']
    pwd = os.getcwd()

    path1 = 'Individual_Component'
    path2 = 'test'

    for path3 in test_set_path:
        for path4 in test_set:
            # Extract test set
            test_dir = os.path.join(pwd, path1, path2, path3, path4)
            if os.path.exists(test_dir):
                print(f'Test data path is:{test_dir}')
                test = os.listdir(test_dir)
                print(f'Test samples number:{len(test)}')

                for f in test:
                    file_path = os.path.join(test_dir, f)
                    print(f'Processing {file_path}')
                    img = cv2.imread(file_path, -1)
                    img = cv2.resize(img, (64, 128))
                    rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
                    if rects == ():
                        # negative detected
                        sum_num += 1
                    else:
                        # positive detected
                        sum_pos_num += 1
                        sum_num += 1

        if flag == 0:
            sum_total_pos = sum_num
            sum_pos_num_pos = sum_pos_num
            sum_pos_num = 0
            sum_num = 0
            flag = 1
        else:
            pass

    print(f'Test positive files test positive/total is {sum_pos_num_pos}/{sum_total_pos}.')
    print(f'Test negative files test positive/total is {sum_pos_num}/{sum_num}.')

    # These three params are used to calculate F1 score
    # TP (True Positive): Positive images classified positive
    TP = sum_pos_num_pos
    # FP (False Positive): Negative images classified positive
    FP = sum_pos_num
    # FN (False Negative): Positive images classified negative
    FN = sum_total_pos - sum_pos_num_pos

    # These two params are used to calculate Accuracy
    # Total number of all test data set
    total_num = sum_total_pos + sum_num
    # Correct classified number of test data set
    correct_num = sum_pos_num_pos + (sum_num - sum_pos_num)
    return TP, FP, FN, total_num, correct_num


def evaluation(TP, FP, FN, total, correct):
    '''
    Evaluation of the performance of SVM classifier
    :param: Corresponding to the return of test_hog_detect
    :return: None
    '''
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


if __name__ == '__main__':
    pos_file_path1 = ['train_positive_A', 'train_positive_B', 'train_positive_C']
    pos_file_path2 = ['00000000', '00000001', '00000002', '00000003', '00000004', '00000005', '00000006', '00000007']
    neg_file_path = ['00000000', '00000001', '00000002', '00000003', '00000004', '00000005', '00000006', '00000007',
                     '00000008', '00000009', '00000010', '00000011', '00000012', '00000013', '00000014', '00000015',
                     '00000016', '00000017', '00000018', '00000019', '00000020', '00000021']
    # pos_file_path2 = ['00000000']
    # neg_file_path = ['00000000']
    # Load training data set (positive and negative)
    pos_samples, pos_labels, pos_labels_len = load_positive_sample(pos_file_path1, pos_file_path2)
    samples, labels = load_negative_sample(neg_file_path, pos_samples, pos_labels, pos_labels_len)

    # Extract HOG feature and then train SVM classifier
    train = extract_hog(samples)
    print(f'Size of feature vectors of samples: {train.shape}')
    print(f'Size of labels of samples: {labels.shape}')
    svm_detector = train_svm(train, labels)

    # Using test data to verify the performance of SVM classifier and then evaluate.
    TP, FP, FN, total_num, correct_num = test_hog_detect(svm_detector)
    evaluation(TP, FP, FN, total_num, correct_num)
