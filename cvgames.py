import cv2
import numpy as np
import time



class CvGames(object):

    def __init__(self):
        self.rho = 0.8
        self.theta = np.pi / 180
        self.threshold = 25
        self.min_line_len = 5
        self.max_line_gap = 10
        self.img = None
        self.p_img = None
        self.running = True
        time.sleep(2)

    def update(self):
        while self.running:
            if self.img is not None:
                obs = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                edges = self.detect_edges(obs, low_threshold=50, high_threshold=150)

                hl = self.hough_lines(edges, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)

                left_lines, right_lines = self.separate_lines(hl)

                filtered_right, filtered_left = [], []
                if len(left_lines):
                    filtered_left = self.reject_outliers(left_lines, cutoff=(-30.0, -0.1), lane='left')
                if len(right_lines):
                    filtered_right = self.reject_outliers(right_lines, cutoff=(0.1, 30.0), lane='right')

                lines = []
                if len(filtered_left) and len(filtered_right):
                    lines = np.expand_dims(np.vstack((np.array(filtered_left), np.array(filtered_right))), axis=0).tolist()
                elif len(filtered_left):
                    lines = np.expand_dims(np.expand_dims(np.array(filtered_left), axis=0), axis=0).tolist()
                elif len(filtered_right):
                    lines = np.expand_dims(np.expand_dims(np.array(filtered_right), axis=0), axis=0).tolist()

                ret_img = np.zeros((80, 80))

                if len(lines):
                    try:
                        self.draw_lines(ret_img, lines, thickness=1)
                    except:
                        pass

                self.p_img = ret_img

    def run_threaded(self,img):
        # self.running = True
        self.img = img
        if self.p_img is not None:
            cv2.imshow('frame', self.p_img)
            cv2.waitKey(1)
        return self.p_img

    def remove_noise(self, image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def discard_colors(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_edges(self, image, low_threshold, high_threshold):
        return cv2.Canny(image, low_threshold, high_threshold)

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2, slope in line:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    def hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        return lines

    def slope(self, x1, y1, x2, y2):
        try:
            return (y1 - y2) / (x1 - x2)
        except:
            return 0

    def separate_lines(self, lines):
        right = []
        left = []

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                m = self.slope(x1, y1, x2, y2)
                if m >= 0:
                    right.append([x1, y1, x2, y2, m])
                else:
                    left.append([x1, y1, x2, y2, m])
        return left, right

    def reject_outliers(self, data, cutoff, threshold=0.08, lane='left'):
        data = np.array(data)
        data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
        try:
            if lane == 'left':
                return data[np.argmin(data, axis=0)[-1]]
            elif lane == 'right':
                return data[np.argmax(data, axis=0)[-1]]
        except:
            return []

    def extend_point(self, x1, y1, x2, y2, length):
        line_len = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        x = x2 + (x2 - x1) / line_len * length
        y = y2 + (y2 - y1) / line_len * length
        return x, y

    def shutdown(self):
        self.running = False
        cv2.destroyAllWindows()
        time.sleep(0.2)
