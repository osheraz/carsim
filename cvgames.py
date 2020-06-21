import cv2
import numpy as np
import time



class CvGames(object):

    def __init__(self):
        self.rho = 0.8
        self.theta = np.pi / 180
        self.threshold = 25  # 15
        self.min_line_len = 5 # 10
        self.max_line_gap = 10 # 20
        self.img = None
        self.p_img = None
        self.running = True
        time.sleep(2)

    def update(self):
        while self.running:
            if self.img is not None:

                ysize, xsize = self.img.shape[:2]

                output_image = self.grayscale(self.img)     # apply grayscale
                output_image = self.remove_noise(output_image, 3)
                output_image = self.canny(output_image, low_threshold=50, high_threshold=150)  # detect_edges

                lines_image = self.hough_lines(output_image, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap) # 3D for some reason

                left_lines, right_lines = self.separate_lines(lines_image)

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

                mid = [(filtered_left[0] + filtered_right[0]) / 2,
                       (filtered_left[1] + filtered_right[1]) / 2,
                       (filtered_left[2] + filtered_right[2]) / 2,
                       (filtered_left[3] + filtered_right[3]) / 2,
                       (filtered_left[4] + filtered_right[4]) / 2,]
                mid = np.expand_dims(np.expand_dims(np.array(mid), axis=0), axis=0).tolist()

                lines_image = np.zeros((ysize, xsize, 3), dtype=np.uint8)

                if len(lines):
                    try:
                        self.draw_lines(lines_image, lines, thickness=1)
                        # self.draw_lines(lines_image, mid, thickness=1)
                        final_image = self.combine(lines_image, self.img)
                    except:
                        pass

                self.p_img = final_image

    def run_threaded(self,img):
        self.img = img
        if self.p_img is not None:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame', self.p_img)
            cv2.resizeWindow('frame', 400, 400)
            # cv2.namedWindow("orginal", cv2.WINDOW_NORMAL)
            # cv2.imshow('orginal', img)
            # cv2.resizeWindow('frame', 400, 400)
            cv2.waitKey(1)
        return self.p_img

    def remove_noise(self, image, kernel_size = 5):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def canny(self, image, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(image, low_threshold, high_threshold)

    def draw_lines(self, image, lines, color=[0, 255, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        # lines = self.extrapolate_lines(image, lines)

        for line in lines:
            for x1, y1, x2, y2, slope in line:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    def region_of_interest(self,img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
        The Hough Line Transform is a transform used to detect straight lines.
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        return lines

    def combine(self,img, initial_img, α=0.7, β=1., γ=0.0):
        """
        `img` is An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + γ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)

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

    def lines_linear_regression(self,lines_array):
        x = np.reshape(lines_array[:, [0, 2]], (1, len(lines_array) * 2))[0]
        y = np.reshape(lines_array[:, [1, 3]], (1, len(lines_array) * 2))[0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Solves the equation a x = b
        x = np.array(x)
        y = np.array(x * m + c)
        return x, y, m, c

    def shutdown(self):
        self.running = False
        cv2.destroyAllWindows()
        time.sleep(0.2)

