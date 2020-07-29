import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt


class LaneDetector(object):

    def __init__(self):
        self.rho = 0.8
        self.theta = np.pi / 180
        self.threshold = 25  # 15
        self.min_line_len = 5 # 10
        self.max_line_gap = 10 # 20
        self.img = None
        self.out_img = None
        self.per_img = None

        self.running = True
        time.sleep(2)

    def update(self):
        while self.running:
            if self.img is not None:

                ysize, xsize = self.img.shape[:2]
                img_size = (self.img.shape[1], self.img.shape[0])

                lines_image_like = np.zeros((ysize, xsize, 3), dtype=np.uint8)

                # apply grayscale
                output_image = self.grayscale(self.img)
                output_image = self.remove_noise(output_image, 3)
                # detect_edges
                output_image = self.canny(output_image, low_threshold=50, high_threshold=150)
                # detect straight lines
                lines_image = self.hough_lines(output_image, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)
                # separate lines to left and right
                left_lines, right_lines = self.separate_lines(lines_image)

                # reject outliers
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

                if len(lines):
                    try:
                        self.draw_lines(lines_image_like, lines, thickness=2)
                        self.out_img = self.combine(lines_image_like, self.img)
                    except:
                        pass

                src_pts = self.make_mask(img_size, 0.65, 0.05, 0.60, 0.3)
                dst_pts = self.make_mask(img_size, 0.1, 0.0, 0.4, 0.36)
                # warped = self.perspective_transform(lines_image_like, np.float32(src_pts), np.float32(dst_pts))
                # binary_warped = self.bin_thresh_img(warped, 1)
                #left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = self.walk_lines(binary_warped)
                # self.plot_lane_walk(binary_warped, left_fit, right_fit, out_img)


    def run_threaded(self,img):
        self.img = img
        if self.out_img is not None:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame', self.out_img)
            cv2.resizeWindow('frame', 800, 800)
            # cv2.namedWindow("orginal", cv2.WINDOW_NORMAL)
            # cv2.imshow('orginal', self.per_img)
            # cv2.resizeWindow('frame', 800, 800)
            cv2.waitKey(1)
        return self.out_img

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

    def draw_rectangle(self,img, corners, color, line_w):
        n = len(corners)
        for i in range(n):
            iS = i
            iE = (i + 1) % n
            a = corners[iS]
            b = corners[iE]
            cv2.line(img, a, b, color, line_w)

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

    def compute_perspective_transform_matrices(self, src, dst):
        """
        Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
        and M_inv is the matrix used to revert the transformed image back to the original one
        """
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        return (M, M_inv)

    def perspective_transform(self,img, src, dst):
        """
        Applies a perspective
        """
        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def thresh_mask(self,img, thresh, val=0.1):
        flt_img = np.zeros_like(img, dtype=float)
        flt_img[(img > thresh[0]) & (img <= thresh[1])] = val
        return flt_img

    def sobel_masks(self,img):
        # construct the Sobel x-axis kernel - diagonal right
        sobelDR = np.array((
            [-1, -1, 0, 0, 1],
            [-2, -1, 0, 0, 2],
            [-2, -1, 0, 0, 2],
            [-1, 0, 0, 0, 2],
            [-1, 0, 0, 1, 1]), dtype="int")

        sobelDL = np.fliplr(sobelDR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diagR = cv2.filter2D(gray, -1, sobelDR)
        diagL = cv2.filter2D(gray, -1, sobelDL)

        thresh = (50, 255)
        diagRth = self.thresh_mask(diagR, thresh, val=0.05)

        thresh = (50, 255)
        diagLth = self.thresh_mask(diagL, thresh, val=0.05)

        return diagRth, diagLth

    def combine_masks(self,S, U, V, R, G, B, DR, DL, thresh=0.1):
        combined = np.zeros_like(S, dtype=np.uint8)
        # combined[(S == 1) | (U == 1) | (R == 1) | (G == 1) | (B == 1)] = 1
        combined[(S + U + R + G + B + DR + DL) > thresh] = 1
        return combined

    def make_mask(self,img_size,  # width, height tuple
                  horizon_perc,  # the upper threshold, as a percent of height
                  bottom_perc,  # the lower thresh, as a percent of height
                  mask_bottom_perc=1.0,  # the lower percent of width
                  mask_top_perc=0.5):  # the upper percent of width

        img_width = img_size[0]
        img_height = img_size[1]
        centerX = img_width / 2

        horizon_y = math.floor(horizon_perc * img_height)
        bottom_y_margin = math.floor(bottom_perc * img_height)
        bottom = img_height - bottom_y_margin
        top = horizon_y

        mask_bottom_left_x = math.floor(centerX - img_width * (mask_bottom_perc * 0.5))
        mask_bottom_right_x = math.floor(centerX + img_width * (mask_bottom_perc * 0.5))
        mask_top_left_x = math.floor(centerX - img_width * (mask_top_perc * 0.5))
        mask_top_right_x = math.floor(centerX + img_width * (mask_top_perc * 0.5))

        mask_points = [(mask_bottom_left_x, bottom),
                       (mask_top_left_x, top),
                       (mask_top_right_x, top),
                       (mask_bottom_right_x, bottom)]

        return mask_points

    def color_thresold(self,img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(hls)

        thresh = (150, 255)
        S = self.thresh_mask(S, thresh)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(rgb)

        thresh = (200, 255)
        R = self.thresh_mask(R, thresh)

        thresh = (200, 255)
        G = self.thresh_mask(G, thresh)

        thresh = (220, 255)
        B = self.thresh_mask(B, thresh)

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(yuv)

        thresh = (50, 100)
        U = self.thresh_mask(U, thresh)

        thresh = (150, 255)
        V = self.thresh_mask(V, thresh)

        DR, DL = self.sobel_masks(img)

        return self.combine_masks(S, U, V, R, G, B, DL, DR)

    def apply_mask(self,img, mask_points):
        ignore_mask_color = 255
        mask = np.zeros_like(img)
        vertices = np.array([mask_points], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        result = cv2.bitwise_and(img, mask)
        return result

    def dir_threshold(self,gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return binary_output

    def bin_thresh_img(self,img, thresh):
        binary_img = np.zeros_like(img)
        binary_img[(img >= thresh)] = 1
        return binary_img

    def walk_lines(self,binary_warped, prevLeftXBase=None, prevRightXBase=None):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 6)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 6)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img

    def update_walk_lines(self,binary_warped, left_fit, right_fit, debug=False):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
                    (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
                    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if debug:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    def sample_curves_to_points(self,left_fit, right_fit, out_img):
        '''
        sample the points of two y paramatized curves in an image
        returns:
            ploty, the array of y values
            leftX, the array of x values sampled from left_fit curve
            rightX, the array of x values sampled from right_fit curve
        '''
        h, w = out_img.shape

        # sample curve
        ploty = np.linspace(0, h - 1, num=h)
        leftX = []
        rightX = []

        for val in range(0, h):
            lX = sample_curve(left_fit, val)
            rX = sample_curve(right_fit, val)
            leftX.append(lX)
            rightX.append(rX)

        leftX = np.array(leftX)
        rightX = np.array(rightX)
        return ploty, leftX, rightX

    def plot_lane_walk(self,binary_warped, left_fit, right_fit, out_img):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        h, w, ch = out_img.shape

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, w)
        plt.ylim(h, 0)

    def shutdown(self):
        self.running = False
        cv2.destroyAllWindows()
        time.sleep(0.2)

